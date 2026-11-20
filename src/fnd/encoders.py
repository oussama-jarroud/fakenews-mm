"""Feature extraction module using pretrained CLIP (openai/clip-vit-base-patch32).

Extracts and L2-normalizes pooled text and vision representations in float16.
Embeddings are cached to disk so subsequent fusion models train in seconds.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class MultimodalDataset(Dataset):
    """PyTorch Dataset for paired headline text and news images."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        processor: CLIPProcessor,
        text_col: str = "clean_title",
        id_col: str = "id",
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.text_col = text_col
        self.id_col = id_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, str, Image.Image]:
        row = self.df.iloc[idx]
        item_id = str(row[self.id_col])
        text = str(row[self.text_col]) if pd.notna(row[self.text_col]) else ""
        
        img_path = os.path.join(self.image_dir, f"{item_id}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback for unexpected I/O error: black 224x224 placeholder
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        return item_id, text, image


def collate_fn(batch, processor: CLIPProcessor):
    """Collate function that runs CLIP processor tokenization and image transformations."""
    ids = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    images = [b[2] for b in batch]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    return ids, inputs


def extract_clip_embeddings(
    table_path: str,
    image_dir: str,
    output_dir: str,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 128,
    num_workers: int = 2,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and normalize CLIP vision and text embeddings from dataset."""
    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    # Load table
    print(f"Loading metadata from {table_path}...")
    df = pd.read_parquet(table_path)
    total_samples = len(df)
    print(f"Total samples to process: {total_samples:,}")

    # Load CLIP model and processor
    print(f"Loading model '{model_name}'...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    dataset = MultimodalDataset(df, image_dir, processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, processor),
        pin_memory=(device == "cuda"),
    )

    all_ids: List[str] = []
    text_embeddings: List[np.ndarray] = []
    image_embeddings: List[np.ndarray] = []

    def to_feature_tensor(output):
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state[:, 0, :]
        return output

    print(f"Extracting embeddings with batch size {batch_size}...")
    with torch.no_grad():
        for batch_ids, inputs in tqdm(loader, desc="Extracting CLIP embeddings"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
                # Text features
                text_out = model.get_text_features(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask"),
                )
                text_features = to_feature_tensor(text_out)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Image features
                image_out = model.get_image_features(
                    pixel_values=inputs.get("pixel_values")
                )
                image_features = to_feature_tensor(image_out)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_ids.extend(batch_ids)
            text_embeddings.append(text_features.cpu().to(torch.float16).numpy())
            image_embeddings.append(image_features.cpu().to(torch.float16).numpy())

    # Concatenate
    print("Concatenating output arrays...")
    ids_arr = np.array(all_ids, dtype=object)
    text_emb_arr = np.concatenate(text_embeddings, axis=0)
    image_emb_arr = np.concatenate(image_embeddings, axis=0)

    # Sanity checks
    assert len(ids_arr) == total_samples, f"Expected {total_samples} IDs, got {len(ids_arr)}"
    assert text_emb_arr.shape == (total_samples, 512), f"Unexpected shape {text_emb_arr.shape}"
    assert image_emb_arr.shape == (total_samples, 512), f"Unexpected shape {image_emb_arr.shape}"
    assert not np.isnan(text_emb_arr).any(), "NaN found in text embeddings!"
    assert not np.isnan(image_emb_arr).any(), "NaN found in image embeddings!"

    # Save
    out_text_path = os.path.join(output_dir, "text_emb.npy")
    out_img_path = os.path.join(output_dir, "image_emb.npy")
    out_ids_path = os.path.join(output_dir, "ids.npy")
    out_meta_path = os.path.join(output_dir, "table.parquet")

    print(f"Saving to {output_dir}...")
    np.save(out_text_path, text_emb_arr)
    np.save(out_img_path, image_emb_arr)
    np.save(out_ids_path, ids_arr)
    df.to_parquet(out_meta_path, index=False)

    print("Phase A2 feature extraction complete.")
    print(f"  - text_emb: {text_emb_arr.shape} ({out_text_path})")
    print(f"  - image_emb: {image_emb_arr.shape} ({out_img_path})")
    print(f"  - ids: {ids_arr.shape} ({out_ids_path})")

    return text_emb_arr, image_emb_arr, ids_arr
