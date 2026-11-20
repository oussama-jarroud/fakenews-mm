"""Phase A2: Multimodal Feature Extraction with Frozen CLIP (ViT-B/32).

Extracts and L2-normalizes text and image embeddings in float16.
Embeddings are cached to disk so subsequent fusion models train in seconds.

Target Environment: Kaggle Notebook with GPU Accelerator (T4 or P100).
Input: Output from Notebook N1 (N1 prepare-data) containing table.parquet and images_v1/.
Output: text_emb.npy, image_emb.npy, ids.npy, metadata.parquet.
"""

import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ==============================================================================
# 1. Path Auto-Resolution (robust across multiple mounted datasets)
# ==============================================================================
def find_input_paths() -> Tuple[str, str]:
    """Auto-detects table.parquet and images_v1, prioritizing paired notebook outputs."""
    tables = []
    image_dirs = []

    for base in ["/kaggle/input", "/kaggle/working", "./data", "."]:
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            if "table.parquet" in files:
                tables.append(os.path.join(root, "table.parquet"))
            if "images_v1" in dirs:
                image_dirs.append(os.path.join(root, "images_v1"))

    print(f"Detected Tables: {tables}")
    print(f"Detected Image Dirs: {image_dirs}")

    chosen_table = None
    chosen_img_dir = None

    # Priority 1: Match images_v1 and table.parquet from the SAME notebook output
    if image_dirs:
        for img_dir in image_dirs:
            parent = os.path.dirname(img_dir)
            candidate_table = os.path.join(parent, "table.parquet")
            if os.path.exists(candidate_table):
                chosen_table = candidate_table
                chosen_img_dir = img_dir
                break
        
        # If no table directly beside images_v1, pick first images_v1
        if chosen_img_dir is None:
            chosen_img_dir = image_dirs[0]

    # Priority 2: If table not paired yet, select the table that mentions prepare-data
    if chosen_table is None and tables:
        for t in tables:
            if "prepare" in t.lower() or "n1" in t.lower():
                chosen_table = t
                break
        if chosen_table is None:
            chosen_table = tables[0]

    # Diagnostic print if either is missing
    if chosen_img_dir is None or chosen_table is None:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC: Directory contents of /kaggle/input:")
        if os.path.exists("/kaggle/input"):
            for item in os.listdir("/kaggle/input"):
                p = os.path.join("/kaggle/input", item)
                if os.path.isdir(p):
                    try:
                        sub = os.listdir(p)[:6]
                        print(f"  📁 {item}/ -> {sub}")
                    except Exception as e:
                        print(f"  📁 {item}/ -> error reading: {e}")
                else:
                    print(f"  📄 {item}")
        print("=" * 60 + "\n")
        raise FileNotFoundError(
            "Could not locate both 'images_v1' and 'table.parquet'!\n"
            "Please check the diagnostic listing above to ensure 'N1 prepare-data' is attached as Input."
        )

    return chosen_table, chosen_img_dir


# ==============================================================================
# 2. PyTorch Multimodal Dataset
# ==============================================================================
class MultimodalCLIPDataset(Dataset):
    """Pairs Reddit titles with downloaded images for batched CLIP forward passes."""

    def __init__(self, df: pd.DataFrame, image_dir: str):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.ids = self.df["id"].astype(str).tolist()
        
        # Robust column resolution: our parquet contract renamed clean_title to 'text'
        text_col = "text" if "text" in self.df.columns else ("clean_title" if "clean_title" in self.df.columns else "title")
        self.texts = [
            str(t) if pd.notna(t) and str(t).strip() else "news image"
            for t in self.df[text_col]
        ]
        self.img_paths = [
            os.path.join(self.image_dir, f"{i}.jpg") for i in self.ids
        ]
        self.fallback_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, str, Image.Image]:
        item_id = self.ids[idx]
        text = self.texts[idx]
        img_path = self.img_paths[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = self.fallback_img

        return item_id, text, image


def clip_collate_fn(batch, processor: CLIPProcessor):
    """Batched tokenization and image resizing via CLIPProcessor."""
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


# ==============================================================================
# 3. Main Extraction Pipeline
# ==============================================================================
def run_embedding_pipeline(
    output_dir: str = "/kaggle/working",
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 128,
    num_workers: int = 4,
):
    print("=" * 70)
    print("PHASE A2: CLIP EMBEDDING EXTRACTION")
    print(f"Model: {model_name}")
    print("=" * 70)

    # Device check
    if not torch.cuda.is_available():
        print("\n[WARNING] CUDA is NOT detected! You are running on CPU.")
        print("To run fast on Kaggle, open the right sidebar -> 'Notebook options' -> 'Accelerator' -> 'GPU T4' or 'P100'.")
        print("Continuing on CPU (this will be slow)...\n")
        device = "cpu"
    else:
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Found GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")

    # Find paths
    table_path, image_dir = find_input_paths()
    print(f"Selected Metadata Table: {table_path}")
    print(f"Selected Images Directory: {image_dir}")

    # Read table
    print("\nReading metadata table...")
    df = pd.read_parquet(table_path)
    total_samples = len(df)
    print(f"Loaded {total_samples:,} rows from {table_path}")
    print(f"Split distribution:\n{df['split'].value_counts()}")

    # Initialize CLIP
    print(f"\nLoading CLIP model ({model_name})...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    # DataLoader
    dataset = MultimodalCLIPDataset(df, image_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: clip_collate_fn(b, processor),
        pin_memory=(device == "cuda"),
    )

    # Embedding extraction
    all_ids: List[str] = []
    text_emb_list: List[np.ndarray] = []
    image_emb_list: List[np.ndarray] = []

    print(f"\nStarting batch inference on {total_samples:,} items (batch_size={batch_size})...")
    start_time = time.time()

    def to_feature_tensor(output):
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state[:, 0, :]
        return output

    with torch.no_grad():
        for batch_ids, inputs in tqdm(loader, desc="Extracting embeddings", unit="batch"):
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Mixed precision for 2x faster GPU throughput
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
                # Text forward
                text_out = model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                text_feats = to_feature_tensor(text_out)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

                # Image forward
                image_out = model.get_image_features(
                    pixel_values=inputs["pixel_values"]
                )
                image_feats = to_feature_tensor(image_out)
                image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

            all_ids.extend(batch_ids)
            text_emb_list.append(text_feats.cpu().to(torch.float16).numpy())
            image_emb_list.append(image_feats.cpu().to(torch.float16).numpy())

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\nInference completed in {elapsed_minutes:.2f} minutes!")

    # Concatenate results
    print("Consolidating embedding matrices...")
    ids_arr = np.array(all_ids, dtype=object)
    text_emb_arr = np.concatenate(text_emb_list, axis=0)
    image_emb_arr = np.concatenate(image_emb_list, axis=0)

    # Integrity Assertions
    print("\nRunning verification assertions...")
    assert len(ids_arr) == total_samples, f"ID count mismatch: {len(ids_arr)} vs {total_samples}"
    assert text_emb_arr.shape == (total_samples, 512), f"Text shape mismatch: {text_emb_arr.shape}"
    assert image_emb_arr.shape == (total_samples, 512), f"Image shape mismatch: {image_emb_arr.shape}"
    assert not np.isnan(text_emb_arr).any(), "NaN values found in text embeddings!"
    assert not np.isnan(image_emb_arr).any(), "NaN values found in image embeddings!"

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    out_ids = os.path.join(output_dir, "ids.npy")
    out_text = os.path.join(output_dir, "text_emb.npy")
    out_img = os.path.join(output_dir, "image_emb.npy")
    out_meta = os.path.join(output_dir, "metadata.parquet")

    print(f"\nSaving arrays to '{output_dir}'...")
    np.save(out_ids, ids_arr)
    np.save(out_text, text_emb_arr)
    np.save(out_img, image_emb_arr)
    df.to_parquet(out_meta, index=False)

    size_ids = os.path.getsize(out_ids) / (1024**2)
    size_text = os.path.getsize(out_text) / (1024**2)
    size_img = os.path.getsize(out_img) / (1024**2)
    size_meta = os.path.getsize(out_meta) / (1024**2)

    print("=" * 70)
    print("PHASE A2 COMPLETE: EMBEDDINGS CACHED SUCCESSFULLY")
    print(f"  - {out_ids} ({size_ids:.1f} MB)")
    print(f"  - {out_text} ({size_text:.1f} MB, shape: {text_emb_arr.shape}, dtype: float16)")
    print(f"  - {out_img} ({size_img:.1f} MB, shape: {image_emb_arr.shape}, dtype: float16)")
    print(f"  - {out_meta} ({size_meta:.1f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    run_embedding_pipeline()
