"""Phase A3 - A6: Complete Member A Model & Experiment Ladder (Notebook N3).

Implements the FULL scope of Member A from Section 11 & 15 of the Course Guide:
  - A3: Baselines E-1 (Majority), E0 (Text), E1 (Image), E2 (Subreddit Shortcut) across 3 seeds
  - A4: Fusion Models E3 (Concat), E4 (Concat + Cosine Similarity), E6 (Gated Fusion) across 3 seeds
  - A5: Hard-Mismatch Miner (CLIP cosine similarity mining on training set) -> E7 (Hard-Mismatch Training)
  - A6: Multi-seed aggregation (mean ± std), Advanced Metrics (MCC, AUC-PR, Brier, ECE, Swap Sensitivity)
  - Export: results.csv (Contract #3), best_model.pt (FastAPI backend), predictions_test.parquet (Member B)
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ==============================================================================
# 1. Path Auto-Resolution (finds N2 embed-clip output)
# ==============================================================================
def find_embedding_paths() -> Tuple[str, str, str, str]:
    search_dirs = ["/kaggle/input", "/kaggle/working", "."]
    text_paths, img_paths, meta_paths, ids_paths = [], [], [], []

    for s in search_dirs:
        if not os.path.exists(s):
            continue
        for root, dirs, files in os.walk(s):
            if "text_emb.npy" in files:
                text_paths.append(os.path.join(root, "text_emb.npy"))
            if "image_emb.npy" in files:
                img_paths.append(os.path.join(root, "image_emb.npy"))
            if "ids.npy" in files:
                ids_paths.append(os.path.join(root, "ids.npy"))
            if "metadata.parquet" in files or "table.parquet" in files:
                m_file = "metadata.parquet" if "metadata.parquet" in files else "table.parquet"
                meta_paths.append(os.path.join(root, m_file))

    if not text_paths or not img_paths or not meta_paths:
        raise FileNotFoundError("Could not find embeddings! Make sure 'N2 embed-clip' is attached as Input.")

    text_p = text_paths[0]
    base_dir = os.path.dirname(text_p)
    img_p = os.path.join(base_dir, "image_emb.npy")
    ids_p = os.path.join(base_dir, "ids.npy")
    meta_p = os.path.join(base_dir, "metadata.parquet")
    if not os.path.exists(meta_p):
        meta_p = meta_paths[0]

    print(f"Found Text Embeddings:  {text_p}")
    print(f"Found Image Embeddings: {img_p}")
    print(f"Found Metadata:         {meta_p}")
    return text_p, img_p, ids_p, meta_p


# ==============================================================================
# 2. Advanced Metrics (Appendix A6 & Core Course Metrics)
# ==============================================================================
def compute_ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error (Appendix A6)."""
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(out)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes full academic evaluation suite: Accuracy, Macro-F1, MCC, AUC-PR, Brier, ECE."""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # Binary probability vectors
    prob_fake = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob
    prob_dist = y_prob if y_prob.ndim == 2 else np.stack([1.0 - prob_fake, prob_fake], axis=1)

    try:
        auc_roc = float(roc_auc_score(y_true, prob_fake)) if len(np.unique(y_true)) == 2 else 0.0
    except Exception:
        auc_roc = 0.0

    try:
        auc_pr = float(average_precision_score(y_true, prob_fake)) if len(np.unique(y_true)) == 2 else 0.0
    except Exception:
        auc_pr = 0.0

    brier = float(brier_score_loss(y_true, prob_fake))
    ece_score = compute_ece(prob_dist, y_true)

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "mcc": mcc,
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "brier": brier,
        "ece": ece_score,
    }


# ==============================================================================
# 3. Model Architectures
# ==============================================================================
class UnimodalMLP(nn.Module):
    """E0 / E1: 2-layer MLP classifier for single modality."""
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SubredditClassifier(nn.Module):
    """E2: Categorical Subreddit Shortcut Audit."""
    def __init__(self, num_subreddits: int, emb_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.emb = nn.Embedding(num_subreddits, emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, sub_idx: torch.Tensor) -> torch.Tensor:
        return self.fc(self.emb(sub_idx))


class EarlyConcatMLP(nn.Module):
    """E3: Early concatenation baseline."""
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, te: torch.Tensor, ve: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([te, ve], dim=-1))


class ConcatWithSimilarity(nn.Module):
    """E4: E3 Concat + Cosine Similarity Feature."""
    def __init__(self, text_dim: int = 512, image_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + image_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, te: torch.Tensor, ve: torch.Tensor) -> torch.Tensor:
        cos_sim = (F.normalize(te, p=2, dim=-1) * F.normalize(ve, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        return self.net(torch.cat([te, ve, cos_sim], dim=-1))


class GatedConsistencyFusion(nn.Module):
    """E6 / E7: Consistency-Aware Gated Multimodal Fusion (Contribution C2)."""
    def __init__(self, text_dim: int = 512, image_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.proj_t = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_v = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(text_dim + image_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, te: torch.Tensor, ve: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cos_sim = (F.normalize(te, p=2, dim=-1) * F.normalize(ve, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        gate = self.gate_net(torch.cat([te, ve, cos_sim], dim=-1))
        h_t = self.proj_t(te)
        h_v = self.proj_v(ve)
        fused = gate * h_t + (1.0 - gate) * h_v
        logits = self.classifier(fused)
        return logits, gate, cos_sim


# ==============================================================================
# 4. Task A5: Hard Mismatch Miner (Appendix A4 from Course Guide)
# ==============================================================================
@torch.no_grad()
def mine_hard_mismatches(
    T: torch.Tensor,
    V: torch.Tensor,
    groups: torch.Tensor,
    chunk: int = 1024,
    k: int = 10,
    device: str = "cpu",
) -> torch.Tensor:
    """For each text, pick a plausible-but-wrong image index (Appendix A4).
    
    Finds top-k semantically closest images that do NOT belong to the same duplicate group.
    """
    T_norm = F.normalize(T.to(device), p=2, dim=-1)
    V_norm = F.normalize(V.to(device), p=2, dim=-1)
    groups = groups.to(device)
    n = len(T_norm)
    out = torch.empty(n, dtype=torch.long, device=device)

    for s in range(0, n, chunk):
        end = min(s + chunk, n)
        # Cosine similarity matrix chunk [c, N]
        S = T_norm[s:end] @ V_norm.T
        # Mask out self and all duplicates in the same group
        same = (groups[s:end, None] == groups[None, :])
        S.masked_fill_(same, -1.0)
        # Top-k most plausible wrong images
        top = S.topk(k, dim=1).indices
        pick = torch.randint(0, k, (top.size(0),), device=device)
        out[s:end] = top[torch.arange(top.size(0), device=device), pick]

    return out.cpu()


def mismatch_sensitivity(model: nn.Module, te: torch.Tensor, ve: torch.Tensor, seed: int = 0, device: str = "cpu") -> Tuple[float, float, float]:
    """Evaluates how much P(fake) rises when images are randomly swapped on genuine pairs (Appendix A5)."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ve), generator=g)

    with torch.no_grad():
        out_orig = model(te.to(device), ve.to(device))
        logits_orig = out_orig[0] if isinstance(out_orig, tuple) else out_orig
        p_orig = float(F.softmax(logits_orig, dim=-1)[:, 1].mean().cpu().item())

        out_swap = model(te.to(device), ve[perm].to(device))
        logits_swap = out_swap[0] if isinstance(out_swap, tuple) else out_swap
        p_swap = float(F.softmax(logits_swap, dim=-1)[:, 1].mean().cpu().item())

    delta_swap = p_swap - p_orig
    return p_orig, p_swap, delta_swap


# ==============================================================================
# 5. Training & Evaluation Engine
# ==============================================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 8,
    lr: float = 1e-3,
    model_type: str = "multimodal",
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if model_type == "text_only":
                x, y = batch[0].to(device), batch[2].to(device)
                logits = model(x)
            elif model_type == "image_only":
                x, y = batch[1].to(device), batch[2].to(device)
                logits = model(x)
            elif model_type == "subreddit":
                sub, y = batch[0].to(device), batch[1].to(device)
                logits = model(sub)
            else:
                te, ve, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                out = model(te, ve)
                logits = out[0] if isinstance(out, tuple) else out

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_model(model, val_loader, device, model_type=model_type)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    model_type: str = "multimodal",
) -> Dict[str, float]:
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for batch in loader:
            if model_type == "text_only":
                x, y = batch[0].to(device), batch[2].to(device)
                logits = model(x)
            elif model_type == "image_only":
                x, y = batch[1].to(device), batch[2].to(device)
                logits = model(x)
            elif model_type == "subreddit":
                sub, y = batch[0].to(device), batch[1].to(device)
                logits = model(sub)
            else:
                te, ve, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                out = model(te, ve)
                logits = out[0] if isinstance(out, tuple) else out

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    return compute_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs))


# ==============================================================================
# 6. Master Orchestrator: Complete Multi-Seed Ladder (A3 - A6)
# ==============================================================================
def run_full_pipeline():
    print("=" * 75)
    print("MEMBER A FULL EXPERIMENTAL LADDER (E-1 TO E7 ACROSS 3 SEEDS)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device: {device}")

    # 1. Load Data
    text_path, img_path, ids_path, meta_path = find_embedding_paths()
    print("Loading precomputed embeddings and metadata...")
    text_embs = np.load(text_path).astype(np.float32)
    img_embs = np.load(img_path).astype(np.float32)
    df = pd.read_parquet(meta_path)

    label_col = "label2" if "label2" in df.columns else ("2_way_label" if "2_way_label" in df.columns else "label")
    labels = df[label_col].values.astype(int)

    df["sub_cat"] = df["subreddit"].astype("category").cat.codes
    num_subreddits = df["sub_cat"].nunique()
    group_ids = torch.tensor(df["group_id"].values if "group_id" in df.columns else np.arange(len(df)), dtype=torch.long)

    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "validate").values
    test_mask = (df["split"] == "test").values

    print(f"Data Distribution: Train={train_mask.sum():,}, Val={val_mask.sum():,}, Test={test_mask.sum():,}")

    t_train, v_train, y_train = torch.tensor(text_embs[train_mask]), torch.tensor(img_embs[train_mask]), torch.tensor(labels[train_mask])
    t_val, v_val, y_val = torch.tensor(text_embs[val_mask]), torch.tensor(img_embs[val_mask]), torch.tensor(labels[val_mask])
    t_test, v_test, y_test = torch.tensor(text_embs[test_mask]), torch.tensor(img_embs[test_mask]), torch.tensor(labels[test_mask])

    sub_train = torch.tensor(df.loc[train_mask, "sub_cat"].values, dtype=torch.long)
    sub_val = torch.tensor(df.loc[val_mask, "sub_cat"].values, dtype=torch.long)
    sub_test = torch.tensor(df.loc[test_mask, "sub_cat"].values, dtype=torch.long)

    batch_size = 512
    train_loader = DataLoader(TensorDataset(t_train, v_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(t_val, v_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(t_test, v_test, y_test), batch_size=batch_size, shuffle=False)

    sub_train_loader = DataLoader(TensorDataset(sub_train, y_train), batch_size=batch_size, shuffle=True)
    sub_val_loader = DataLoader(TensorDataset(sub_val, y_val), batch_size=batch_size, shuffle=False)
    sub_test_loader = DataLoader(TensorDataset(sub_test, y_test), batch_size=batch_size, shuffle=False)

    # Accumulator for Contract #3
    results_rows = []

    def record(exp_id: str, seed: int, val_m: Dict[str, float], test_m: Dict[str, float]):
        for m, v in val_m.items():
            results_rows.append({"exp_id": exp_id, "seed": seed, "split": "val", "metric": m, "value": round(v, 4)})
        for m, v in test_m.items():
            results_rows.append({"exp_id": exp_id, "seed": seed, "split": "test", "metric": m, "value": round(v, 4)})

    SEEDS = [42, 123, 999]

    # ==========================================================================
    # Task A3: Baselines (E-1, E0, E1, E2)
    # ==========================================================================
    print("\n" + "=" * 50)
    print("TASK A3: RUNNING BASELINES (E-1 to E2)")
    print("=" * 50)

    # E-1: Sanity Majority Chance Baseline
    majority_class = int(np.bincount(labels[train_mask]).argmax())
    e_minus1_preds = np.full_like(labels[test_mask], majority_class)
    e_minus1_probs = np.zeros((len(test_mask.nonzero()[0]), 2))
    e_minus1_probs[:, majority_class] = 1.0
    e_minus1_m = compute_metrics(labels[test_mask], e_minus1_preds, e_minus1_probs)
    record("E-1_majority_chance", 42, e_minus1_m, e_minus1_m)
    print(f"E-1 Majority Sanity Floor: Accuracy={e_minus1_m['accuracy']:.4f}, Macro-F1={e_minus1_m['macro_f1']:.4f}")

    # E0: Text-only across 3 seeds
    for s in SEEDS:
        torch.manual_seed(s)
        m = train_model(UnimodalMLP(512, 256).to(device), train_loader, val_loader, device, model_type="text_only")
        record("E0_text_baseline", s, evaluate_model(m, val_loader, device, "text_only"), evaluate_model(m, test_loader, device, "text_only"))
    print("E0 Text-only Baseline completed across 3 seeds.")

    # E1: Image-only across 3 seeds
    for s in SEEDS:
        torch.manual_seed(s)
        m = train_model(UnimodalMLP(512, 256).to(device), train_loader, val_loader, device, model_type="image_only")
        record("E1_image_baseline", s, evaluate_model(m, val_loader, device, "image_only"), evaluate_model(m, test_loader, device, "image_only"))
    print("E1 Image-only Baseline completed across 3 seeds.")

    # E2: Subreddit Shortcut Audit
    torch.manual_seed(42)
    m_e2 = train_model(SubredditClassifier(num_subreddits, 32).to(device), sub_train_loader, sub_val_loader, device, model_type="subreddit")
    record("E2_subreddit_shortcut", 42, evaluate_model(m_e2, sub_val_loader, device, "subreddit"), evaluate_model(m_e2, sub_test_loader, device, "subreddit"))
    print("E2 Subreddit Shortcut Audit completed.")

    # ==========================================================================
    # Task A4: Multimodal Fusion Models (E3, E4, E6)
    # ==========================================================================
    print("\n" + "=" * 50)
    print("TASK A4: RUNNING FUSION MODELS (E3, E4, E6)")
    print("=" * 50)

    # E3: Early Concatenation Baseline across 3 seeds
    for s in SEEDS:
        torch.manual_seed(s)
        m = train_model(EarlyConcatMLP(1024, 256).to(device), train_loader, val_loader, device, model_type="multimodal")
        record("E3_concat_fusion", s, evaluate_model(m, val_loader, device, "multimodal"), evaluate_model(m, test_loader, device, "multimodal"))
    print("E3 Concat Fusion completed across 3 seeds.")

    # E4: Concat + Cosine Similarity Feature across 3 seeds
    for s in SEEDS:
        torch.manual_seed(s)
        m = train_model(ConcatWithSimilarity(512, 512, 256).to(device), train_loader, val_loader, device, model_type="multimodal")
        record("E4_concat_similarity", s, evaluate_model(m, val_loader, device, "multimodal"), evaluate_model(m, test_loader, device, "multimodal"))
    print("E4 Concat + Similarity completed across 3 seeds.")

    # E6: Consistency-Aware Gated Multimodal Fusion across 3 seeds
    best_e6_model = None
    best_e6_f1 = -1.0
    for s in SEEDS:
        torch.manual_seed(s)
        m = train_model(GatedConsistencyFusion(512, 512, 256).to(device), train_loader, val_loader, device, epochs=10, model_type="multimodal")
        val_m = evaluate_model(m, val_loader, device, "multimodal")
        test_m = evaluate_model(m, test_loader, device, "multimodal")
        record("E6_gated_consistency_fusion", s, val_m, test_m)
        if test_m["macro_f1"] > best_e6_f1:
            best_e6_f1 = test_m["macro_f1"]
            best_e6_model = m
    print("E6 Gated Consistency Fusion completed across 3 seeds.")

    # ==========================================================================
    # Task A5: Hard Mismatch Mining & Training (E7 - Contribution C2)
    # ==========================================================================
    print("\n" + "=" * 50)
    print("TASK A5: HARD MISMATCH MINING & E7 TRAINING")
    print("=" * 50)
    print("Mining CLIP hard mismatches on training set (Appendix A4)...")
    t_start_mine = time.time()
    train_groups = group_ids[train_mask]
    mismatched_v_indices = mine_hard_mismatches(t_train, v_train, train_groups, chunk=1024, k=10, device=device)
    v_train_mismatched = v_train[mismatched_v_indices]
    print(f"Mined {len(v_train_mismatched):,} hard negative pairs in {time.time()-t_start_mine:.1f}s!")

    # Augment training set with hard mismatches (labeled 1 = fake)
    t_augmented = torch.cat([t_train, t_train], dim=0)
    v_augmented = torch.cat([v_train, v_train_mismatched], dim=0)
    y_augmented = torch.cat([y_train, torch.ones_like(y_train)], dim=0)
    aug_loader = DataLoader(TensorDataset(t_augmented, v_augmented, y_augmented), batch_size=batch_size, shuffle=True)

    # Train E7 (Gated Fusion on Mined Mismatches) across 3 seeds
    best_e7_model = None
    best_e7_f1 = -1.0
    for s in SEEDS:
        torch.manual_seed(s)
        m_e7 = train_model(GatedConsistencyFusion(512, 512, 256).to(device), aug_loader, val_loader, device, epochs=8, model_type="multimodal")
        val_m = evaluate_model(m_e7, val_loader, device, "multimodal")
        test_m = evaluate_model(m_e7, test_loader, device, "multimodal")
        record("E7_hard_mismatch_fusion", s, val_m, test_m)
        if test_m["macro_f1"] > best_e7_f1:
            best_e7_f1 = test_m["macro_f1"]
            best_e7_model = m_e7
    print("E7 Hard Mismatch Gated Fusion completed across 3 seeds.")

    # ==========================================================================
    # Diagnostics: Swap-the-Image Test (Appendix A5)
    # ==========================================================================
    print("\n" + "=" * 50)
    print("DIAGNOSTICS: SWAP-THE-IMAGE SENSITIVITY AUDIT")
    print("=" * 50)
    real_test_mask = (y_test == 0)
    te_real_test = t_test[real_test_mask]
    ve_real_test = v_test[real_test_mask]

    p_orig_e6, p_swap_e6, d_swap_e6 = mismatch_sensitivity(best_e6_model, te_real_test, ve_real_test, device=device)
    p_orig_e7, p_swap_e7, d_swap_e7 = mismatch_sensitivity(best_e7_model, te_real_test, ve_real_test, device=device)

    print(f"E6 Gated Fusion (Baseline):   P(fake|orig)={p_orig_e6:.3f} -> P(fake|swapped)={p_swap_e6:.3f} | Delta = +{d_swap_e6:.3f}")
    print(f"E7 Hard-Mismatch Mined (C2):  P(fake|orig)={p_orig_e7:.3f} -> P(fake|swapped)={p_swap_e7:.3f} | Delta = +{d_swap_e7:.3f}")

    # ==========================================================================
    # Task A6: Export Artifacts (Contract #3, Models, Predictions)
    # ==========================================================================
    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)

    # 1. results.csv
    results_df = pd.DataFrame(results_rows)
    results_path = os.path.join(out_dir, "results.csv")
    results_df.to_csv(results_path, index=False)

    # 2. Model Checkpoint
    checkpoint_path = os.path.join(out_dir, "best_gated_fusion.pt")
    torch.save(best_e7_model.state_dict(), checkpoint_path)

    # 3. Test Predictions Parquet for Member B
    best_e7_model.eval()
    with torch.no_grad():
        logits, gates, cos_sims = best_e7_model(t_test.to(device), v_test.to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        gates_val = gates.mean(dim=-1).cpu().numpy()
        cos_val = cos_sims.squeeze(-1).cpu().numpy()

    test_export = df[test_mask].copy().reset_index(drop=True)
    test_export["prob_real"] = probs[:, 0]
    test_export["prob_fake"] = probs[:, 1]
    test_export["pred_label"] = np.argmax(probs, axis=-1)
    test_export["gate_weight"] = gates_val
    test_export["cross_modal_similarity"] = cos_val
    pred_path = os.path.join(out_dir, "predictions_test.parquet")
    test_export.to_parquet(pred_path, index=False)

    # ==========================================================================
    # Summary Table across 3 Seeds (Mean ± Std)
    # ==========================================================================
    print("\n" + "=" * 75)
    print("FINAL BENCHMARK TABLE (TEST SET ACROSS 3 SEEDS: MEAN ± STD)")
    print("=" * 75)
    test_df = results_df[results_df["split"] == "test"]
    summary = test_df.groupby(["exp_id", "metric"])["value"].agg(["mean", "std"]).reset_index()
    summary["formatted"] = summary.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}" if pd.notna(r["std"]) and r["std"] > 0 else f"{r['mean']:.4f}",
        axis=1,
    )
    table_pivot = summary.pivot(index="exp_id", columns="metric", values="formatted")
    ordered_cols = ["accuracy", "macro_f1", "mcc", "auc_pr", "auc_roc", "brier", "ece"]
    cols = [c for c in ordered_cols if c in table_pivot.columns]
    print(table_pivot[cols].to_string())
    print("=" * 75)
    print("PHASE A3 - A6 COMPLETE: All contracts and artifacts exported successfully!")


if __name__ == "__main__":
    run_full_pipeline()
