# Beyond Shortcuts: Multimodal Fake News Detection

Consistency-Aware Multimodal Fake News Detection using CLIP representations, group-aware zero-leakage deduplication, dynamic gating fusion, and post-hoc calibration on the Fakeddit benchmark.

The [Member A handoff](docs/MEMBER_A_GUIDE.md) lists the two required archives, their extraction paths, the corrected benchmark, and remaining evaluation work.

---

## 👥 Team Work Allocation

| Component | Member A (Data, Models & API) | Member B (Evaluation, Calibration & UI) |
| :--- | :--- | :--- |
| **Data Ingestion & Splits** | **Phase A1:** 94,719-post cleaning, dHash deduplication & group split (`N1`) | **Task B1:** Group-aware split unit tests (`test_splits.py`) |
| **Embeddings Extraction** | **Phase A2:** Frozen CLIP ViT-B/32 GPU pooled embeddings (`N2`) | - |
| **Baselines & Fusion** | **Phase A3–A4:** E-1 to E6 training across 3 seeds (`N3`) | - |
| **Hard-Mismatch Miner** | **Phase A5:** In-batch cosine mining & E7 training (`N3`) | **Task B2:** Swap-the-Image test & modality dropout (`eval/diagnostics.py`) |
| **Evaluation & Metrics** | **Phase A6:** Macro-F1, MCC, AUC-PR, Brier, ECE, Contract #3 export | **Task B3:** Temperature scaling & risk-coverage curves (`eval/calibration.py`, `eval/plots.py`) |
| **Explainability** | - | **Task B4:** Cross-modal heatmaps & token attribution (`explain/`) |
| **Production Serving** | **Phase A7:** FastAPI backend service & Docker (`api/`, `serve/`) | **Task B5:** Web frontend UI (`web/`) connected to `POST /predict` |

---

## 📁 Repository Directory Structure

```text
fakenews-mm/
├── api/                    # Production FastAPI backend service
│   ├── main.py             # REST API application (/predict, /health)
│   ├── schemas.py          # Pydantic data contracts (Contract #4)
│   └── Dockerfile          # Containerization for FastAPI service
├── checkpoints/            # Saved model weights (best_gated_fusion.pt)
├── docker-compose.yml      # Orchestration for one-command API deployment
├── eval/                   # [Member B] Evaluation, calibration & plotting
│   ├── calibration.py      # Temperature scaling, ECE, risk-coverage curves
│   ├── diagnostics.py      # Swap-the-image sensitivity audit & modality dropout
│   └── plots.py            # Reliability diagrams, risk-coverage curves, benchmark charts
├── explain/                # [Member B] Interpretability & explainability
│   ├── heatmap.py          # Cross-modal token-patch cosine alignment heatmap
│   └── tokens.py           # Occlusion-based word importance attribution
├── notebooks/              # Complete Kaggle GPU execution pipeline
│   ├── kaggle_01_prepare.ipynb    # N1: Data ingestion & zero-leakage split
│   ├── kaggle_02_embeddings.ipynb # N2: GPU CLIP ViT-B/32 feature extraction
│   └── kaggle_03_train.ipynb      # N3: Multi-seed benchmark (E-1 to E7)
├── pyproject.toml          # Project metadata and dependencies
├── results/                # Academic benchmark tables & generated curves
│   └── results_template.csv# Contract #3 results schema
├── serve/                  # Production inference pipeline
│   └── detector.py         # End-to-end multimodal detector (CLIP + Gated Fusion)
├── src/fnd/                # Core reusable deep learning & data modules
│   ├── dataset.py          # Precomputed PyTorch dataset loaders
│   ├── dedupe.py           # 64-bit dHash & Union-Find clusterer
│   ├── download.py         # Resilient image downloader with PIL guards
│   ├── encoders.py         # CLIP ViT-B/32 pooled extraction module
│   ├── fusion.py           # Consistency-Aware Gated Multimodal Fusion (Contribution C2)
│   ├── heads.py            # Unimodal MLP & Early Concatenation heads
│   └── splits.py           # Group-aware split generator & zero-leakage validator
├── tests/                  # Automated pytest test suites (9/9 passing)
│   ├── test_api.py         # FastAPI endpoint integration & schema tests
│   └── test_splits.py      # Zero-leakage & dHash deduplication tests
└── web/                    # [Member B] Interactive Web Frontend Application
    ├── index.html          # Modern dark-mode UI querying POST /predict live
    └── README.md           # Instructions for running and testing the frontend
```

---

## 🚀 Quickstart for Member B (Getting Started in 2 Minutes)

### 1. Start the FastAPI Backend
```bash
uvicorn api.main:app --reload --port 8000
```
* **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

### 2. Launch the Web UI
Open [`web/index.html`](web/index.html) in your browser:
```bash
# On Windows
start web/index.html
# Or serve on port 3000
python -m http.server 3000 --directory web/
```
Type any headline and paste an image URL to see live real-time multimodal predictions!

### 3. Run Automated Tests
```bash
pytest -v
```
All 9 unit tests pass in under 4 seconds!
