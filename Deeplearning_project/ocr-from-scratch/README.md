# Handwritten Pseudocode → Python Code

> Take a photo of a handwritten algorithm. Get back runnable Python code.

This project builds the **full pipeline from scratch** — a custom OCR engine that reads handwritten algorithm descriptions written in French pseudocode, then pipes the recognized text into **LLaMA 3** to automatically generate Python code. No pre-trained OCR backbone was used anywhere in the model.

---

## The Pipeline

```
 📷  Photo of handwritten algorithm (French pseudocode)
      │
      ▼
 ┌─────────────────────────────────────────────┐
 │              OCR Engine (custom)            │
 │                                             │
 │  CNN Backbone                               │
 │  (5 conv blocks → visual column features)  │
 │      │                                      │
 │  Linear Projection  512 → 256              │
 │      │                                      │
 │  Positional Encoding                        │
 │      │                                      │
 │  Transformer Encoder                        │
 │  (4 layers · 8 heads · GELU)               │
 │      │                                      │
 │  CTC Greedy Decoder                         │
 └──────────────────┬──────────────────────────┘
                    │
                    ▼
      Recognized text  (e.g. "algorithme de tri à bulles …")
                    │
                    ▼
 ┌─────────────────────────────────────────────┐
 │         LLaMA 3 8B  ·  Ollama              │
 │  "Translate this French algorithm to Python"│
 └──────────────────┬──────────────────────────┘
                    │
                    ▼
      ✅  Runnable Python code
```

---

## Why it's interesting

| Challenge | How it's solved |
|-----------|-----------------|
| No pre-trained backbone | CNN + Transformer trained from scratch with CTC loss |
| Variable-length handwritten sequences | CTC loss + greedy decoder (no forced alignment) |
| MPS doesn't support CTC | Forward pass on MPS, CTC loss computed on CPU |
| French pseudocode → code | LLaMA 3 8B via Ollama, prompt-engineered for code generation |
| Deployment | FastAPI REST API · single-image & batch endpoints · Docker |

---

## OCR Model Architecture

```
Input image [3 × 64 × 1024]
    │
    ▼
CNN Backbone
  Conv(3→64)   → BN → ReLU → MaxPool    [32 × 512]
  Conv(64→128)  → BN → ReLU → MaxPool    [16 × 256]
  Conv(128→256) → BN → ReLU ×2
  Conv(256→512) → BN → ReLU
  AdaptiveAvgPool → [1 × W]
    │
    ▼
Linear Projection  512 → 256
    │
Positional Encoding
    │
Transformer Encoder  (4 layers · 8 heads · d_model=256 · GELU · dropout=0.2)
    │
LayerNorm → Linear  256 → vocab_size
    │
    ▼
CTC Greedy Decoder  →  character sequence
```

| Hyperparameter | Value |
|----------------|-------|
| Image size | 64 × 1024 px |
| Hidden dim | 256 |
| Transformer heads | 8 |
| Transformer layers | 4 |
| Loss | CTCLoss (blank = 0) |
| Optimizer | AdamW (weight decay 1e-4) |
| Scheduler | OneCycleLR (max lr = 3e-4) |
| Batch size | 16 |
| Early stopping patience | 12 epochs |
| Hardware | Apple M4 MPS |

---

## Repository Structure

```
ocr-from-scratch/
├── 01_eda.ipynb               # Dataset exploration: image stats, character distribution, vocab
├── 02_training_mps.ipynb      # Full pipeline: train → evaluate → OCR → LLaMA code generation
├── results/
│   ├── best_mps_model.pth          # Best checkpoint by validation accuracy (after training)
│   ├── mps_training_results.png    # Loss & accuracy curves
│   ├── predictions.json            # OCR predictions on the validation set
│   └── converted_python_codes.json # LLaMA-generated Python code from OCR output
└── api/                       # Production deployment
    ├── main.py                # FastAPI: POST /predict  ·  POST /predict/batch
    ├── model.py               # Model class + CTC decoder + inference utilities
    ├── requirements.txt
    ├── Dockerfile
    ├── docker-compose.yml
    └── README.md
```

Training data: `../ocr_data/dataset/` (images + `labels.csv`)

---

## Quick Start — Training

```bash
# Step 1: explore the dataset
jupyter notebook 01_eda.ipynb

# Step 2: train OCR, evaluate, then run the full OCR → LLaMA pipeline
jupyter notebook 02_training_mps.ipynb
# outputs → results/best_mps_model.pth
#         → results/predictions.json
#         → results/converted_python_codes.json
```

---

## Quick Start — API

Full details in [api/README.md](api/README.md).

```bash
# Local dev
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# → http://localhost:8000/docs  (Swagger UI)

# Docker
docker compose up --build
```

Example request:
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@handwritten_algorithm.png"
# {"filename":"handwritten_algorithm.png","text":"algorithme de tri bulles ...","char_count":42}
```

---

## Training Results

![Training curves](results/mps_training_results.png)
