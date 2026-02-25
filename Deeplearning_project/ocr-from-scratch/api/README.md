# OCR API

A production-ready REST API that wraps the trained OCR model, built with **FastAPI**.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`  | Health check — returns model info |
| `POST` | `/predict` | Predict text from a single image upload |
| `POST` | `/predict/batch` | Predict text from up to 20 images at once |

Interactive docs available at `http://localhost:8000/docs` once the server is running.

---

## Prerequisites

> You need to have trained the model first (run `02_training_mps.ipynb`).  
> The checkpoint `best_mps_model.pth` must exist in `../results/`.

---

## Option 1 — Run locally (recommended for development)

```bash
# From the api/ directory
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@/path/to/your/image.png"
```

---

## Option 2 — Docker

```bash
# From the api/ directory
docker compose up --build
```

The container mounts `../results/best_mps_model.pth` and `../ocr_data/dataset/labels.csv` as read-only volumes, so no image rebuild is needed when you retrain.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_WEIGHTS` | `../results/best_mps_model.pth` | Path to model checkpoint |
| `OCR_LABELS`  | `../ocr_data/dataset/labels.csv` | Path to labels CSV (used to build vocab) |

---

## Example response

```json
{
  "filename": "sample.png",
  "text": "algorithme de tri à bulles",
  "char_count": 26
}
```
