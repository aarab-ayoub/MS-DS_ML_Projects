"""
OCR REST API — built with FastAPI.

Endpoints
---------
GET  /             → health check & vocab info
POST /predict      → upload an image, get back the predicted text
POST /predict/batch → upload multiple images at once

Usage
-----
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""

import os
import csv
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model import ImprovedOCR, load_model, predict

# ─── Config ───────────────────────────────────────────────────────────────────

WEIGHTS_PATH = os.getenv("OCR_WEIGHTS", "../results/best_mps_model.pth")
LABELS_CSV   = os.getenv("OCR_LABELS",  "../ocr_data/dataset/labels.csv")
MAX_UPLOAD_MB = 10

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ocr-api")

# ─── App state ────────────────────────────────────────────────────────────────

state: dict = {}


def build_vocab(labels_csv: str) -> tuple[dict[str, int], dict[int, str]]:
    """Read labels CSV and derive the character vocabulary."""
    texts: list[str] = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            if text:
                texts.append(text)

    unique_chars = sorted(set("".join(texts)))
    char_list    = ["<blank>"] + unique_chars
    char_to_idx  = {ch: i for i, ch in enumerate(char_list)}
    idx_to_char  = {i: ch for i, ch in enumerate(char_list)}

    log.info("Vocabulary size: %d characters", len(char_list))
    return char_to_idx, idx_to_char


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──────────────────────────────────────────────────────────────
    log.info("Loading vocabulary from %s", LABELS_CSV)
    if not Path(LABELS_CSV).exists():
        raise FileNotFoundError(
            f"Labels CSV not found at {LABELS_CSV}. "
            "Set the OCR_LABELS environment variable to the correct path."
        )

    char_to_idx, idx_to_char = build_vocab(LABELS_CSV)
    vocab_size = len(char_to_idx)

    # Device — prefer MPS (Apple Silicon) → CUDA → CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Using CUDA")
    else:
        device = torch.device("cpu")
        log.info("Using CPU")

    log.info("Loading model weights from %s", WEIGHTS_PATH)
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(
            f"Model weights not found at {WEIGHTS_PATH}. "
            "Train the model first (02_training_mps.ipynb) or set OCR_WEIGHTS."
        )

    model = load_model(WEIGHTS_PATH, num_classes=vocab_size, device=device)
    log.info("Model loaded — %d parameters", sum(p.numel() for p in model.parameters()))

    state.update(
        model=model,
        idx_to_char=idx_to_char,
        vocab_size=vocab_size,
        device=device,
    )
    yield
    # ── shutdown ─────────────────────────────────────────────────────────────
    state.clear()


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="OCR from Scratch — API",
    description=(
        "Optical Character Recognition trained from scratch using a "
        "CNN + Transformer architecture with CTC loss on Apple MPS.\n\n"
        "Upload an image of handwritten or printed text and receive the predicted string."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Response schemas ─────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    filename: str
    text: str
    char_count: int


class HealthResponse(BaseModel):
    status: str
    device: str
    vocab_size: int
    model_params: int


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, summary="Health check")
def health():
    """Returns API status and model metadata."""
    model: ImprovedOCR = state["model"]
    return {
        "status":       "ok",
        "device":       str(state["device"]),
        "vocab_size":   state["vocab_size"],
        "model_params": sum(p.numel() for p in model.parameters()),
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict text from image")
async def predict_single(file: UploadFile = File(..., description="Image file (PNG, JPEG, etc.)")):
    """
    Upload a single image and receive the recognized text.

    The image is resized to 64×1024 internally — no pre-processing needed on the client side.
    """
    _validate_upload(file)
    image_bytes = await file.read()

    try:
        text = predict(
            model       = state["model"],
            image_bytes = image_bytes,
            idx_to_char = state["idx_to_char"],
            device      = state["device"],
        )
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(filename=file.filename or "upload", text=text, char_count=len(text))


@app.post("/predict/batch", summary="Predict text from multiple images")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Upload up to 20 images at once. Returns a list of predictions in the same order.
    """
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images per batch request.")

    results = []
    for file in files:
        _validate_upload(file)
        image_bytes = await file.read()
        try:
            text = predict(
                model       = state["model"],
                image_bytes = image_bytes,
                idx_to_char = state["idx_to_char"],
                device      = state["device"],
            )
            results.append({"filename": file.filename, "text": text, "char_count": len(text)})
        except Exception as exc:
            results.append({"filename": file.filename, "error": str(exc)})

    return JSONResponse(content={"predictions": results, "count": len(results)})


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile) -> None:
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'. Allowed: {allowed}",
        )
