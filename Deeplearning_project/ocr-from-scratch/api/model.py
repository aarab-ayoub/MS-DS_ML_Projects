"""
OCR Model definition and inference utilities.

Architecture: CNN backbone + Transformer encoder + CTC decoding
Trained with Apple MPS acceleration on a custom handwritten text dataset.
"""

import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io


# ─── Constants ────────────────────────────────────────────────────────────────
IMG_HEIGHT = 64
IMG_WIDTH  = 1024

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─── Model Architecture ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ImprovedOCR(nn.Module):
    """
    CNN + Transformer OCR model trained with CTC loss.

    Input : [B, 3, 64, 1024] image tensor
    Output: [T, B, vocab_size] logits (CTC format)
    """

    def __init__(self, num_classes: int, hidden_dim: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()

        # CNN backbone — extracts visual features column by column
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),   nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2, 2),                                              # 32×512

            nn.Conv2d(64, 128, 3, 1, 1),  nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),                                              # 16×256

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),                                 # 1×W
        )

        self.proj = nn.Linear(512, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                      # [B, 512, 1, W]
        x = x.squeeze(2).permute(0, 2, 1)    # [B, W, 512]
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)            # [T, B, vocab_size]  — CTC format


# ─── Decoding ─────────────────────────────────────────────────────────────────

def ctc_greedy_decode(logits: torch.Tensor, idx_to_char: dict[int, str]) -> list[str]:
    """
    Greedy CTC decoder — collapse repeated tokens and remove blank (index 0).

    Args:
        logits:      [T, B, vocab_size] tensor (raw logits, not log-softmax)
        idx_to_char: mapping from index to character

    Returns:
        List of decoded strings, one per batch item.
    """
    preds = []
    logits = logits.permute(1, 0, 2)          # → [B, T, vocab_size]
    for i in range(logits.size(0)):
        indices = torch.argmax(logits[i], dim=-1).tolist()
        decoded, prev = [], None
        for idx in indices:
            if idx != 0 and idx != prev:       # skip blank (0) and repeats
                decoded.append(idx_to_char.get(idx, ""))
            prev = idx
        preds.append("".join(decoded))
    return preds


# ─── Inference helper ─────────────────────────────────────────────────────────

def load_model(weights_path: str, num_classes: int, device: torch.device) -> ImprovedOCR:
    """Load model weights from a checkpoint file."""
    model = ImprovedOCR(num_classes=num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Convert raw image bytes to a model-ready tensor.

    Returns: [1, 3, IMG_HEIGHT, IMG_WIDTH]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = INFERENCE_TRANSFORM(img)
    return tensor.unsqueeze(0)   # add batch dimension


def predict(
    model: ImprovedOCR,
    image_bytes: bytes,
    idx_to_char: dict[int, str],
    device: torch.device,
) -> str:
    """End-to-end inference: bytes → predicted text string."""
    tensor = preprocess_image(image_bytes).to(device)
    with torch.no_grad():
        logits = model(tensor)
    results = ctc_greedy_decode(logits.cpu(), idx_to_char)
    return results[0]
