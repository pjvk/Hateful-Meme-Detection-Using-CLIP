"""
FastAPI service for hateful meme detection.

Run from project root:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import clip
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.model import load_clip_mlp
from src.predict import LABEL_MAP
from src.utils import default_checkpoint_path, get_device

logger = logging.getLogger(__name__)

# Global model loaded once at startup
_model = None
_device = None


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device
    logging.basicConfig(level=logging.INFO)
    _device = get_device()
    env_ckpt = os.environ.get("CHECKPOINT_PATH")
    checkpoint = Path(env_ckpt) if env_ckpt else default_checkpoint_path()
    if not checkpoint.exists():
        logger.warning(
            "Checkpoint not found at %s. Predictions use untrained MLP weights. "
            "Train in Colab and set CHECKPOINT_PATH or place file at checkpoints/clip_mlp.pt",
            checkpoint,
        )
    _model = load_clip_mlp(checkpoint_path=checkpoint, device=_device)
    logger.info("Model loaded on %s from %s", _device, checkpoint)
    yield
    _model = None


app = FastAPI(
    title="Multimodal Hateful Meme Detection",
    description="CLIP + MLP classifier for Facebook Hateful Memes",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
async def predict(
    image: Annotated[UploadFile, File(description="Meme image (PNG/JPG)")],
    text: Annotated[str, Form(description="Meme caption text")] = "",
):
    """
    Classify uploaded meme as Hateful or Non-Hateful.

    Returns JSON: {"label": "Hateful", "confidence": 0.91}
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    try:
        model = get_model()
        image_tensor = model.preprocess(pil_image).unsqueeze(0).to(_device)
        text_tokens = clip.tokenize([text or ""], truncate=True).to(_device)

        with torch.no_grad():
            logits = model(image_tensor, text_tokens)
            probs = torch.softmax(logits, dim=1)[0]
            class_id = int(probs.argmax().item())
            confidence = float(probs[class_id].item())

        return JSONResponse(
            content={
                "label": LABEL_MAP[class_id],
                "confidence": round(confidence, 4),
            }
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.get("/")
async def root():
    return {
        "message": "Hateful Meme Detection API",
        "docs": "/docs",
        "predict": "POST /predict (multipart: image file, optional text field)",
    }
