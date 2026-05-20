import os
import uuid
import requests
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

THRESHOLD  = 0.35
MODEL_PATH = Path(__file__).parent / "model" / "chart_classifier.pth"
TEMP_DIR   = "temp"
SAVE_DIR   = "downloads"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

_model = None


def _get_model() -> nn.Module:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                "Run train.py first to fine-tune the classifier."
            )
        print("[classifier] Loading MobileNetV3-Small...")
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        m.eval()
        _model = m
        print("[classifier] Model ready.")
    return _model


def _download_image(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    except Exception as e:
        print(f"[classifier] download error: {e}")
        return None


def _save_chart(src: str, site: str) -> str:
    dst_dir = os.path.join(SAVE_DIR, site.replace(".", "_"))
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    os.rename(src, dst)
    return dst


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass


def classify_image(image_path: str) -> Dict[str, Any]:
    model = _get_model()
    img   = Image.open(image_path).convert("RGB")
    x     = _transform(img).unsqueeze(0)

    with torch.no_grad():
        confidence = torch.sigmoid(model(x).squeeze()).item()

    return {
        "is_chart":   confidence >= THRESHOLD,
        "confidence": round(confidence, 4),
    }


def analyze_chart(url: str, page: str, site: str) -> Dict[str, Any]:
    path = _download_image(url)
    if not path:
        return {"success": False, "error": "Failed to download image"}

    try:
        w, h = Image.open(path).size
        if w < 300 or h < 180:
            _discard(path)
            return {"success": True, "is_chart": False, "reason": "Image too small"}

        result = classify_image(path)

        if result["is_chart"]:
            saved = _save_chart(path, site)
            return {"success": True, "is_chart": True, "saved": saved, "confidence": result["confidence"]}

        _discard(path)
        return {
            "success": True,
            "is_chart": False,
            "confidence": result["confidence"],
            "reason": f"Confidence {result['confidence']} below threshold ({THRESHOLD})",
        }

    except Exception as e:
        _discard(path)
        print(f"[classifier] error: {e}")
        return {"success": False, "error": str(e)}
