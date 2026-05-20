import os
import uuid
import requests
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cnn import classify_image, _get_model, THRESHOLD
from detector import detect_from_file

TEMP_DIR = "temp"
SAVE_DIR = "downloads"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    url: str
    page: str
    site: str


def _download(url: str):
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    except Exception as e:
        print(f"[download] {e}")
        return None


def _save(src: str, site: str) -> str:
    dst_dir = os.path.join(SAVE_DIR, site.replace(".", "_"))
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    os.rename(src, dst)
    return dst


def _discard(path: str):
    try:
        os.remove(path)
    except Exception:
        pass


@app.on_event("startup")
def startup():
    _get_model()


@app.post("/analyze")
def analyze(req: ImageRequest):
    path = _download(req.url)
    if not path:
        return {"success": False, "error": "Failed to download image"}

    try:
        # 1. CNN
        cnn_result = classify_image(path)
        if cnn_result["is_chart"]:
            saved = _save(path, req.site)
            return {"success": True, "is_chart": True, "method": "cnn", "confidence": cnn_result["confidence"], "saved": saved}

        # 2. Rule-based fallback
        rule_result = detect_from_file(path)
        if rule_result.get("is_chart"):
            saved = _save(path, req.site)
            return {"success": True, "is_chart": True, "method": "rule", "score": rule_result["score"], "saved": saved}

        _discard(path)
        return {
            "success": True,
            "is_chart": False,
            "cnn_confidence": cnn_result["confidence"],
            "rule_score": rule_result.get("score", 0),
        }

    except Exception as e:
        _discard(path)
        print(f"[analyze] {e}")
        return {"success": False, "error": str(e)}
