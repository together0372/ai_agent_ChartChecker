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
    title: str = ""


def _slugify(text: str, maxlen: int = 60) -> str:
    import re
    text = re.sub(r'[\\/:*?"<>|]', "", text).strip()
    text = re.sub(r"\s+", "_", text)
    return text[:maxlen]


def _article_folder(title: str, page: str) -> str:
    from urllib.parse import urlparse, parse_qs
    if title:
        slug = _slugify(title)
        if slug:
            return slug
    # title이 없거나 비면 URL에서 식별자 추출
    parsed = urlparse(page)
    # 쿼리스트링에서 기사 ID 키 탐색 (ncd, news_id, article_id, id 등)
    for key in ("ncd", "news_id", "article_id", "id", "no"):
        val = parse_qs(parsed.query).get(key)
        if val:
            return _slugify(val[0]) or "untitled"
    # 경로 마지막 세그먼트
    seg = [s for s in parsed.path.split("/") if s]
    if seg:
        return _slugify(seg[-1]) or "untitled"
    return "untitled"


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


def _save(src: str, site: str, title: str = "", page: str = "") -> str:
    article = _article_folder(title, page)
    dst_dir = os.path.join(SAVE_DIR, site.replace(".", "_"), article)
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
        cnn_result = classify_image(path)
        rule_result = detect_from_file(path)
        
        is_chart = cnn_result["is_chart"]   # CNN이 최종 결정
        
        if is_chart:
            saved = _save(path, req.site, req.title, req.page)
        else:
            _discard(path)
            saved = None

        return {
            "success": True,
            "is_chart": is_chart,
            "cnn_confidence": cnn_result["confidence"],
            "rule_score": rule_result.get("score", 0),
            "rule_is_chart": rule_result.get("is_chart", False),
            "agree": cnn_result["is_chart"] == rule_result.get("is_chart", False),
            "saved": saved,
        }

    except Exception as e:
        _discard(path)
        return {"success": False, "error": str(e)}
