import os
import cv2
import uuid
import numpy as np
import requests
from typing import Dict, Tuple, Optional, Any


# =========================================================
# Directory Setup
# =========================================================

TEMP_DIR = "temp"
SAVE_DIR = "downloads"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# Download Image from URL
# =========================================================

def download_image(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return None

        path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
        with open(path, "wb") as f:
            f.write(response.content)
        return path

    except Exception as e:
        print(f"[download_image] {e}")
        return None


# =========================================================
# Load Image with OpenCV
# =========================================================

def load_image(path: str) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[load_image] Failed to load: {path}")
        return image
    except Exception as e:
        print(f"[load_image] {e}")
        return None


# =========================================================
# Detect Edges (Canny)
# =========================================================

def detect_edges(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (800, 600))
    return cv2.Canny(small, 80, 180)


# =========================================================
# Detect Horizontal / Vertical Lines
# =========================================================

def detect_chart_lines(edges: np.ndarray) -> Tuple[int, int]:
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=15
    )

    if lines is None:
        return 0, 0

    h = v = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx > dy * 2:
            h += 1
        elif dy > dx * 2:
            v += 1

    return h, v


# =========================================================
# Detect Bar Chart Structures
# =========================================================

def detect_bar_chart(gray: np.ndarray) -> int:
    """
    Rectangular contour detection for bar structures.
    Permissive thresholds to prioritize recall.
    """
    small = cv2.resize(gray, (800, 600))
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = 800 * 600
    rect_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.003 or area > img_area * 0.40:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) != 4:
            continue

        x, y, cw, ch = cv2.boundingRect(approx)
        if cw == 0:
            continue

        fill_ratio = area / (cw * ch)
        aspect = ch / cw

        if fill_ratio > 0.70 and (aspect > 1.1 or aspect < 0.55):
            rect_count += 1

    return rect_count


# =========================================================
# Detect Pie Chart Structures
# =========================================================

def detect_pie_chart(gray: np.ndarray) -> int:
    """
    HoughCircles with low param2 for higher recall.
    """
    small = cv2.resize(gray, (800, 600))
    blurred = cv2.GaussianBlur(small, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=150,
        param1=80,
        param2=30,
        minRadius=50,
        maxRadius=280,
    )

    return 0 if circles is None else len(circles[0])


# =========================================================
# Text / Numeric (OCR disabled)
# =========================================================

def detect_text(gray: np.ndarray) -> str:
    return ""


def numeric_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit() or c in "%.,-") / len(text)


# =========================================================
# Chart Likelihood Score
# =========================================================

def calculate_chart_score(
    text: str,
    horizontal: int,
    vertical: int,
    rect_count: int,
    pie_count: int,
    num_ratio: float,
) -> int:
    """
    Recall-first scoring. Threshold >= 3 in analyze_chart.
    Max 11 pts: lines(4) + bars(4) + pie(3)
    """
    score = 0

    # Grid / axis lines
    if horizontal >= 5 and vertical >= 3:
        score += 4
    elif horizontal >= 3 and vertical >= 2:
        score += 3
    elif horizontal >= 2 and vertical >= 1:
        score += 2
    elif horizontal >= 2 or vertical >= 2:
        score += 1

    # Bar rectangles
    if rect_count >= 5:
        score += 4
    elif rect_count >= 3:
        score += 3
    elif rect_count >= 2:
        score += 2
    elif rect_count >= 1:
        score += 1

    # Circular structures
    if pie_count >= 1:
        score += 3

    # Numeric density bonus (active when OCR enabled)
    if num_ratio >= 0.15:
        score += 2
    elif num_ratio >= 0.08:
        score += 1

    return score


# =========================================================
# Save Detected Chart Image
# =========================================================

def save_chart(src_path: str, site: str) -> str:
    site_dir = os.path.join(SAVE_DIR, site.replace(".", "_"))
    os.makedirs(site_dir, exist_ok=True)

    dst = os.path.join(site_dir, os.path.basename(src_path))
    os.rename(src_path, dst)
    return dst


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass


# =========================================================
# Main Chart Analysis Pipeline
# =========================================================

def detect_from_file(path: str) -> Dict[str, Any]:
    image = load_image(path)
    if image is None:
        return {"success": False, "error": "Failed to load image"}

    h, w = image.shape[:2]
    if w < 300 or h < 180:
        return {"success": True, "is_chart": False, "reason": "Image too small"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = detect_edges(gray)
    horizontal, vertical = detect_chart_lines(edges)
    rect_count = detect_bar_chart(gray)

    pie_count = 0
    if rect_count < 2:
        pie_count = detect_pie_chart(gray)

    if horizontal == 0 and vertical == 0 and rect_count == 0 and pie_count == 0:
        return {"success": True, "is_chart": False, "reason": "No visual structure detected"}

    text = detect_text(gray)
    num_ratio = numeric_ratio(text)
    score = calculate_chart_score(text, horizontal, vertical, rect_count, pie_count, num_ratio)

    return {
        "success": True,
        "is_chart": score >= 3,
        "score": score,
        "h_lines": horizontal,
        "v_lines": vertical,
        "rects": rect_count,
        "circles": pie_count,
    }


def analyze_chart(url: str, page: str, site: str) -> Dict[str, Any]:
    try:
        path = download_image(url)
        if not path:
            return {"success": False, "error": "Failed to download image"}

        image = load_image(path)
        if image is None:
            return {"success": False, "error": "Failed to load image"}

        h, w = image.shape[:2]
        if w < 300 or h < 180:
            _discard(path)
            return {"success": True, "is_chart": False, "reason": "Image too small"}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = detect_edges(gray)
        horizontal, vertical = detect_chart_lines(edges)
        rect_count = detect_bar_chart(gray)

        # pie detection before early exit to avoid missing pie-only charts
        pie_count = 0
        if rect_count < 2:
            pie_count = detect_pie_chart(gray)

        # only early-exit when absolutely no visual structure exists
        if horizontal == 0 and vertical == 0 and rect_count == 0 and pie_count == 0:
            _discard(path)
            return {"success": True, "is_chart": False, "reason": "No visual structure detected"}

        text = detect_text(gray)
        num_ratio = numeric_ratio(text)
        score = calculate_chart_score(text, horizontal, vertical, rect_count, pie_count, num_ratio)

        is_chart = score >= 3

        if is_chart:
            saved = save_chart(path, site)
            return {
                "success": True,
                "is_chart": True,
                "saved": saved,
                "score": score,
            }

        _discard(path)
        return {
            "success": True,
            "is_chart": False,
            "score": score,
            "reason": f"Score {score} below threshold (3)",
        }

    except Exception as e:
        print(f"[analyze_chart] {e}")
        return {"success": False, "error": str(e)}
