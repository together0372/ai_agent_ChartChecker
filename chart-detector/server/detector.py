import os
import re
import cv2
import uuid
import numpy as np
import pytesseract
import requests


# =========================================================
# Directory Setup
# =========================================================

TEMP_DIR = "temp"
SAVE_DIR = "downloads"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# Tesseract OCR Path
# =========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)


# =========================================================
# Chart-Related Keywords
# =========================================================

CHART_WORDS = [
    "%",
    "억원",
    "추이",
    "통계",
    "비율",
    "증가",
    "감소",
    "그래프",
    "차트",
    "매출",
    "실적",
    "점유율",
    "코스피",
    "주가",
    "시장",
    "금리",
    "수출",
    "수입",
    "관세",
    "환율"
]


# =========================================================
# OCR Resize Limit
# =========================================================

MAX_OCR_WIDTH = 1200


# =========================================================
# Download Image from URL
# =========================================================

def download_image(url):

    try:

        response = requests.get(
            url,
            timeout=5,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if response.status_code != 200:
            return None

        path = os.path.join(
            TEMP_DIR,
            f"{uuid.uuid4()}.jpg"
        )

        with open(path, "wb") as f:
            f.write(response.content)

        return path

    except:
        return None


# =========================================================
# Load Image with OpenCV
# =========================================================

def load_image(path):

    return cv2.imread(path)


# =========================================================
# Preprocess Image for OCR
# =========================================================

def preprocess_for_ocr(gray):

    h, w = gray.shape

    if w > MAX_OCR_WIDTH:

        ratio = MAX_OCR_WIDTH / w

        gray = cv2.resize(
            gray,
            None,
            fx=ratio,
            fy=ratio
        )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh


# =========================================================
# Detect Image Edges
# =========================================================

def detect_edges(gray):

    small = cv2.resize(
        gray,
        (800, 600)
    )

    return cv2.Canny(
        small,
        80,
        180
    )


# =========================================================
# Detect Horizontal / Vertical Lines
# =========================================================

def detect_chart_lines(edges):

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=70,
        minLineLength=50,
        maxLineGap=10
    )

    if lines is None:
        return 0, 0

    h = 0
    v = 0

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
# Detect Bar Chart Structure
# =========================================================

def detect_bar_chart(gray):

    resized = cv2.resize(
        gray,
        (800, 600)
    )

    _, thresh = cv2.threshold(
        resized,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    rects = 0

    for c in contours:

        x, y, w, h = cv2.boundingRect(c)

        if w * h < 300:
            continue

        ratio = h / (w + 1)

        if ratio > 1.3:
            rects += 1

    return rects


# =========================================================
# Detect Pie Chart Structure
# =========================================================

def detect_pie_chart(gray):

    small = cv2.resize(
        gray,
        (400, 300)
    )

    circles = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        1,
        80,
        param1=50,
        param2=25,
        minRadius=20,
        maxRadius=150
    )

    if circles is None:
        return 0

    return len(circles[0])


# =========================================================
# OCR Text Detection
# =========================================================

def detect_text(gray):

    try:

        processed = preprocess_for_ocr(gray)

        return pytesseract.image_to_string(
            processed,
            lang="kor+eng"
        )

    except:
        return ""


# =========================================================
# Calculate Numeric Ratio in OCR Text
# =========================================================

def numeric_ratio(text):

    nums = re.findall(r'[0-9%]+', text)

    if len(text) == 0:
        return 0

    return len(nums) / len(text)


# =========================================================
# Calculate Final Chart Score
# =========================================================

def calculate_chart_score(
    text,
    horizontal,
    vertical,
    rect_count,
    pie_count,
    num_ratio
):

    score = 0

    lower = text.lower()

    for word in CHART_WORDS:

        if word.lower() in lower:
            score += 1

    if horizontal >= 3:
        score += 2

    if vertical >= 2:
        score += 2

    if rect_count >= 2:
        score += 4

    if pie_count > 0:
        score += 4

    if num_ratio > 0.05:
        score += 2

    if num_ratio > 0.12:
        score += 3

    return score


# =========================================================
# Save Detected Chart Image
# =========================================================

def save_chart(src_path, site):

    site_dir = os.path.join(
        SAVE_DIR,
        site.replace(".", "_")
    )

    os.makedirs(site_dir, exist_ok=True)

    dst = os.path.join(
        site_dir,
        os.path.basename(src_path)
    )

    os.rename(src_path, dst)

    return dst


# =========================================================
# Main Chart Analysis Pipeline
# =========================================================

def analyze_chart(url, page, site):

    path = download_image(url)

    if not path:

        return {
            "success": False
        }

    image = load_image(path)

    if image is None:

        return {
            "success": False
        }

    h, w = image.shape[:2]

    if w < 300 or h < 180:

        os.remove(path)

        return {
            "success": False
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = detect_edges(gray)

    horizontal, vertical = detect_chart_lines(edges)

    rect_count = detect_bar_chart(gray)

    if (
        horizontal < 2 and
        vertical < 1 and
        rect_count == 0
    ):

        os.remove(path)

        return {
            "success": True,
            "is_chart": False
        }

    pie_count = 0

    if rect_count < 2:
        pie_count = detect_pie_chart(gray)

    text = detect_text(gray)

    num_ratio = numeric_ratio(text)

    score = calculate_chart_score(
        text,
        horizontal,
        vertical,
        rect_count,
        pie_count,
        num_ratio
    )

    is_chart = score >= 6

    if is_chart:

        saved = save_chart(path, site)

        return {
            "success": True,
            "is_chart": True,
            "saved": saved,
            "score": score
        }

    os.remove(path)

    return {
        "success": True,
        "is_chart": False,
        "score": score
    }