import os
import re
import sys
import cv2
import uuid
import numpy as np
import pytesseract
import requests
from pathlib import Path
from typing import Dict, Tuple, Optional, Any


# =========================================================
# Directory Setup
# =========================================================

TEMP_DIR = "temp"
SAVE_DIR = "downloads"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# Tesseract OCR Path (Cross-Platform)
# =========================================================

def configure_tesseract():
    """Configure Tesseract path based on operating system"""
    if sys.platform == "win32":
        # Windows paths
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    elif sys.platform == "darwin":
        # macOS paths
        possible_paths = [
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
        ]
    else:
        # Linux paths
        possible_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True
    
    # If no path found, pytesseract will try to use system PATH
    return False

configure_tesseract()


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

def download_image(url: str) -> Optional[str]:
    """
    Download image from URL and save to temp directory.
    
    Args:
        url: Image URL to download
        
    Returns:
        Path to downloaded image file, or None if download fails
    """
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

    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}")
        return None


# =========================================================
# Load Image with OpenCV
# =========================================================

def load_image(path: str) -> Optional[np.ndarray]:
    """
    Load image from file using OpenCV.
    
    Args:
        path: Path to image file
        
    Returns:
        Image as numpy array, or None if loading fails
    """
    try:
        image = cv2.imread(path)
        if image is None:
            print(f"Failed to load image from {path}")
        return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None


# =========================================================
# Preprocess Image for OCR
# =========================================================

def preprocess_for_ocr(gray: np.ndarray) -> np.ndarray:
    """
    Preprocess grayscale image for OCR text detection.
    
    Args:
        gray: Grayscale image array
        
    Returns:
        Preprocessed binary image array
    """
    h, w = gray.shape

    if w > MAX_OCR_WIDTH:
        ratio = MAX_OCR_WIDTH / w
        gray = cv2.resize(gray, None, fx=ratio, fy=ratio)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


# =========================================================
# Detect Image Edges
# =========================================================

def detect_edges(gray: np.ndarray) -> np.ndarray:
    """
    Detect edges in grayscale image using Canny edge detection.
    
    Args:
        gray: Grayscale image array
        
    Returns:
        Edge-detected image array
    """
    small = cv2.resize(gray, (800, 600))
    return cv2.Canny(small, 80, 180)


# =========================================================
# Detect Horizontal / Vertical Lines
# =========================================================

def detect_chart_lines(edges: np.ndarray) -> Tuple[int, int]:
    """
    Detect horizontal and vertical lines in edge image.
    
    Args:
        edges: Edge-detected image array
        
    Returns:
        Tuple of (horizontal_line_count, vertical_line_count)
    """
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

def detect_bar_chart(gray: np.ndarray) -> int:
    """
    Detect bar-like rectangular structures in image.
    
    Args:
        gray: Grayscale image array
        
    Returns:
        Count of detected rectangular bar structures
    """
    resized = cv2.resize(gray, (800, 600))
    _, thresh = cv2.threshold(resized, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

def detect_pie_chart(gray: np.ndarray) -> int:
    """
    Detect circular pie chart structures in image.
    
    Args:
        gray: Grayscale image array
        
    Returns:
        Count of detected circles (pie chart segments)
    """
    small = cv2.resize(gray, (400, 300))
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

def detect_text(gray: np.ndarray) -> str:
    """
    Extract text from image using Tesseract OCR.
    
    Args:
        gray: Grayscale image array
        
    Returns:
        Extracted text string (empty string if OCR fails)
    """
    try:
        processed = preprocess_for_ocr(gray)
        return pytesseract.image_to_string(processed, lang="kor+eng")
    except pytesseract.TesseractNotFoundError:
        print("Warning: Tesseract not found. OCR disabled. Install Tesseract for text detection.")
        return ""
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return ""


# =========================================================
# Calculate Numeric Ratio in OCR Text
# =========================================================

def numeric_ratio(text: str) -> float:
    """
    Calculate ratio of numeric characters in text.
    
    Args:
        text: Text string to analyze
        
    Returns:
        Ratio of numeric characters (0.0 to 1.0)
    """
    if len(text) == 0:
        return 0.0
    
    nums = re.findall(r'[0-9%]+', text)
    return len(nums) / len(text)


# =========================================================
# Calculate Final Chart Score
# =========================================================

def calculate_chart_score(
    text: str,
    horizontal: int,
    vertical: int,
    rect_count: int,
    pie_count: int,
    num_ratio: float
) -> int:
    """
    Calculate likelihood score that image contains a chart.
    
    Args:
        text: OCR extracted text
        horizontal: Count of horizontal lines detected
        vertical: Count of vertical lines detected
        rect_count: Count of rectangular bar structures
        pie_count: Count of circular pie structures
        num_ratio: Ratio of numeric characters in text
        
    Returns:
        Chart likelihood score (higher = more likely to be a chart)
    """
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

def save_chart(src_path: str, site: str) -> str:
    """
    Save detected chart image to organized directory structure.
    
    Args:
        src_path: Path to source image file
        site: Website domain name
        
    Returns:
        Path to saved image file
    """
    site_dir = os.path.join(SAVE_DIR, site.replace(".", "_"))
    os.makedirs(site_dir, exist_ok=True)

    dst = os.path.join(site_dir, os.path.basename(src_path))
    os.rename(src_path, dst)

    return dst


# =========================================================
# Main Chart Analysis Pipeline
# =========================================================

def analyze_chart(url: str, page: str, site: str) -> Dict[str, Any]:
    """
    Main pipeline: Download image, detect if it's a chart, and save if detected.
    
    Args:
        url: Image URL to analyze
        page: Page URL where image was found
        site: Website domain name
        
    Returns:
        Dictionary with analysis results:
        - success: bool - whether analysis completed without error
        - is_chart: bool - whether image is detected as a chart
        - score: int - chart likelihood score (optional)
        - saved: str - path to saved image (optional, if is_chart=True)
    """
    try:
        # Download image
        path = download_image(url)
        if not path:
            return {"success": False, "error": "Failed to download image"}

        # Load image
        image = load_image(path)
        if image is None:
            return {"success": False, "error": "Failed to load image"}

        # Check image dimensions
        h, w = image.shape[:2]
        if w < 300 or h < 180:
            try:
                os.remove(path)
            except:
                pass
            return {"success": True, "is_chart": False, "reason": "Image too small"}

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect edges and lines
        edges = detect_edges(gray)
        horizontal, vertical = detect_chart_lines(edges)

        # Early exit if no chart-like structures
        rect_count = detect_bar_chart(gray)
        if horizontal < 2 and vertical < 1 and rect_count == 0:
            try:
                os.remove(path)
            except:
                pass
            return {"success": True, "is_chart": False, "reason": "No chart structures detected"}

        # Detect pie charts
        pie_count = 0
        if rect_count < 2:
            pie_count = detect_pie_chart(gray)

        # Extract text and calculate score
        text = detect_text(gray)
        num_ratio = numeric_ratio(text)
        score = calculate_chart_score(text, horizontal, vertical, rect_count, pie_count, num_ratio)

        is_chart = score >= 6

        if is_chart:
            saved = save_chart(path, site)
            return {
                "success": True,
                "is_chart": True,
                "saved": saved,
                "score": score,
                "text": text[:100],  # First 100 chars
            }

        try:
            os.remove(path)
        except:
            pass

        return {
            "success": True,
            "is_chart": False,
            "score": score,
            "reason": f"Score {score} below threshold (6)"
        }

    except Exception as e:
        print(f"Error analyzing chart: {str(e)}")
        return {"success": False, "error": str(e)}