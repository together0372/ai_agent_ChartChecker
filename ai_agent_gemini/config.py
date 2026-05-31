import os


class Config:
    """프로젝트 설정 관리 클래스 (Gemini API 버전)"""

    # Google Gemini API 모델 설정
    # GOOGLE_API_KEY 환경변수가 반드시 설정되어 있어야 합니다.
    # 예) export GOOGLE_API_KEY="AIzaSy..."
    MODEL_NAME: str = os.environ.get("CHART_MODEL", "gemini-3.5-flash")
    # gemini-3.5-flash      → GA. 100만 토큰 컨텍스트, 65K 출력 (기본값)
    # gemini-3.1-pro-preview→ 3.1 Pro 프리뷰
    # gemini-2.5-pro        → 최고 정확도 (비용 높음, 안정)
    # gemini-2.5-flash      → 이전 Flash 안정버전

    # Gemini 3.5 Flash: temperature deprecated → thinking_level 로 대체
    # "minimal" | "low" | "medium"(기본) | "high"
    THINKING_LEVEL: str = os.environ.get("GEMINI_THINKING_LEVEL", "medium")

    # 현재 파일의 위치를 기준으로 프로젝트 루트 디렉토리를 설정
    ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))

    BATCH_SIZE: int = 10

    OUTPUT_DIR: str = f"{ROOT_DIR}/outputs"
