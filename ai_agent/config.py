import os


class Config:
    """프로젝트 설정 관리 클래스"""

    # OpenAI 설정
    # ① 환경변수에서 API 키를 가져오되, 없으면 빈 문자열을 기본값으로 사용
    MODEL_NAME: str = "gemma4:e4b-it-q4_K_M"
    MODEL_TEMPERATURE: float = 0.0
    MODEL_TOP_K: int = 5
    MODEL_TOP_P: int = 64

    # ② 현재 파일의 위치를 기준으로 프로젝트 루트 디렉토리를 설정
    ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))

    # ③ API 호출을 효율적으로 하기 위한 배치 크기를 설정
    BATCH_SIZE: int = 10

    # ⑤ 출력 파일들을 저장할 디렉토리 설정
    OUTPUT_DIR: str = f"{ROOT_DIR}/outputs"
