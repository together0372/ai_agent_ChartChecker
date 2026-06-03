"""
서버 설정 (통합 버전)
"""
import os


class Config:
    # ── LLM 설정 (quiz_generator용) ─────────────────────────
    # core agent의 LLM은 agents/core/config.py의 LLM_NAME 참조
    MODEL_NAME:        str   = os.environ.get("CHART_MODEL", "qwen3.5:9b")
    MODEL_TEMPERATURE: float = 0.1
    MODEL_TOP_K:       int   = 64
    MODEL_TOP_P:       float = 0.95

    # ── RAG 설정 ─────────────────────────────────────────────
    USE_VECTOR_DB: bool = True
    # True  → MisViz 벡터DB 사용 (첫 실행 시 초기화 ~1분)
    # False → RAG 비활성화 (빠른 실행)

    # ── 경로 ─────────────────────────────────────────────────
    ROOT_DIR:  str = os.path.dirname(os.path.abspath(__file__))
    TEMP_DIR:  str = os.path.join(ROOT_DIR, "temp")
    SAVE_DIR:  str = os.path.join(ROOT_DIR, "downloads")
