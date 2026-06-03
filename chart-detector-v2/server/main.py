"""
FastAPI 서버 (통합 버전)

흐름:
  Chrome Extension → POST /analyze (이미지 URL)
    → CNN 차트 판별 (cnn.py)
    → 차트이면 멀티에이전트 오류 탐지 (ai_agent_1 core)
    → OX 퀴즈 생성 (quiz_generator)
    → 결과 반환 (오류 목록 + 퀴즈)
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from config import Config
from detector import analyze_chart
from state import ChartCheckState
from workflow import create_news_workflow

# RAG 초기화 (USE_VECTOR_DB=True 일 때만)
if Config.USE_VECTOR_DB:
    try:
        # core/rag_utils의 init_vector_db 호출
        # 작업 디렉토리를 server/로 설정해야 상대경로 정상 작동
        os.chdir(Path(__file__).parent)
        from agents.core.rag_utils import init_vector_db
        print("\n[RAG] 벡터DB 초기화 중...")
        init_vector_db()
        print("[RAG] 초기화 완료\n")
    except Exception as e:
        print(f"[RAG] 초기화 실패 (계속 진행): {e}\n")


# ── FastAPI 앱 ──────────────────────────────────────────────
app = FastAPI(title="Chart Distortion Detector v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── AI 워크플로우 (서버 시작 시 1회 로드) ──────────────────
llm = ChatOllama(
    model=Config.MODEL_NAME,
    temperature=Config.MODEL_TEMPERATURE,
    top_p=Config.MODEL_TOP_P,
    top_k=Config.MODEL_TOP_K,
)
workflow_app = create_news_workflow(llm, use_rag=Config.USE_VECTOR_DB)


# ── 요청 스키마 ────────────────────────────────────────────
class ImageRequest(BaseModel):
    url:   str
    page:  str
    site:  str
    title: str = ""


# ── 엔드포인트 ────────────────────────────────────────────
@app.post("/analyze")
async def analyze(req: ImageRequest):
    """
    1. CNN으로 이미지가 차트인지 판별
    2. 차트이면 멀티에이전트로 오류 탐지 + OX 퀴즈 생성
    3. 결과 반환
    """
    # ── Step 1: CNN 차트 판별 ─────────────────────────────
    result = analyze_chart(req.url, req.page, req.site, req.title)

    if not result.get("is_chart"):
        return result

    # ── Step 2: 멀티에이전트 오류 탐지 ───────────────────
    saved_path = result.get("saved", "")
    print(f"\n[AI 분석 시작] {saved_path}")

    try:
        initial_state = ChartCheckState(
            chart_image_path=saved_path,
            is_chart=True,     # CNN이 이미 확인했으므로 True로 설정
        )

        final_state = await workflow_app.ainvoke(initial_state)

        # dict 또는 Pydantic 모두 처리
        def _get(s, key, default=None):
            if isinstance(s, dict):
                return s.get(key, default)
            return getattr(s, key, default)

        is_misleading  = _get(final_state, "is_misleading", False)
        visual_errors  = _get(final_state, "visual_errors",  []) or []
        verdict        = _get(final_state, "verdict",        "")
        explanation    = _get(final_state, "explanation",    "")
        confidence     = _get(final_state, "confidence",     "")
        quiz_question  = _get(final_state, "quiz_question",  "")
        quiz_answer    = _get(final_state, "quiz_answer",    "")
        quiz_expl      = _get(final_state, "quiz_explanation", "")

        result.update({
            "is_misleading":    is_misleading,
            "visual_errors":    visual_errors,
            "verdict":          verdict,
            "explanation":      explanation,
            "confidence":       confidence,
            "quiz_question":    quiz_question,
            "quiz_answer":      quiz_answer,
            "quiz_explanation": quiz_expl,
        })

        status = "⚠️ 오류 발견" if is_misleading else "✅ 정상"
        print(f"[AI 분석 완료] {status} | 오류: {visual_errors}")

    except Exception as e:
        print(f"[AI 분석 오류] {e}")
        result.update({
            "is_misleading":    False,
            "visual_errors":    [],
            "quiz_question":    "",
            "quiz_answer":      "",
            "quiz_explanation": "",
            "error":            str(e),
        })

    return result
