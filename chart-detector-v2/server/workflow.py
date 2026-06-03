"""
워크플로우 정의 (통합 버전)

흐름:
  START → distort (멀티에이전트 오류 탐지) → quiz (OX 퀴즈 생성) → END

NOTE: CNN 기반 차트 판별은 main.py에서 workflow 진입 전에 수행.
      워크플로우는 차트로 확인된 이미지만 받으므로 classify 노드 없음.
"""
from langgraph.graph import StateGraph, START, END

from state import ChartCheckState
from agents.distortion_detector import DistortionDetectorAgent
from agents.quiz_generator import QuizGeneratorAgent


def create_news_workflow(llm=None, use_rag: bool = True):
    """
    통합 차트 분석 워크플로우 생성.

    Args:
        llm:     ChatOllama 인스턴스 (quiz_generator용). None이면 내부 생성.
        use_rag: MisViz 벡터DB 기반 RAG 사용 여부.
    """
    # ── 에이전트 인스턴스 생성 ──────────────────────────────
    # core DistortionDetectorAgent는 내부에서 LLM을 직접 생성하므로 llm 불필요
    distortion = DistortionDetectorAgent(llm=None, use_rag=use_rag)
    quiz       = QuizGeneratorAgent(llm)

    # ── 그래프 구성 ────────────────────────────────────────
    workflow = StateGraph(ChartCheckState)

    workflow.add_node("distort", distortion.distortion_detector_grape)
    workflow.add_node("quiz",    quiz.generate_quiz_node)

    workflow.add_edge(START,    "distort")
    workflow.add_edge("distort", "quiz")
    workflow.add_edge("quiz",    END)

    return workflow.compile()
