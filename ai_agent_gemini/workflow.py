from langgraph.graph import StateGraph, START, END

from state import ChartCheckState
from agents.chart_classifier import ChartClassifierAgent
from agents.distortion_detector import DistortionDetectorAgent


def _route_after_classify(state: ChartCheckState) -> str:
    """
    차트 판별 결과에 따라 다음 노드 결정.
      - 차트  → "distort" (왜곡 검사)
      - 비차트 → END (즉시 종료)
    """
    if state.is_chart:
        return "distort"
    return END


def create_news_workflow(llm=None, use_rag: bool = True):
    """
    llm 파라미터는 하위 호환성을 위해 유지하지만 사용하지 않음.
    Gemini API LLM은 agents/core/agent.py 내부에서 ChatGoogleGenerativeAI로 생성된다.
    """
    classifier = ChartClassifierAgent()
    distortion = DistortionDetectorAgent(llm=None, use_rag=use_rag)

    workflow = StateGraph(ChartCheckState)

    # 노드 등록
    workflow.add_node("classify", classifier.classify)
    workflow.add_node("distort",  distortion.distortion_detector_grape)

    # 실행 순서
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"distort": "distort", END: END},
    )
    workflow.add_edge("distort", END)

    return workflow.compile()
