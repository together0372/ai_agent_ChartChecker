from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from state import ChartCheckState
from agents.distortion_detector import DistortionDetectorAgent


def create_news_workflow(llm: ChatOllama = None) :

    # 각 작업을 담당할 에이전트 인스턴스 생성
    distortion = DistortionDetectorAgent(llm)  # 차트 왜곡 탐색
    #ex ) summarizer = ChartSummarizerAgent(llm)  # AI 요약 생성 전담

    # ChartCheckState를 state객체로 사용하는 워크플로우 그래프 생성
    workflow = StateGraph(ChartCheckState)

    # 각 에이전트의 메서드를 워크플로우 노드로 등록
    workflow.add_node("distort", distortion.distortion_detector_grape)
    #ex) workflow.add_node("summarize", summarizer.summarize_news)

    # 워크플로우 실행 순서 정의 (순차적 파이프라인)
    workflow.add_edge(START, "distort")   # 시작점 설정
    #ex) workflow.add_edge("collect", "summarize")  # 수집 → 요약
    workflow.add_edge("distort", END)  # 보고서 → 종료

    # 실행 가능한 워크플로우 객체로 컴파일하여 반환
    return workflow.compile()