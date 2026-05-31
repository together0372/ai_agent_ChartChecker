import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama

from detector import analyze_chart
from workflow import create_news_workflow
from state import ChartCheckState

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 서버 시작 시 AI 모델 미리 로드해두기
llm = ChatOllama(
    model=" gemma4:e4b-it-q4_K_M",
    temperature=0.1,
    top_p=0.95,
    top_k=64
)
workflow_app = create_news_workflow(llm)


class ImageRequest(BaseModel):
    url: str
    page: str
    site: str
    title: str = ""

@app.post("/analyze")
async def analyze(req: ImageRequest):

    # 2. 먼저 OpenCV + Tesseract로 차트인지 검사
    result = analyze_chart(
        req.url,
        req.page,
        req.site,
        req.title
    )

    # 3. 차트가 아니면 프론트엔드로 그냥 결과 반환
    if not result.get("is_chart"):
        return result

    # 4. 차트가 맞다면 AI 에이전트 실행
    saved_image_path = result.get("saved")
    print(f"\n[AI 분석 시작] 발견된 차트 이미지: {saved_image_path}")

    try:
        # AI에게 넘겨줄 상태 생성
        initial_state = ChartCheckState(chart_image_path=saved_image_path)

        # 워크플로우 비동기 실행
        final_state = await workflow_app.ainvoke(initial_state)

        # 5. 기존 결과에 AI가 만든 퀴즈 데이터를 합쳐서 익스텐션으로 보냄
        result["is_misleading"] = final_state.get("is_misleading", False)
        result["quiz_question"] = final_state.get("quiz_question", "")
        result["quiz_answer"] = final_state.get("quiz_answer", "")
        result["quiz_explanation"] = final_state.get("quiz_explanation", "")

        print(f"[AI 분석 완료] 퀴즈 생성 성공: {result['quiz_question']}")

    except Exception as e:
        print(f"AI 분석 중 에러 발생: {e}")
        result["is_misleading"] = False
        result["error"] = str(e)

    return result
