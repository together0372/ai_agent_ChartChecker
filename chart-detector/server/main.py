import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama

from cnn import analyze_chart as cnn_analyze_chart
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

@app.post("/analyze")
async def analyze(req: ImageRequest):
    
    # 2. 먼저 OpenCV + Tesseract로 차트인지 검사
    cnn_result = cnn_analyze_chart(req.url, req.page, req.site)

    # 3. 차트가 아니면 프론트엔드로 그냥 결과 반환
    if not cnn_result.get("success") or not cnn_result.get("is_chart"):
        return {
            "success": cnn_result.get("success", False),
            "is_chart": False,
            "cnn_confidence": cnn_result.get("confidence", 0.0),
            "reason": cnn_result.get("reason", cnn_result.get("error", "Not a chart image"))
        }
    
    # 4. CNN 관문을 완벽히 통과한 진짜 차트 정보 확보
    saved_image_path = cnn_result.get("saved")
    cnn_confidence = cnn_result.get("confidence", 0.0)
    print(f"\n[AI 분석 시작] 발견된 차트 이미지: {saved_image_path} (CNN 신뢰도: {cnn_confidence})")

    result = {
        "success": True,
        "is_chart": True,
        "cnn_confidence": float(cnn_confidence),
        "rule_score": 3,          # 비교용 룰 베이스 점수 기본값
        "rule_is_chart": True,
        "agree": True,
        "saved": saved_image_path
    }

    try:
        # 4. [우리 코드] 진짜 차트이므로 2단계 심층 분석(LangGraph 에이전트) 가동
        initial_state = ChartCheckState(chart_image_path=saved_image_path)
        
        # 워크플로우 비동기 실행
        final_state = await workflow_app.ainvoke(initial_state)

        # 5. 팀원 결과 포맷에 우리가 프론트엔드 UI를 그릴 때 쓸 데이터들을 탑탑이 쌓아줍니다.
        result["is_misleading"] = final_state.get("is_misleading", False)
        result["quiz_question"] = final_state.get("quiz_question", "")
        result["quiz_answer"] = final_state.get("quiz_answer", "")
        result["quiz_explanation"] = final_state.get("quiz_explanation", "")
        
        # 🚨 중요: cnn_confidence(차트 확률)와 이름이 겹치지 않도록 주의합니다.
        # final_state의 왜곡 확신도("높음", "중간", "낮음")를 'confidence' 키로 그대로 유지하여 신호등 UI에 공급합니다.
        result["confidence"] = final_state.get("confidence", "중간")
        
        print(f"[AI 분석 완료] 퀴즈 생성 성공: {result['quiz_question']}")

    except Exception as e:
        print(f"AI 분석 중 에러 발생: {e}")
        result["is_misleading"] = False
        result["error"] = str(e)

    return result