import os
import time
import uuid
import asyncio
import aiohttp
import json
import re
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# 팀원이 작성한 LangGraph 모듈 (수정 없이 그대로 사용!)
from workflow import create_news_workflow
from state import ChartCheckState
from agents.core.rag_utils import init_vector_db

# 우리가 작성했던 CNN 판별 모듈
try:
    from cnn import is_chart_image
except ImportError:
    print("⚠️ cnn.py 모듈을 찾을 수 없어 CNN 1차 판별은 무조건 통과로 처리합니다.")
    def is_chart_image(img_path): return True, 0.99

app = FastAPI(title="ChartQuiz AI Server")

# 크롬 익스텐션과 통신하기 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
ai_workflow = None
llm = None
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    global ai_workflow, llm
    print("\n🚀 ChartQuiz FastAPI 서버 시작 중...")
    
    # 1. RAG 벡터 DB 초기화 (필요시 작동)
    # init_vector_db() 
    
    # 2. 통일하기로 한 qwen3.5:9b 모델 로드
    llm = ChatOllama(model="qwen3.5:9b", temperature=0.1)
    
    # 3. 팀원이 만든 LangGraph 워크플로우 엔진 장착
    ai_workflow = create_news_workflow(llm, use_rag=False) # 로컬 속도를 위해 우선 False 설정
    print("✅ 시스템 준비 완료! 크롬 익스텐션의 요청을 기다립니다.\n")


# 프론트엔드에서 날아오는 JSON 데이터 형식
class AnalyzeRequest(BaseModel):
    url: str
    page: str
    site: str

async def download_image(url: str) -> str:
    """익스텐션이 보낸 URL의 이미지를 서버 임시 폴더에 다운로드"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.read()
                    ext = ".png"
                    if "jpeg" in url.lower() or "jpg" in url.lower(): ext = ".jpg"
                    filename = f"{uuid.uuid4().hex[:8]}{ext}"
                    filepath = TEMP_DIR / filename
                    with open(filepath, "wb") as f:
                        f.write(content)
                    return str(filepath)
    except Exception as e:
        print(f"이미지 다운로드 에러: {e}")
    return ""

@app.post("/analyze")
async def analyze_chart(req: AnalyzeRequest):
    print(f"\n📥 [요청 수신] {req.site} 에서 이미지 도착!")
    
    # 1. 이미지 다운로드
    img_path = await download_image(req.url)
    if not img_path:
        return {"is_chart": False, "error": "다운로드 실패"}

    # 2. CNN 1차 판별 (0.1초 컷)
    is_chart, cnn_conf = is_chart_image(img_path)
    if not is_chart:
        print(f"  ❌ [CNN 거절] 차트 아님 (확신도: {cnn_conf:.2f})")
        return {"is_chart": False, "cnn_confidence": cnn_conf}

    print(f"  ✅ [CNN 통과] 차트 확인! LangGraph 멀티에이전트 분석 시작...")

    # 3. 팀원의 LangGraph 워크플로우(2차 판별) 실행
    state = ChartCheckState(chart_image_path=img_path)
    result = await ai_workflow.ainvoke(state)
    
    # 반환값이 dict인 경우 Pydantic으로 변환하여 안전하게 접근
    if isinstance(result, dict):
        final_state = ChartCheckState(**result)
    else:
        final_state = result

    # 워크플로우 내의 ChartClassifier가 비차트로 판별한 경우
    if not final_state.is_chart:
        print(f"  ❌ [LLM 거절] {final_state.skip_reason}")
        return {"is_chart": False, "reason": final_state.skip_reason}

    # 4. 퀴즈 에이전트: 왜곡이 발견되었다면 O/X 퀴즈 생성!
    quiz_question = "이 차트에는 과장되거나 생략된 정보가 있다?"
    quiz_answer = "O"
    quiz_explanation = final_state.explanation

    if final_state.is_misleading:
        print("  💡 왜곡 발견! O/X 퀴즈 생성 중...")
        quiz_sys = "주어진 차트 오류를 바탕으로 일반인을 위한 흥미로운 O/X 퀴즈를 생성하세요. 반드시 JSON 형식으로만 출력하세요. {'question': '퀴즈 질문(O/X로 답할 수 있게)', 'answer': 'O' 또는 'X'}"
        quiz_user = f"발견된 오류 목록: {final_state.visual_errors}\n상세 설명: {final_state.explanation}"
        
        try:
            # 퀴즈용 LLM 가볍게 호출
            quiz_llm = ChatOllama(model="qwen3.5:9b", temperature=0.3)
            resp = await asyncio.to_thread(quiz_llm.invoke, [SystemMessage(content=quiz_sys), HumanMessage(content=quiz_user)])
            
            # JSON만 정교하게 파싱
            m = re.search(r"\{[\s\S]*\}", resp.content)
            if m:
                q_data = json.loads(m.group(0))
                quiz_question = q_data.get("question", quiz_question)
                quiz_answer = q_data.get("answer", quiz_answer)
        except Exception as e:
            print(f"  ⚠️ 퀴즈 생성 실패(기본값 사용): {e}")

    print(f"  🎯 [최종 완료] 왜곡 여부: {final_state.is_misleading}, 판정: {final_state.verdict}")

    # 익스텐션 UI가 기다리고 있는 JSON 규격대로 응답 쏴주기
    return {
        "is_chart": True,
        "is_misleading": final_state.is_misleading,
        "confidence": final_state.confidence,
        "quiz_question": quiz_question,
        "quiz_answer": quiz_answer,
        "quiz_explanation": quiz_explanation
    }