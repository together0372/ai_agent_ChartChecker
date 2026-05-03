import base64
import os
import re
import json
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from PIL import Image, ImageEnhance


class ChartCheckState(BaseModel):
    # === 입력 ===
    #input_url: str = ""  # 뉴스 URL (URL 입력 시)
    #input_image_path: str = "" # 이미지 경로 (직접 업로드 시)

    # === 중간 처리 ===
    #raw_html: str = "" # 크롤링된 HTML
    #article_text: str = "" # 기사 본문 텍스트
    chart_image_path: str = ""  # 추출/저장된 차트 이미지 경로
    chart_description: str = "" # Gemma 4가 읽어낸 차트 내용

    # === 분석 결과 ===
    visual_errors: list[str] = [] # 시각적 오류 목록
    data_errors: list[str] = [] # 수치 데이터 오류 목록

    # === 최종 판정 ===
    is_misleading: bool = False  # 오류 여부
    verdict: str = "" # 최종 판정 텍스트
    explanation: str = ""  # 상세 설명
    confidence: str = ""  # 확신도 (높음/중간/낮음)

    # === 에러 처리 ===
    error: str = ""

# ──────────────────────────────────────────
# 유틸: 이미지를 base64로 인코딩
# ──────────────────────────────────────────
def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@tool
def calculate(expression: str) -> str:
    """
    수학 표현식을 계산한다.
    사칙연산, 거듭제곱, 나머지 연산을 지원한다.
    예: "2 + 3", "10 * 5", "2 ** 8", "10 % 3"
    """
    try:
        # eval 안전하게 제한 (수학 연산만 허용)
        allowed = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "pow": pow,
            "max": max,
            "min": min,
        }
        result = eval(expression, allowed)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "오류: 0으로 나눌 수 없습니다"
    except Exception as e:
        return f"오류: 계산할 수 없는 표현식입니다 ({str(e)})"


# Gemma 4 멀티모달 모델 (Vision 지원)
_llm = ChatOllama(model="gemma4:e4b-it-q4_K_M", temperature=0.1, top_p=0.95, top_k=64)
#_llm = ChatOllama(model="gemma4:26b", temperature=0.1, top_p=0.95, top_k=64)
llm = _llm.bind_tools([calculate])


# ──────────────────────────────────────────
# Node : Gemma 4 Vision으로 차트 내용 파악
# ──────────────────────────────────────────

def analyze_chart_content_node(state: ChartCheckState) -> Dict[str, Any]:
    """Gemma 4 멀티모달로 차트의 내용을 텍스트로 읽어냄"""
    image_path = state.chart_image_path
    if not image_path or not os.path.exists(image_path):
        return {"error": "차트 이미지 없음"}

    try:
        img_b64 = _encode_image(image_path)

        messages = [
            SystemMessage(content="<|think|>당신은 데이터 시각화 분석 전문가입니다."),
            HumanMessage(content=[
                # 이미지 먼저
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                # 텍스트 나중에
                {"type": "text", "text": """차트 이미지를 보고 아래 항목을 하나씩 확인해서 JSON으로만 출력하세요. 다른 텍스트는 절대 포함하지 마세요.

        확인 항목:
        1. 차트 종류는 무엇인가? (막대/꺾은선/원형/산점도/지도 등)
        2. 차트 제목은 무엇인가?
        3. X축 라벨과 값의 범위는?
        4. Y축 라벨과 값의 범위는? (시작값을 반드시 포함)
        5. Y축은 정확히 몇에서 시작하는가? (숫자만, 확인 불가면 null)
        6. 주요 데이터 포인트는 무엇인가? (수치 포함)
        7. 범례 내용은 무엇인가?
        8. 데이터 출처가 표시되어 있는가?
        9. 퍼센트 값이 표시되어 있다면 모두 나열하라
        10. 시계열 데이터인가? (true/false)
        11. 시간 순서가 오름차순인가, 내림차순인가, 불규칙한가?
        12. Y축이 두 개인가? 있다면 각각의 스케일은?

        출력 형식 (JSON만):
        {
          "chart_type": "",
          "title": "",
          "x_axis": "",
          "y_axis": "",
          "y_axis_start": null,
          "data_points": [],
          "legend": "",
          "source": "",
          "percentages": [],
          "time_series": false,
          "time_order": "",
          "dual_axis": ""
        }"""}
            ])
        ]

        response = llm.invoke(messages)
        description = response.content

        print(description, "\n")

        return {"chart_description": description}

    except Exception as e:
        return {"error": f"차트 분석 실패: {str(e)}"}


# ──────────────────────────────────────────
# Node : 시각적 오류 검사
# ──────────────────────────────────────────
def check_visual_errors_node(state: ChartCheckState) -> Dict[str, Any]:
    """시각적 오류 검사 (잘린 축, 왜곡된 비율 등)"""
    description = state.chart_description
    image_path = state.chart_image_path

    if not image_path:
        return { "visual_errors": []}

    try:
        img_b64 = _encode_image(image_path)

        messages = [
            SystemMessage(content="<|think|>당신은 데이터 시각화 분석 전문가입니다."),
            HumanMessage(content=[
                # 이미지 먼저 (세부 읽기용 높은 해상도)
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                        "max_tokens": 1120  # OCR/축 읽기용 최고 품질
                    }
                },
                {"type": "text", "text": f"""차트 분석 결과:
        {description}

        아래 항목을 하나씩 검사하세요.

        검사 항목:
        1. Y축이 0이 아닌 값에서 시작해서 변화량을 과장하는가? (잘린 축)
        2. 두 개의 독립적인 Y축이 서로 다른 스케일로 존재하는가? (이중 축)
        3. 3D 효과로 인해 비율이 왜곡되는가? (3D 효과)
        4. 막대/원형 차트의 시각적 크기가 실제 수치에 비례하지 않는가? (수치 왜곡)
        5. Y축이 위→아래로 증가하거나 X축이 우→좌로 증가하는가? (반전된 축)
        6. 축 범위가 너무 넓거나 좁아서 데이터 패턴을 왜곡하는가? (부적절한 축 범위)
        7. 지도에서 연속 변수를 이산 범주로 잘라 경계값 차이를 과장하는가? (비연속 변수의 이산화)

        발견된 오류만 JSON 배열로 출력하세요. 없으면 빈 배열 []. 최대 3개.
        출력 형식: ["오류 유형: 구체적 설명"]"""}
            ])
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()

        #print(raw, "\n")

        # JSON 파싱 시도
        try:
            json_match = re.search(r'\[.*?\]', raw, re.DOTALL)
            errors = json.loads(json_match.group()) if json_match else []
        except Exception:
            errors = [raw] if raw and raw != "[]" else []

        return { "visual_errors": errors}

    except Exception as e:
        return {"visual_errors": [f"시각 검사 오류: {str(e)}"]}


# ──────────────────────────────────────────
# Node : 수치 데이터 오류 검사
# ──────────────────────────────────────────
def check_data_errors_node(state: ChartCheckState) -> Dict[str, Any]:
    """수치 오류 검사 (합계 불일치, 날짜 역전 등)"""
    description = state.chart_description

    messages = [
        SystemMessage(content="<|think|>당신은 데이터 시각화 분석 전문가입니다."),
        HumanMessage(content=f"""차트 분석 결과:
    {description}


    아래 항목을 하나씩 검사하세요.

    검사 항목:
    1. 원형 차트의 퍼센트 합계가 100%가 아닌가? (부적절한 원형 차트 사용)
    2. 날짜/시간 순서가 비연대기적으로 배열되어 있는가? (부적절한 항목 순서)
    3. 연도·나이 등의 구간 크기가 불균등한가? (불일치하는 구간 크기)
    4. 눈금이 시각적으로 균등하나 실제 값 간격이 다른가? (불일치하는 눈금 간격)
    5. 범주형 변수에 꺾은선 차트를 쓰거나 시간 축이 Y축에 있는가? (부적절한 꺾은선 차트 사용)
    6. 기사 본문의 수치와 차트에 표시된 수치가 다른가? (수치 왜곡)

    발견된 오류만 JSON 배열로 출력하세요. 없으면 빈 배열 []. 최대 3개.
    출력 형식: ["오류 유형: 구체적 설명"]""")
    ]


    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        #print(raw, "\n")

        try:
            json_match = re.search(r'\[.*?\]', raw, re.DOTALL)
            errors = json.loads(json_match.group()) if json_match else []
        except Exception:
            errors = [raw] if raw and raw != "[]" else []

        return { "data_errors": errors}

    except Exception as e:
        return { "data_errors": [f"수치 검사 오류: {str(e)}"]}


# ──────────────────────────────────────────
# Node : 최종 판정
# ──────────────────────────────────────────
def final_verdict_node(state: ChartCheckState) -> Dict[str, Any]:
    """모든 오류를 종합해서 최종 판정"""
    visual_errors = state.visual_errors
    data_errors = state.data_errors
    all_errors = visual_errors + data_errors

    is_misleading = len(all_errors) > 0

    if not all_errors:
        verdict = "✅ 정상"
        explanation = "시각적 오류와 수치 오류가 발견되지 않았습니다. 차트가 데이터를 정확하게 표현하고 있습니다."
        confidence = "높음"
    else:
        verdict = "⚠️ 오류 발견"
        error_list = "\n".join(f"- {e}" for e in all_errors)

        messages = [
            SystemMessage(content="<|think|>당신은 데이터 시각화 분석 전문가입니다."),
            HumanMessage(content=f"""다음 차트 misleader(오해 유발 요소)들이 발견되었습니다:

        {error_list}

        아래 두 가지를 순서대로 출력하세요. 다른 텍스트는 절대 포함하지 마세요.

        1. 확신도: "높음", "중간", "낮음" 중 하나
           - 높음: 잘린 축, 수치 왜곡, 이중 축, 반전된 축, 부적절한 원형 차트처럼 오해가 명백한 경우
           - 중간: 오해 가능성은 있으나 의도적인지 불분명한 경우
           - 낮음: 관습 차이일 수 있는 경우

        2. 설명: 이 misleader들이 독자에게 어떤 오해를 줄 수 있는지 2-3문장으로 설명

        출력 형식 (반드시 이 형식 그대로):
        확신도: 높음
        설명: 여기에 설명 작성""")
        ]
        response = llm.invoke(messages).content.strip()

        # 확신도 파싱
        if "높음" in response:
            confidence = "높음"
        elif "낮음" in response:
            confidence = "낮음"
        else:
            confidence = "중간"

        # 설명 파싱
        if "설명:" in response:
            explanation = response.split("설명:")[-1].strip()
        else:
            explanation = response

    return {
        "is_misleading": is_misleading,
        "verdict": verdict,
        "explanation": explanation,
        "confidence": confidence,
    }



# ──────────────────────────────────────────
# 그래프 빌드
# ──────────────────────────────────────────
def build_graph():
    graph = StateGraph(ChartCheckState)

    # 노드 등록
    graph.add_node("analyze_content", analyze_chart_content_node)
    graph.add_node("check_visual", check_visual_errors_node)
    graph.add_node("check_data", check_data_errors_node)
    graph.add_node("final_verdict", final_verdict_node)


    # analyze → 시각/수치 검사 (순차)
    graph.add_edge(START, "analyze_content")
    graph.add_edge("analyze_content", "check_visual")
    graph.add_edge("check_visual", "check_data")
    graph.add_edge("check_data", "final_verdict")
    graph.add_edge("final_verdict", END)


    return graph.compile()




def main():
    print("=======차트 분석 시작=========\n")
    app = build_graph()

    for i in range(4):
        final_state = app.invoke(ChartCheckState(chart_image_path = f"C:/Users/toget/OneDrive/Desktop/NLP/ai_agent/test/test_image{i}.png"))
        print(final_state["is_misleading"], "\n")
        print(final_state["verdict"], "\n")
        print(final_state["explanation"], "\n")
        print(final_state["confidence"], "\n")

    #print(final_state["response"])
    print("\n=========차트 분석 완료=============")

if __name__ == "__main__":
    main()