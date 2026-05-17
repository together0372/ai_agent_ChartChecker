
import os
import re
import json
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

from state import ChartCheckState
from config import Config


class DistortionDetectorAgent:
    """차트에서 왜곡된 부분을 찾는 에이전트"""

    def __init__(self, llm: ChatOllama):
        self.name = "Chart DistortionDetector"
        self.llm = llm
        # ① 튜플 형식의 메시지로 간결하게 프롬프트 템플릿 구성
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",  # ② 시스템 역할 메시지로 AI의 행동 지침 설정
                    """당신은 전문 뉴스 요약 전문가입니다. 
                    주어진 뉴스를 핵심만 간결하게 2-3문장으로 요약해주세요.
                    - 사실만을 전달하고 추측은 피하세요
                    - 중요한 숫자나 날짜는 포함하세요
                    - 명확하고 이해하기 쉽게 작성하세요""",
                ),
                (
                    "human",  # ③ 사용자 메시지 템플릿에 변수 플레이스홀더 포함
                    "제목: {title}\n내용: {content}\n\n위 뉴스를 2-3문장으로 요약해주세요:",
                ),
            ]
        )
    async def analyze_chart_content_node(self, state: ChartCheckState) -> Dict[str, Any]:
        """Gemma 4 멀티모달로 차트의 내용을 텍스트로 읽어냄"""
        image_path = state.chart_image_path
        if not image_path or not os.path.exists(image_path):
            return {"error": "차트 이미지 없음"}


        try:
            #img_b64 = _encode_image(image_path)

            messages = [
                SystemMessage(content="<|think|>당신은 데이터 시각화 분석 전문가입니다."),
                HumanMessage(content=[
                    # 이미지 먼저
                    {"type": "image_url", "image_url": {"url": f"{image_path}"}},
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

            response = self.llm.invoke(messages)
            description = response.content

            print(description, "\n")

            return {"chart_description": description}

        except Exception as e:
            return {"error": f"차트 분석 실패: {str(e)}"}

    async def check_visual_errors_node(self, state: ChartCheckState) -> Dict[str, Any]:
        """시각적 오류 검사 (잘린 축, 왜곡된 비율 등)"""
        description = state.chart_description
        image_path = state.chart_image_path

        if not image_path:
            return { "visual_errors": []}

        try:

            messages = [
                SystemMessage(content="<|think|>당신은 데이터 시각화 분석 전문가입니다."),
                HumanMessage(content=[
                    # 이미지 먼저 (세부 읽기용 높은 해상도)
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_path}",
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
            response = self.llm.invoke(messages)
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

    async def check_data_errors_node(self, state: ChartCheckState) -> Dict[str, Any]:
        """수치 오류 검사 (합계 불일치, 날짜 역전 등)"""
        description = state.chart_description
        image_path = state.chart_image_path

        messages = [
            SystemMessage(content="<|think|>당신은 데이터 시각화 분석 전문가입니다."),
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{image_path}",
                        "max_tokens": 1120  # OCR/축 읽기용 최고 품질
                    }
                },
                {"type": "text", "text": f"""차트 분석 결과:
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
        출력 형식: ["오류 유형: 구체적 설명"]"""}])
        ]


        try:
            response = self.llm.invoke(messages)
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

    async def final_verdict_node(self, state: ChartCheckState) -> Dict[str, Any]:
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
            response = self.llm.invoke(messages).content.strip()

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

    async def build_graph(self):
        graph = StateGraph(ChartCheckState)

        # 노드 등록
        graph.add_node("analyze_content", self.analyze_chart_content_node)
        graph.add_node("check_visual", self.check_visual_errors_node)
        graph.add_node("check_data", self.check_data_errors_node)
        graph.add_node("final_verdict", self.final_verdict_node)


        # analyze → 시각/수치 검사 (순차)
        graph.add_edge(START, "analyze_content")
        graph.add_edge("analyze_content", "check_visual")
        graph.add_edge("check_visual", "check_data")
        graph.add_edge("check_data", "final_verdict")
        graph.add_edge("final_verdict", END)


        return graph.compile()

    async def distortion_detector_grape(self, state: ChartCheckState) :
        print("=======차트 분석 시작=========\n")
        app = await self.build_graph()

        final_state = await app.ainvoke(state)

        # print(final_state["response"])
        print("\n=========차트 분석 완료=============")

        return final_state

