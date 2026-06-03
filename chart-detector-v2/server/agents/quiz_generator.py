# agents/quiz_generator.py
import os
from typing import Any, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from state import ChartCheckState


class QuizGeneratorAgent:
    """판정된 왜곡을 바탕으로 OX 퀴즈를 생성하는 에이전트"""

    def __init__(self, llm: ChatOllama = None):
        self.name = "Chart QuizGenerator"
        # llm이 None이면 config에서 모델명을 읽어 자체 생성
        if llm is None:
            model = os.environ.get("CHART_MODEL", "qwen3.5:9b")
            llm = ChatOllama(model=model, temperature=0.3, num_predict=400)
        self.llm = llm

    async def generate_quiz_node(self, state: ChartCheckState) -> Dict[str, Any]:
        """최종 판정 결과를 바탕으로 OX 퀴즈 생성"""

        # 왜곡이 없는 정상 차트라면 퀴즈 생성 생략
        if not state.is_misleading:
            return {"quiz_question": "", "quiz_answer": "", "quiz_explanation": ""}

        print("=======퀴즈 생성 시작=========")

        # ── 오류 정보 요약 (긴 마크다운 보고서 대신 핵심만) ──
        visual_errors = state.visual_errors or []
        errors_str = "\n".join(f"- {e}" for e in visual_errors) if visual_errors else "왜곡 기법 발견됨"

        # explanation이 길면 앞 400자만 사용
        explanation_snippet = (state.explanation or "")[:400]

        messages = [
            SystemMessage(content="당신은 사용자의 직관적 오류를 깨우쳐주는 교육용 퀴즈 출제자입니다."),
            HumanMessage(content=f"""아래 차트 왜곡 분석 결과를 바탕으로, 독자가 착각하기 쉬운 OX 퀴즈를 하나 만드세요.

[발견된 왜곡 기법]
{errors_str}

[분석 요약]
{explanation_snippet}

[퀴즈 제작 조건]
1. 질문은 "결론은 OOO이다?" 형태로, 독자가 차트만 보고 내릴 법한 잘못된 결론을 담으세요.
2. 정답은 "O" 또는 "X" 중 하나입니다. (왜곡된 차트의 메시지가 실제로 틀렸으면 X)
3. 해설은 왜 독자가 속았는지 1~2문장으로 쉽게 설명하세요.

반드시 아래 형식으로만 출력하세요:

질문: (질문 내용)
정답: (O 또는 X)
해설: (해설 내용)""")
        ]

        try:
            response = self.llm.invoke(messages).content.strip()

            q = ""
            a = ""
            e = ""

            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("질문:"):
                    q = line[len("질문:"):].strip()
                elif line.startswith("정답:"):
                    a = line[len("정답:"):].strip().replace("(", "").replace(")", "").strip()
                    # "O 또는 X" 형태 → 첫 글자만 추출
                    a = next((c for c in a if c in "OX"), "")
                elif line.startswith("해설:"):
                    e = line[len("해설:"):].strip()

            # 기본값 보정
            if not a:
                a = "X"

            print(f"생성된 퀴즈: {q} (정답: {a})")
            print("============================\n")

            return {
                "quiz_question":    q,
                "quiz_answer":      a,
                "quiz_explanation": e,
            }

        except Exception as err:
            print(f"퀴즈 생성 실패: {err}")
            return {
                "quiz_question":    "이 차트에는 오해를 유발하는 요소가 있다?",
                "quiz_answer":      "O",
                "quiz_explanation": "AI가 퀴즈를 생성하지 못했습니다.",
            }