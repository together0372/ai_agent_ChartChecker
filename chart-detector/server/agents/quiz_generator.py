# agents/quiz_generator.py
from typing import Any, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from state import ChartCheckState

class QuizGeneratorAgent:
    """판정된 왜곡을 바탕으로 OX 퀴즈를 생성하는 에이전트"""

    def __init__(self, llm: ChatOllama):
        self.name = "Chart QuizGenerator"
        self.llm = llm

    async def generate_quiz_node(self, state: ChartCheckState) -> Dict[str, Any]:
        """최종 판정 결과를 바탕으로 OX 퀴즈 생성"""
        
        # 왜곡이 없는 정상 차트라면 퀴즈를 만들지 않습니다.
        if not state.is_misleading:
            return {
                "quiz_question": "",
                "quiz_answer": "",
                "quiz_explanation": ""
            }

        print("=======퀴즈 생성 시작=========")

        # AI에게 퀴즈를 만들도록 지시하는 프롬프트
        messages = [
            SystemMessage(content="<|think|>당신은 사용자의 직관적 오류를 깨우쳐주는 교육용 퀴즈 출제자입니다."),
            HumanMessage(content=f"""
            아래의 '차트 왜곡 분석 결과'를 바탕으로, 기사를 읽는 독자가 착각하기 쉬운 매력적인 O/X 퀴즈를 하나 만들어주세요.

            [차트 왜곡 분석 결과]
            설명: {state.explanation}

            [퀴즈 제작 조건]
            1. 질문(question)은 반드시 "결론은 OOO이다?" 같은 형태여야 합니다. (독자가 시각적으로 속았을 때 내릴 법한 결론을 질문으로 만드세요)
            2. 정답(answer)은 무조건 "O" 또는 "X" 둘 중 하나여야 합니다.
            3. 해설(explanation)은 독자가 왜 속았는지 아주 쉽고 친절하게 1~2문장으로 설명해주세요.
            
            반드시 아래 형식에 맞춰서 출력하세요. 다른 텍스트는 포함하지 마세요.
            
            질문: (여기에 질문)
            정답: (O 또는 X)
            해설: (여기에 해설)
            """)
        ]

        try:
            response = self.llm.invoke(messages).content.strip()
            
            # 파싱 로직 (질문, 정답, 해설 분리)
            q = ""
            a = ""
            e = ""
            
            for line in response.split('\n'):
                if line.startswith("질문:"): q = line.replace("질문:", "").strip()
                elif line.startswith("정답:"): a = line.replace("정답:", "").strip()
                elif line.startswith("해설:"): e = line.replace("해설:", "").strip()

            print(f"생성된 퀴즈: {q} ({a})")
            print("============================\n")

            return {
                "quiz_question": q,
                "quiz_answer": a,
                "quiz_explanation": e
            }

        except Exception as e:
            return {"error": f"퀴즈 생성 실패: {str(e)}"}