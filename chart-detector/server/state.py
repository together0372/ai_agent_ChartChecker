from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict


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

    # === 퀴즈 데이터 ===
    quiz_question: str = "" # 예: "이 차트의 실제 변화량은 2배 이상이다?"
    quiz_answer: str = ""   # 예: "X"
    quiz_explanation: str = "" # 예: "Y축이 잘려 있어서 시각적으로 과장되었습니다."