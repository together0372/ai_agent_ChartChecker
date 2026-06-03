"""
ChartCheckState — 통합 상태 정의

ai_agent_1 ChartCheckState + chart-detector quiz 필드 통합
"""
from typing import Any, Dict
from pydantic import BaseModel, Field


class ChartCheckState(BaseModel):
    # === 입력 ===
    chart_image_path: str = ""

    # === 차트 판별 (CNN 결과는 workflow 시작 전 설정) ===
    is_chart:    bool = False
    skip_reason: str  = ""

    # === 중간 처리 ===
    chart_description: str = ""
    chart_type:        str = ""

    # === 분석 결과 ===
    visual_errors: list[str] = []   # 한국어 오류명 목록 (표시용)
    data_errors:   list[str] = []   # 수치 오류 목록

    # === 최종 판정 ===
    is_misleading: bool = False
    verdict:       str  = ""        # "정상" | "경고" | "오류" | "확인불가"
    explanation:   str  = ""        # 상세 마크다운 분석 보고서
    confidence:    str  = ""        # "높음" | "중간" | "낮음"

    # === core 시스템 상세 결과 ===
    final_misleaders: list[str]       = []
    self_confidence:  Dict[str, Any]  = Field(default_factory=dict)

    # === 에러 처리 ===
    error: str = ""

    # === 퀴즈 데이터 (QuizGeneratorAgent가 채움) ===
    quiz_question:    str = ""  # 예: "이 차트는 변화량을 2배 이상 과장했다?"
    quiz_answer:      str = ""  # "O" 또는 "X"
    quiz_explanation: str = ""  # 해설 텍스트
