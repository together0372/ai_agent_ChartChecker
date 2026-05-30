from typing import Any, Dict
from pydantic import BaseModel, Field


class ChartCheckState(BaseModel):
    # === 입력 ===
    chart_image_path: str = ""

    # === 차트 판별 ===
    is_chart:    bool = False   # True면 차트/그래프로 판별됨
    skip_reason: str  = ""      # 차트가 아닐 때 스킵 이유

    # === 중간 처리 ===
    chart_description: str = ""
    chart_type: str = ""

    # === 분석 결과 ===
    visual_errors: list[str] = []   # 한국어 오류명 목록
    data_errors:   list[str] = []

    # === 최종 판정 ===
    is_misleading: bool = False
    verdict: str = ""
    explanation: str = ""
    confidence: str = ""

    # === core 시스템 상세 결과 ===
    final_misleaders: list[str] = []          # 내부 misleader 키 목록
    self_confidence:  Dict[str, Any] = Field(default_factory=dict)

    # === 에러 처리 ===
    error: str = ""
