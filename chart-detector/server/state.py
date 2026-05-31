from typing import Any, Dict
from pydantic import BaseModel, Field

class ChartCheckState(BaseModel):
    chart_image_path: str = ""
    is_chart: bool = False
    skip_reason: str = ""
    chart_description: str = ""
    chart_type: str = ""
    visual_errors: list[str] = []
    data_errors: list[str] = []
    is_misleading: bool = False
    verdict: str = ""
    explanation: str = ""
    confidence: str = ""
    final_misleaders: list[str] = []
    self_confidence: Dict[str, Any] = Field(default_factory=dict)
    error: str = ""