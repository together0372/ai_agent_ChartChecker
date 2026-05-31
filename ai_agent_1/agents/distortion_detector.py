"""
DistortionDetectorAgent — ai_agent 워크플로우용 서브에이전트 래퍼

외부 워크플로우(ai_agent)의 단일 LangGraph 노드로서,
내부적으로 Observer→Debate→Math→Specialist→Judge 파이프라인을 실행한다.
"""
from __future__ import annotations

from typing import Any, Dict

from .core.agent import DistortionDetectorAgent as _CoreAgent
from .core.config import MISLEADER_TAXONOMY


class DistortionDetectorAgent:
    """
    ai_agent 워크플로우용 서브에이전트.

    workflow.py가 이 클래스의 distortion_detector_grape 메서드를
    LangGraph 노드로 등록한다. 실제 분석은 core.agent의 멀티에이전트
    시스템이 수행하고, 결과를 ChartCheckState 필드로 매핑해 반환한다.
    """

    def __init__(self, llm=None, use_rag: bool = True):
        # core 시스템은 ChatOllama를 내부에서 생성하므로 llm 파라미터는 미사용
        self._core = _CoreAgent(use_rag=use_rag)

    async def distortion_detector_grape(self, state) -> Dict[str, Any]:
        """
        LangGraph 노드 진입점.

        1. core 멀티에이전트 시스템 실행 (Observer→Debate→Math→Specialist→Judge)
        2. AgentState → ChartCheckState 필드 매핑
        """
        image_path = state.chart_image_path

        print("\n=======차트 분석 시작 (멀티에이전트 서브시스템)=========")

        # ── core 분석 실행 ──────────────────────────────────
        result = await self._core.analyze(image_path)

        # ── 결과 매핑 ───────────────────────────────────────
        verdict          = result.get("verdict", "확인불가")
        is_misleading    = verdict in ("오류", "경고")
        final_misleaders = result.get("final_misleaders", [])
        explanation      = result.get("explanation", "")
        image_desc       = result.get("image_description", "")
        chart_type       = result.get("chart_type", "")

        # 자기신뢰도 → 한국어 등급 변환
        self_conf  = result.get("self_confidence", {})
        overall    = self_conf.get("overall", {})
        conf_score = overall.get("score", 0.0)
        if conf_score >= 0.70:
            confidence = "높음"
        elif conf_score >= 0.50:
            confidence = "중간"
        else:
            confidence = "낮음"

        # misleader 키 → 한국어 이름 목록 (visual_errors 필드에 표시)
        misleader_names = [
            MISLEADER_TAXONOMY.get(m, {}).get("name", m)
            for m in final_misleaders
        ]

        print("\n=========차트 분석 완료=============")

        return {
            "chart_description": image_desc,
            "chart_type":        chart_type,
            "visual_errors":     misleader_names,   # 한국어 오류명 목록
            "data_errors":       [],
            "is_misleading":     is_misleading,
            "verdict":           verdict,
            "explanation":       explanation,
            "confidence":        confidence,
            "final_misleaders":  final_misleaders,  # 내부 키 목록 (추가 필드)
            "self_confidence":   self_conf,          # 상세 신뢰도 (추가 필드)
        }
