"""
DistortionDetectorAgent — ai_agent 워크플로우용 서브에이전트 래퍼

외부 워크플로우(ai_agent)의 단일 LangGraph 노드로서,
내부적으로 Observer→Debate→Math→Specialist→Judge 파이프라인을 실행한다.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from .core.agent import DistortionDetectorAgent as _CoreAgent
from .core.config import MISLEADER_TAXONOMY


# ─────────────────────────────────────────────────────────
# explanation 파싱 헬퍼
# ─────────────────────────────────────────────────────────

def _parse_explanation(text: str) -> dict:
    """
    explanation 마크다운에서 필요한 필드만 추출한다.

    반환:
        found_errors   : [{"name": str, "evidence": str, "mislead_method": str}]
        recommendations: [{"title": str, "detail": str}]
        subtle_analysis: str  (🕵️ 섹션 전체 텍스트)
    """
    result = {
        "found_errors":    [],
        "recommendations": [],
        "subtle_analysis": "",
    }

    # ── 섹션 분리 ──────────────────────────────────────────
    # 헤더 기준으로 섹션 추출
    def _section(header_emoji: str) -> str:
        """헤더 포함 섹션을 다음 # 헤더 전까지 추출"""
        pattern = rf"#[^#].*?{re.escape(header_emoji)}.*?\n(.*?)(?=\n#\s|\Z)"
        m = re.search(pattern, text, re.S)
        return m.group(1).strip() if m else ""

    errors_section  = _section("🚨")
    subtle_section  = _section("🕵")
    recs_section    = _section("✅")

    # ── 🚨 발견된 조작 파싱 ────────────────────────────────
    # 패턴: 번호. **오류명** ... **구체적 증거**: ... **독자 오도 방법**: ...
    error_blocks = re.split(r"\n\d+\.\s+", "\n" + errors_section)
    for block in error_blocks:
        if not block.strip():
            continue

        name_m     = re.search(r"\*\*(.+?)\*\*", block)
        evidence_m = re.search(r"\*\*구체적 증거\*\*\s*:\s*(.+?)(?=\*\*|$)", block, re.S)
        mislead_m  = re.search(r"\*\*독자 오도 방법\*\*\s*:\s*(.+?)(?=\*\*|$)", block, re.S)

        if not name_m:
            continue

        result["found_errors"].append({
            "name":          name_m.group(1).strip(),
            "evidence":      evidence_m.group(1).strip() if evidence_m else "",
            "mislead_method": mislead_m.group(1).strip() if mislead_m else "",
        })

    # ── 🕵️ 교묘한 오류 분석 ───────────────────────────────
    result["subtle_analysis"] = subtle_section

    # ── ✅ 권고사항 파싱 ────────────────────────────────────
    # 패턴: *   **제목**: 내용
    for m in re.finditer(r"\*\*(.+?)\*\*\s*:\s*(.+?)(?=\n\*|\Z)", recs_section, re.S):
        result["recommendations"].append({
            "title":  m.group(1).strip(),
            "detail": m.group(2).strip(),
        })

    return result


# ─────────────────────────────────────────────────────────
# 에이전트 래퍼
# ─────────────────────────────────────────────────────────

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
        3. explanation 파싱 → found_errors / recommendations / subtle_analysis
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

        # ── explanation 파싱 ────────────────────────────────
        parsed_report = _parse_explanation(explanation)

        print("\n=========차트 분석 완료=============\n")
        # 디버그 로그 (파싱 결과 포함)
        print({
            "chart_description": image_desc,
            "chart_type":        chart_type,
            "visual_errors":     misleader_names,
            "is_misleading":     is_misleading,
            "verdict":           verdict,
            "confidence":        confidence,
            "final_misleaders":  final_misleaders,
            "found_errors":      parsed_report["found_errors"],       # 로그용
            "recommendations":   parsed_report["recommendations"],    # 로그용
        })

        # ChartCheckState에 정의된 필드만 반환 (extra 필드는 Pydantic 오류 방지)
        return {
            "chart_description": image_desc,
            "chart_type":        chart_type,
            "visual_errors":     misleader_names,
            "data_errors":       [],
            "is_misleading":     is_misleading,
            "verdict":           verdict,
            "explanation":       explanation,
            "confidence":        confidence,
            "final_misleaders":  final_misleaders,
            "self_confidence":   self_conf,
        }
