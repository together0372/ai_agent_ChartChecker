"""
ChartClassifierAgent — 이미지가 차트/그래프인지 판별하는 에이전트

판별 전략 (3단계):
  1. PIL 기반 빠른 사전 검사 (색상 다양성, 종횡비 등)
  2. 비전 LLM 판별 (구조적 요소 + 분류)
  3. 두 결과를 종합하여 최종 결정
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from PIL import Image

from .core.llm_utils import call_vision_llm, image_to_base64
from .core.config import LLM_NAME, IMAGE_TOKENS


# ─── 차트로 분류 가능한 카테고리 ───────────────────────────
CHART_CATEGORIES = {
    "bar":       "막대 차트",
    "line":      "선 차트",
    "pie":       "파이/도넛 차트",
    "scatter":   "산점도",
    "bubble":    "버블 차트",
    "histogram": "히스토그램",
    "area":      "영역 차트",
    "heatmap":   "히트맵",
    "treemap":   "트리맵",
    "other":     "기타 데이터 시각화",
}

# 비차트로 확정하는 키워드 (LLM 응답 파싱용)
_NOT_CHART_KEYWORDS = [
    "not a chart", "not a graph", "not a visualization",
    "photograph", "photo", "natural image", "portrait",
    "landscape", "drawing", "illustration", "text only",
    "차트가 아닙니다", "그래프가 아닙니다", "사진입니다",
]


class ChartClassifierAgent:
    """
    이미지가 차트/그래프/데이터 시각화인지 판별.

    workflow.py가 이 클래스의 classify 메서드를 LangGraph 노드로 등록한다.
    결과에 따라 워크플로우가 분기한다:
      - is_chart=True  → distort 노드로 진행
      - is_chart=False → 즉시 END
    """

    async def classify(self, state) -> Dict[str, Any]:
        """LangGraph 노드 진입점."""
        image_path = state.chart_image_path
        print(f"\n{'='*50}")
        print(f"  🖼️  차트 판별 중: {image_path}")
        print(f"{'='*50}")

        # ── 1단계: PIL 빠른 사전 검사 ──────────────────────
        pil_result = self._pil_precheck(image_path)
        print(f"     [PIL 사전검사] {pil_result['summary']}")

        # PIL에서 명확히 비차트로 판단되면 LLM 호출 없이 조기 반환
        if pil_result["confidence"] == "definitely_not":
            print("     → PIL: 명확히 차트가 아님 (LLM 스킵)")
            return {
                "is_chart":    False,
                "chart_type":  "",
                "skip_reason": f"PIL 판별: {pil_result['reason']}",
                "error":       "",
            }

        # ── 2단계: 비전 LLM 판별 ───────────────────────────
        llm_result = await self._llm_classify(image_path)
        print(f"     [LLM 판별] is_chart={llm_result['is_chart']} | "
              f"type={llm_result.get('chart_type','?')} | "
              f"conf={llm_result.get('confidence',0):.0%}")

        # ── 3단계: 두 결과 종합 ────────────────────────────
        is_chart, reason, chart_type = self._combine(pil_result, llm_result)

        if is_chart:
            korean_type = CHART_CATEGORIES.get(chart_type, chart_type)
            print(f"     ✅ 차트 확인: {korean_type}")
            return {
                "is_chart":    True,
                "chart_type":  korean_type,
                "skip_reason": "",
                "error":       "",
            }
        else:
            print(f"     ❌ 차트가 아님: {reason}")
            return {
                "is_chart":    False,
                "chart_type":  "",
                "skip_reason": reason,
                "error":       "",
            }

    # ─────────────────────────────────────────────────────
    # PIL 기반 빠른 사전 검사
    # ─────────────────────────────────────────────────────

    def _pil_precheck(self, image_path: str) -> dict:
        """
        PIL로 이미지를 분석해 명확한 비차트만 걸러냄.

        전략 변경: 밝은배경+제한색상 → 차트로 판정하는 로직 제거.
        PIL은 오직 '명확한 비차트'만 걸러내고, 차트 여부 판정은 LLM이 담당.

        이유: 흰 배경에 단순 물체(사과 등)도 밝은배경+제한색상을 가질 수 있어
        오탐이 발생했음. 구조적 특징(축, 격자선)은 LLM이 더 정확하게 판단.
        """
        try:
            with Image.open(image_path) as img:
                rgb  = img.convert("RGB")
                thumb = rgb.resize((100, 100))
                pixels = list(thumb.getdata())

                # 고유 색상 수 (32단계 양자화)
                unique_colors = len(set(
                    (r // 32, g // 32, b // 32) for r, g, b in pixels
                ))

                avg_brightness = sum(r + g + b for r, g, b in pixels) / (len(pixels) * 3)
                bright_ratio   = sum(
                    1 for r, g, b in pixels if r > 200 and g > 200 and b > 200
                ) / len(pixels)

                # 명확한 비차트만 필터링 (조건 강화)
                if avg_brightness < 20:
                    return {
                        "confidence": "definitely_not",
                        "reason":  "이미지가 너무 어두움 (거의 검은색)",
                        "summary": f"어두운 이미지 (밝기:{avg_brightness:.0f})",
                        "stats":   {"unique_colors": unique_colors, "bright_ratio": bright_ratio},
                    }

                if unique_colors > 220 and bright_ratio < 0.05:
                    return {
                        "confidence": "definitely_not",
                        "reason":  "색상이 매우 다양하고 밝은 영역 없음 (자연 사진)",
                        "summary": f"색상:{unique_colors}종 / 밝은영역:{bright_ratio:.0%}",
                        "stats":   {"unique_colors": unique_colors, "bright_ratio": bright_ratio},
                    }

                # 나머지는 모두 LLM으로 판단 (PIL은 긍정 판정 안 함)
                return {
                    "confidence": "uncertain",
                    "reason":  "PIL만으로 판단 불가 → LLM으로 확인",
                    "summary": f"색상:{unique_colors}종 / 밝기:{avg_brightness:.0f} (LLM 확인 필요)",
                    "stats":   {"unique_colors": unique_colors, "bright_ratio": bright_ratio},
                }

        except Exception as e:
            return {
                "confidence": "uncertain",
                "reason":  f"PIL 분석 실패: {e}",
                "summary": "PIL 오류",
                "stats":   {},
            }

    # ─────────────────────────────────────────────────────
    # 비전 LLM 판별
    # ─────────────────────────────────────────────────────

    async def _llm_classify(self, image_path: str) -> dict:
        """비전 LLM으로 차트 여부 및 유형을 판별."""
        system = """You are an image classifier. Determine if the image is a data chart/graph.

## A chart MUST have BOTH:
1. DATA ENCODING: visual elements that represent numeric quantities
   - bars / lines / pie or donut slices / dots / areas / bubbles
2. QUANTITATIVE REFERENCE: numbers or labels that quantify the data
   - axis tick values, percentages, numeric annotations, or a legend with values

NOTE: Pie and donut charts do NOT need axes — circular slices + percentages/labels = chart ✅

## NOT a chart — reject if:
- Photo or illustration of real-world objects (food, people, animals, products) — even on white background
- Text-only document, table, or screenshot with no visual data encoding
- Flowchart or org chart (boxes/arrows, no numeric data)
- Decorative image with only icons and text but NO numeric data encoding
- Single number or statistic shown as plain text

## chart_type options:
bar, line, pie, donut, scatter, bubble, histogram, area, heatmap, treemap, bidirectional_bar, other

Respond ONLY with this JSON (no markdown):
{
  "is_chart": true,
  "chart_type": "bar|line|pie|donut|scatter|bubble|histogram|area|heatmap|treemap|bidirectional_bar|other",
  "confidence": 0.95,
  "visual_evidence": ["specific element 1", "specific element 2"],
  "reason": "brief explanation"
}"""

        try:
            raw = await asyncio.to_thread(
                call_vision_llm,
                "Is this image a chart, graph, or data visualization? Classify it.",
                image_path,
                system,
                300,
                0.0,
                False,   # use_thinking=False — JSON 출력 보장
            )

            # JSON 파싱 (코드블록 또는 순수 JSON 모두 처리)
            import json, re
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if m:
                json_str = m.group(1).strip()
            else:
                m2 = re.search(r"\{[\s\S]*\}", raw)
                json_str = m2.group(0).strip() if m2 else raw.strip()
            try:
                result = json.loads(json_str)
            except Exception:
                # 파싱 실패 → 텍스트에서 패턴 추출
                is_chart = not any(kw in raw.lower() for kw in _NOT_CHART_KEYWORDS)
                return {
                    "is_chart": is_chart,
                    "chart_type": "other" if is_chart else "",
                    "confidence": 0.5,
                    "reason": raw[:200],
                }

            return {
                "is_chart":       bool(result.get("is_chart", False)),
                "chart_type":     result.get("chart_type", "other"),
                "confidence":     float(result.get("confidence", 0.5)),
                "visual_evidence": result.get("visual_evidence", []),
                "reason":         result.get("reason", ""),
            }

        except Exception as e:
            # LLM 실패 시 불확실로 처리 (차트로 가정 — false negative 방지)
            print(f"     ⚠️  LLM 판별 실패: {e} → 차트로 가정")
            return {
                "is_chart":   True,
                "chart_type": "other",
                "confidence": 0.4,
                "reason":     f"LLM 실패, 차트로 가정: {e}",
            }

    # ─────────────────────────────────────────────────────
    # 두 결과 종합
    # ─────────────────────────────────────────────────────

    def _combine(
        self,
        pil: dict,
        llm: dict,
    ) -> tuple[bool, str, str]:
        """
        PIL 사전검사 + LLM 결과를 종합해 최종 판정.

        PIL은 '명확한 비차트'만 걸러내고, 차트 판정은 LLM이 담당.
        불확실할 때는 보수적으로 '비차트'로 처리 (오탐 방지).

        Returns:
            (is_chart, reason, chart_type)
        """
        llm_chart  = llm.get("is_chart", False)
        llm_conf   = llm.get("confidence", 0.5)
        pil_conf   = pil.get("confidence", "uncertain")
        chart_type = llm.get("chart_type", "other") or "other"

        # PIL이 명확히 비차트로 판단 → LLM 결과 무시하고 종료
        if pil_conf == "definitely_not":
            return False, f"PIL 판별: {pil.get('reason', '명확한 비차트')}", ""

        # LLM이 높은 확신으로 차트가 아닌 경우
        if not llm_chart and llm_conf >= 0.75:
            return False, f"LLM 판별: {llm.get('reason', '차트 아님')}", ""

        # LLM이 높은 확신으로 차트인 경우
        if llm_chart and llm_conf >= 0.72:
            return True, "", chart_type

        # LLM 불확실 (0.5~0.72) → 보수적으로 비차트 처리 (오탐 방지)
        # 사과 사진처럼 PIL이 uncertain + LLM이 낮은 확신이면 비차트
        if not llm_chart:
            return False, f"LLM 판별 (저확신 {llm_conf:.0%}): {llm.get('reason','차트 아님')}", ""

        # LLM이 차트라 하지만 확신 낮음 → 차트로 처리 (false negative 방지)
        return True, "", chart_type
