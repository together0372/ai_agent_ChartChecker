"""
LangChain @tool 데코레이터 기반 도구 정의
기본 검증 12개 + 교묘한 오류 탐지 7개 = 총 19개
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from .config import MISLEADER_KEYS, MISLEADER_TAXONOMY
from .math_tools import MathTools
from .rag_utils import misviz_prior_from_neighbors, query_vector_db
from .web_utils import search_misleader_evidence as _search_evidence


# ─────────────────────────────────────────────────────────
# 기본 검증 도구 (12개)
# ─────────────────────────────────────────────────────────

@tool
def tool_search_misleader_evidence(misleader_key: str, chart_description: str) -> str:
    """특정 misleader 유형의 실제 사례와 증거를 인터넷에서 검색합니다.

    Args:
        misleader_key: MISLEADER_TAXONOMY의 키 (예: 'truncated_axis')
        chart_description: 분석 중인 차트의 설명 텍스트
    """
    return _search_evidence(misleader_key, chart_description)


@tool
def tool_check_axis_truncation(y_min: float, y_max: float, data_min: float) -> str:
    """Y축이 0 이외 지점에서 시작해 차이를 과장하는지 수학적으로 검증합니다.

    Args:
        y_min: Y축 최솟값 (차트에 표시된 시작값)
        y_max: Y축 최댓값
        data_min: 데이터의 실제 최솟값 (대부분 y_min과 같음)
    """
    return json.dumps(MathTools.check_axis_truncation(y_min, y_max, data_min), ensure_ascii=False)


@tool
def tool_check_pie_sum(percentages: list = None, pie_pcts: list = None, values: list = None) -> str:
    """파이/도넛 차트의 퍼센트 합계가 100%인지, 음수 포함 여부를 검증합니다.

    Args:
        percentages: 파이 조각별 퍼센트 값 리스트 (예: [45.2, 32.1, 22.7])
        pie_pcts: percentages의 alias
        values: percentages의 alias
    """
    pcts = percentages or pie_pcts or values
    if not pcts:
        return json.dumps({"error": "percentages 리스트가 필요합니다."}, ensure_ascii=False)
    return json.dumps(MathTools.check_pie(pcts), ensure_ascii=False)


@tool
def tool_check_tick_intervals(ticks: list) -> str:
    """눈금 간격이 균등한지 검증합니다 (불규칙 눈금 왜곡 탐지).

    Args:
        ticks: 눈금 값 리스트 (예: [2000, 2005, 2006, 2007, 2010])
    """
    return json.dumps(MathTools.check_tick_intervals(ticks), ensure_ascii=False)


@tool
def tool_check_item_order(labels: list) -> str:
    """레이블(특히 연도)의 순서가 올바른지 검증합니다.

    Args:
        labels: 차트 레이블 리스트 (예: ['2010년', '2008년', '2012년'])
    """
    return json.dumps(MathTools.check_item_order(labels), ensure_ascii=False)


@tool
def tool_check_pie_angles(pie_pcts: list, pie_angles_deg: list) -> str:
    """파이/도넛 차트 조각의 시각적 각도가 퍼센트와 일치하는지 검증합니다.

    Args:
        pie_pcts: 레이블에 표시된 퍼센트 리스트 (예: [82.9, 12.6, 4.5])
        pie_angles_deg: 실제 시각적 조각 각도(도) 리스트 (예: [298, 45, 17])
    """
    return json.dumps(MathTools.check_pie_angles(pie_pcts, pie_angles_deg), ensure_ascii=False)


@tool
def tool_get_misleader_info(misleader_key: str) -> str:
    """특정 misleader의 상세 정보(설명, 차트 유형, 심각도, 은폐 방법)를 가져옵니다.

    Args:
        misleader_key: MISLEADER_TAXONOMY의 키 (예: 'dual_axis')
    """
    if misleader_key not in MISLEADER_TAXONOMY:
        return f"키 없음: {misleader_key}"
    info = MISLEADER_TAXONOMY[misleader_key]
    return json.dumps({
        "name":            info["name"],
        "desc":            info["desc"],
        "chart_types":     info.get("chart_types", []),
        "severity_weight": info.get("severity_weight", 0.7),
        "hide_tip":        info.get("hide_tip", ""),
    }, ensure_ascii=False)


@tool
def tool_query_misviz_db(query: str = "", chart_description: str = "", description: str = "") -> str:
    """MisViz 벡터DB에서 유사한 차트·오류 사례를 검색합니다.

    Args:
        query: 차트에서 관찰한 내용 (예: '막대그래프 Y축이 60에서 시작해 차이 과장')
        chart_description: query의 alias
        description: query의 alias
    """
    q = query or chart_description or description
    if not q:
        return "query를 입력하세요."
    results = query_vector_db(q, n_results=3)
    if not results:
        return "유사 사례 없음"
    out = []
    for r in results:
        names = ", ".join(
            f"{MISLEADER_TAXONOMY.get(m, {}).get('name', m)}[{m}]" for m in r["misleaders"]
        ) or "정상"
        out.append(f"• [{r['chart_type']}] → 오류: {names}\n  {r['description'][:150]}")
    prior = misviz_prior_from_neighbors(q, k=8)
    if prior["ranked"]:
        freq = ", ".join(
            f"{MISLEADER_TAXONOMY.get(k, {}).get('name', k)}({c}건)"
            for k, c in prior["ranked"][:5]
        )
        out.append(f"\n[유사 {prior['n']}건 빈도] {freq}")
    return "\n".join(out)


@tool
def tool_check_dual_axis(
    left_axis_range: list = None,
    right_axis_range: list = None,
    description: str = "",
    left_range: list = None,
    right_range: list = None,
    left_axis: list = None,
    right_axis: list = None,
) -> str:
    """이중 Y축 스케일 조작으로 허위 상관관계를 만드는지 분석합니다.

    Args:
        left_axis_range: 왼쪽 Y축 범위 [최솟값, 최댓값] (left_range, left_axis 별칭 허용)
        right_axis_range: 오른쪽 Y축 범위 [최솟값, 최댓값] (right_range, right_axis 별칭 허용)
        description: 두 데이터 계열 설명
    """
    left  = left_axis_range or left_range or left_axis or []
    right = right_axis_range or right_range or right_axis or []
    if len(left) != 2 or len(right) != 2:
        return "오류: [최솟값, 최댓값] 형식 필요"
    ls = left[1] - left[0]
    rs = right[1] - right[0]
    if ls <= 0 or rs <= 0:
        return "오류: 범위 값 오류"
    scale_ratio = round(rs / ls, 2)
    manipulation = scale_ratio > 100 or scale_ratio < 0.01
    return json.dumps({
        "scale_ratio":         scale_ratio,
        "manipulation_likely": manipulation,
        "description":         description,
        "misleaders":          ["dual_axis", "false_correlation"] if manipulation else ["dual_axis"],
        "verdict": f"스케일 비율 {scale_ratio}:1 → {'조작 가능성 높음' if manipulation else '추가 검토 필요'}",
    }, ensure_ascii=False)


@tool
def tool_check_log_scale(ticks: list) -> str:
    """Y축 눈금이 로그 스케일인지 감지합니다 (레이블 없으면 log_scale_unlabeled).

    Args:
        ticks: Y축 눈금 값 리스트 (예: [1, 10, 100, 1000])
    """
    return json.dumps(MathTools.check_log_scale(ticks), ensure_ascii=False)


@tool
def tool_check_area_distortion(values: list, visual_areas: list) -> str:
    """버블/원 차트에서 면적이 아닌 반지름에 비례시켜 왜곡하는지 검증합니다.

    Args:
        values: 실제 데이터 값 리스트
        visual_areas: 원의 시각적 면적(픽셀² 또는 상대 크기) 리스트
    """
    return json.dumps(MathTools.check_area_distortion(values, visual_areas), ensure_ascii=False)


@tool
def tool_check_bar_scale_symmetry(
    left_values: list, left_px: list,
    right_values: list, right_px: list,
) -> str:
    """좌우 대칭 막대 차트에서 양쪽 스케일이 다른지(비대칭 왜곡) 검증합니다.

    Args:
        left_values/left_px:   왼쪽 막대 수치, 픽셀 길이
        right_values/right_px: 오른쪽 막대 수치, 픽셀 길이
    """
    return json.dumps(MathTools.check_bar_scale_symmetry(left_values, left_px, right_values, right_px), ensure_ascii=False)


# ─────────────────────────────────────────────────────────
# 교묘한 오류 탐지 도구 (7개 신규)
# ─────────────────────────────────────────────────────────

@tool
def tool_check_label_value_match(
    label_values: list,
    visual_ratios: list,
    y_min: float = 0.0,
    y_max: float = 100.0,
) -> str:
    """
    막대/점에 표기된 숫자 레이블과 시각적 높이 비율이 일치하는지 검증합니다.

    Args:
        label_values: 차트에 표기된 숫자 리스트 (예: [75, 82, 60])
        visual_ratios: Vision LLM이 추출한 막대 높이 비율 0~1 리스트 (예: [0.75, 0.80, 0.65])
        y_min: Y축 최솟값 (기본 0)
        y_max: Y축 최댓값 (기본 100)
    """
    return json.dumps(
        MathTools.check_label_value_match(label_values, visual_ratios, y_min, y_max),
        ensure_ascii=False,
    )


@tool
def tool_check_selective_annotation(
    annotated_indices: list,
    all_values: list,
) -> str:
    """
    주석(레이블/강조)이 특정 데이터에만 선택적으로 붙어 있는지 확인합니다.

    Args:
        annotated_indices: 주석이 있는 데이터 인덱스 리스트 (예: [0, 3, 7])
        all_values: 전체 데이터 값 리스트 (예: [45, 62, 38, 91, 55, 70, 48, 88])
    """
    return json.dumps(
        MathTools.check_selective_annotation(annotated_indices, all_values),
        ensure_ascii=False,
    )


@tool
def tool_check_aspect_ratio(
    width_px: int = 0,
    height_px: int = 0,
    x_data_range: float = 0.0,
    y_data_range: float = 0.0,
    # 아래는 LLM이 잘못된 이름으로 호출할 때 받아주는 별칭
    chart_width: Any = None,
    chart_height: Any = None,
    width: Any = None,
    height: Any = None,
    x_range: Any = None,
    y_range: Any = None,
    data_range_x: Any = None,
    data_range_y: Any = None,
    expected_ratio: Any = None,
) -> str:
    """
    차트의 가로·세로 비율이 Cleveland banking-to-45° 원칙을 위반하는지 확인합니다.

    Args:
        width_px: 차트 가로 픽셀 (chart_width, width 별칭 허용)
        height_px: 차트 세로 픽셀 (chart_height, height 별칭 허용)
        x_data_range: X축 데이터 범위 (x_range, data_range_x 별칭 허용)
        y_data_range: Y축 데이터 범위 (y_range, data_range_y 별칭 허용)
    """
    # 별칭 처리
    w = int(width_px or chart_width or width or 400)
    h = int(height_px or chart_height or height or 300)
    xr = float(x_data_range or x_range or data_range_x or 10)
    yr = float(y_data_range or y_range or data_range_y or 100)
    if w <= 0: w = 400
    if h <= 0: h = 300
    if xr <= 0: xr = 10
    if yr <= 0: yr = 100
    return json.dumps(
        MathTools.check_aspect_ratio(w, h, xr, yr),
        ensure_ascii=False,
    )


@tool
def tool_check_bin_widths(
    bin_edges: list,
    visual_widths_px: list,
) -> str:
    """
    히스토그램 구간(bin) 너비가 시각적으로 균등해 보이지만 실제 값 범위가 다른지 확인합니다.

    Args:
        bin_edges: 구간 경계값 리스트 (예: [0, 10, 20, 50, 100])
        visual_widths_px: 각 구간의 시각적 픽셀 너비 리스트 (예: [80, 80, 80, 80])
    """
    return json.dumps(
        MathTools.check_bin_widths(bin_edges, visual_widths_px),
        ensure_ascii=False,
    )


@tool
def tool_check_baseline_alignment(
    bar_start_values: list,
    expected_baseline: float = 0.0,
) -> str:
    """
    막대 차트에서 모든 막대가 동일한 기준선(보통 0)에서 시작하는지 확인합니다.

    Args:
        bar_start_values: 각 막대의 시작 Y값 리스트 (보통 0이어야 함)
        expected_baseline: 기대 기준선 (기본 0.0)
    """
    return json.dumps(
        MathTools.check_baseline_alignment(bar_start_values, expected_baseline),
        ensure_ascii=False,
    )


@tool
def tool_check_data_gap(
    shown_x_values: list,
    expected_step: float = 1.0,
) -> str:
    """
    X축에 표시된 값들 사이에 예상보다 큰 공백(누락 데이터)이 있는지 확인합니다.

    Args:
        shown_x_values: X축에 표시된 값 리스트 (연도/카테고리 등)
        expected_step: 예상 간격 (예: 1년 단위면 1.0)
    """
    return json.dumps(
        MathTools.check_data_gap_detection(shown_x_values, expected_step),
        ensure_ascii=False,
    )


@tool
def tool_check_color_emphasis_bias(
    highlighted_indices: list,
    all_values: list,
) -> str:
    """
    강조 색상(빨간/굵음/원색)이 붙은 데이터가 통계적으로 편향된 방향인지 확인합니다.

    Args:
        highlighted_indices: 강조 색상이 있는 데이터 인덱스 리스트
        all_values: 전체 데이터 값 리스트
    """
    return json.dumps(
        MathTools.check_color_emphasis_bias(highlighted_indices, all_values),
        ensure_ascii=False,
    )


# ─────────────────────────────────────────────────────────
# 도구 목록
# ─────────────────────────────────────────────────────────

OBSERVER_TOOLS: list[Any] = [
    tool_search_misleader_evidence,
    tool_check_axis_truncation,
    tool_check_pie_sum,
    tool_check_tick_intervals,
    tool_check_item_order,
    tool_check_pie_angles,
    tool_get_misleader_info,
    tool_query_misviz_db,
    tool_check_dual_axis,
    tool_check_log_scale,
    tool_check_area_distortion,
    tool_check_bar_scale_symmetry,
]

SUBTLE_TOOLS: list[Any] = [
    tool_check_label_value_match,
    tool_check_selective_annotation,
    tool_check_aspect_ratio,
    tool_check_bin_widths,
    tool_check_baseline_alignment,
    tool_check_data_gap,
    tool_check_color_emphasis_bias,
]

ALL_TOOLS: list[Any] = OBSERVER_TOOLS + SUBTLE_TOOLS

OBSERVER_TOOLS_BY_NAME: dict[str, Any] = {t.name: t for t in ALL_TOOLS}
