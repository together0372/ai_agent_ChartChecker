"""
LangChain @tool 데코레이터 기반 도구 정의
기본 검증 12개 + 교묘한 오류 탐지 7개 = 총 19개

Gemini bind_tools 호환 주의사항:
  - list = None → Optional[List[<type>]] = None 으로 명시해야 함
    (Gemini API는 array 타입에 items 필드가 반드시 필요)
  - Any = None  → Optional[Union[int, float]] = None 으로 명시
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

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
def tool_check_axis_truncation(
    y_min: float = 0.0,
    y_max: float = 100.0,
    data_min: float = 0.0,
    axis_min: float = 0.0,
    axis_max: float = 100.0,
    min_value: float = 0.0,
    chart_min: float = 0.0,
    chart_max: float = 100.0,
    start_value: float = 0.0,
) -> str:
    """Y축이 0 이외 지점에서 시작해 차이를 과장하는지 수학적으로 검증합니다.

    Args:
        y_min: Y축 최솟값 (axis_min, chart_min, start_value 별칭 허용)
        y_max: Y축 최댓값 (axis_max, chart_max 별칭 허용)
        data_min: 데이터의 실제 최솟값 (min_value 별칭 허용)
    """
    lo   = y_min   if y_min   != 0.0 else (axis_min  or chart_min  or start_value or 0)
    hi   = y_max   if y_max   != 100.0 else (axis_max  or chart_max  or 100)
    dmin = data_min if data_min != 0.0 else (min_value or lo)
    return json.dumps(MathTools.check_axis_truncation(float(lo), float(hi), float(dmin)), ensure_ascii=False)


@tool
def tool_check_pie_sum(
    percentages: Optional[List[float]] = None,
    pie_pcts: Optional[List[float]] = None,
    values: Optional[List[float]] = None,
) -> str:
    """파이/도넛 차트의 퍼센트 합계가 100%인지, 음수 포함 여부를 검증합니다.

    Args:
        percentages: 파이 조각별 퍼센트 값 리스트 (예: [45.2, 32.1, 22.7])
        pie_pcts: percentages의 alias
        values: percentages의 alias
    """
    pcts = percentages or pie_pcts or values
    if not pcts:
        return json.dumps({"checked": False, "note": "percentages 리스트가 필요합니다."}, ensure_ascii=False)
    return json.dumps(MathTools.check_pie(pcts), ensure_ascii=False)


@tool
def tool_check_multi_pie_sum(
    all_pcts: Optional[List[float]] = None,
    chart_count: int = 0,
    pie_pcts: Optional[List[float]] = None,
    pcts: Optional[List[float]] = None,
    percentages: Optional[List[float]] = None,
    chart_index: Optional[int] = None,
) -> str:
    """한 이미지에 파이/도넛 차트가 여러 개 있을 때 각 차트의 합계를 개별 검증합니다.
    단순 tool_check_pie_sum은 전체를 합산해 200%로 오탐할 수 있으므로 이 도구를 사용합니다.

    Args:
        all_pcts: 이미지에서 추출한 모든 퍼센트 값 (여러 차트 합산, 예: [45, 32, 23, 60, 25, 15])
        chart_count: 차트 개수 (0=자동감지, 2=두 개, 3=세 개)
        pie_pcts: all_pcts의 별칭
        pcts: all_pcts의 별칭
        percentages: all_pcts의 별칭
        chart_index: 무시됨 (호환성용)
    """
    data = all_pcts or pie_pcts or pcts or percentages or []
    return json.dumps(MathTools.check_multi_pie_sum(data, chart_count), ensure_ascii=False)


@tool
def tool_check_tick_intervals(
    ticks: Optional[List[float]] = None,
    y_ticks: Optional[List[float]] = None,
    tick_values: Optional[List[float]] = None,
    intervals: Optional[List[float]] = None,
    axis_ticks: Optional[List[float]] = None,
) -> str:
    """눈금 간격이 균등한지 검증합니다 (불규칙 눈금 왜곡 탐지).

    Args:
        ticks: 눈금 값 리스트 (y_ticks, tick_values, intervals, axis_ticks 별칭 허용)
    """
    data = ticks or y_ticks or tick_values or intervals or axis_ticks or []
    return json.dumps(MathTools.check_tick_intervals(data), ensure_ascii=False)


@tool
def tool_check_item_order(
    labels: Optional[List[str]] = None,
    x_labels: Optional[List[str]] = None,
) -> str:
    """레이블(특히 연도)의 순서가 올바른지 검증합니다.

    Args:
        labels: 차트 레이블 리스트 (예: ['2010년', '2008년', '2012년'])
        x_labels: labels의 별칭
    """
    items = labels or x_labels or []
    return json.dumps(MathTools.check_item_order(items), ensure_ascii=False)


@tool
def tool_check_pie_angles(
    pie_pcts: Optional[List[float]] = None,
    pie_angles_deg: Optional[List[float]] = None,
    angles: Optional[List[float]] = None,
    percentages: Optional[List[float]] = None,
    pcts: Optional[List[float]] = None,
    values: Optional[List[float]] = None,
    degrees: Optional[List[float]] = None,
) -> str:
    """파이/도넛 차트 조각의 시각적 각도가 퍼센트와 일치하는지 검증합니다.

    Args:
        pie_pcts: 레이블에 표시된 퍼센트 리스트 (percentages, pcts, values 별칭 허용)
        pie_angles_deg: 실제 시각적 조각 각도(도) 리스트 (angles, degrees 별칭 허용)
    """
    pct_data = pie_pcts or percentages or pcts or values or []
    if not pct_data:
        return json.dumps({"checked": False, "note": "퍼센트 데이터가 필요합니다."}, ensure_ascii=False)
    deg = pie_angles_deg or angles or degrees or []
    if not deg:
        total = sum(pct_data) or 100
        deg = [round(p / total * 360, 1) for p in pct_data]
    return json.dumps(MathTools.check_pie_angles(pct_data, deg), ensure_ascii=False)


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
    left_axis_range: Optional[List[float]] = None,
    right_axis_range: Optional[List[float]] = None,
    description: str = "",
    left_range: Optional[List[float]] = None,
    right_range: Optional[List[float]] = None,
    left_axis: Optional[List[float]] = None,
    right_axis: Optional[List[float]] = None,
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
        return json.dumps({"error": "[최솟값, 최댓값] 형식 필요"}, ensure_ascii=False)
    result = MathTools.check_dual_axis(float(left[0]), float(left[1]),
                                       float(right[0]), float(right[1]))
    result["description"] = description
    return json.dumps(result, ensure_ascii=False)


@tool
def tool_check_log_scale(
    ticks: Optional[List[float]] = None,
    y_ticks: Optional[List[float]] = None,
    tick_values: Optional[List[float]] = None,
    axis_values: Optional[List[float]] = None,
) -> str:
    """Y축 눈금이 로그 스케일인지 감지합니다 (레이블 없으면 log_scale_unlabeled).

    Args:
        ticks: Y축 눈금 값 리스트 (y_ticks, tick_values, axis_values 별칭 허용)
    """
    data = ticks or y_ticks or tick_values or axis_values or []
    return json.dumps(MathTools.check_log_scale(data), ensure_ascii=False)


@tool
def tool_check_area_distortion(
    values: Optional[List[float]] = None,
    visual_areas: Optional[List[float]] = None,
    data_values: Optional[List[float]] = None,
    areas: Optional[List[float]] = None,
    bubble_sizes: Optional[List[float]] = None,
    actual_values: Optional[List[float]] = None,
    circle_areas: Optional[List[float]] = None,
) -> str:
    """버블/원 차트에서 면적이 아닌 반지름에 비례시켜 왜곡하는지 검증합니다.

    Args:
        values: 실제 데이터 값 리스트 (data_values, actual_values 별칭 허용)
        visual_areas: 원의 시각적 면적(픽셀² 또는 상대 크기) 리스트 (areas, bubble_sizes, circle_areas 별칭 허용)
    """
    vals = values or data_values or actual_values or []
    arrs = visual_areas or areas or bubble_sizes or circle_areas or []
    if not vals:
        return json.dumps({"checked": False, "note": "values 데이터가 필요합니다."}, ensure_ascii=False)
    return json.dumps(MathTools.check_area_distortion(vals, arrs), ensure_ascii=False)


@tool
def tool_check_bar_scale_symmetry(
    left_values: Optional[List[float]] = None,
    left_px: Optional[List[float]] = None,
    right_values: Optional[List[float]] = None,
    right_px: Optional[List[float]] = None,
    left_bar_values: Optional[List[float]] = None,
    right_bar_values: Optional[List[float]] = None,
    left_bar_px: Optional[List[float]] = None,
    right_bar_px: Optional[List[float]] = None,
    left_lengths: Optional[List[float]] = None,
    right_lengths: Optional[List[float]] = None,
) -> str:
    """좌우 대칭 막대 차트에서 양쪽 스케일이 다른지(비대칭 왜곡) 검증합니다.

    Args:
        left_values: 왼쪽 막대 수치 (left_bar_values 별칭 허용)
        left_px: 왼쪽 막대 픽셀 길이 (left_bar_px, left_lengths 별칭 허용)
        right_values: 오른쪽 막대 수치 (right_bar_values 별칭 허용)
        right_px: 오른쪽 막대 픽셀 길이 (right_bar_px, right_lengths 별칭 허용)
    """
    lv = left_values  or left_bar_values  or []
    lp = left_px      or left_bar_px      or left_lengths  or []
    rv = right_values or right_bar_values or []
    rp = right_px     or right_bar_px     or right_lengths or []
    if not lv or not rv:
        return json.dumps({"checked": False, "note": "left_values, right_values 데이터가 필요합니다."}, ensure_ascii=False)
    return json.dumps(MathTools.check_bar_scale_symmetry(lv, lp, rv, rp), ensure_ascii=False)


# ─────────────────────────────────────────────────────────
# 교묘한 오류 탐지 도구 (7개 신규)
# ─────────────────────────────────────────────────────────

@tool
def tool_check_label_value_match(
    label_values: Optional[List[float]] = None,
    visual_ratios: Optional[List[float]] = None,
    y_min: float = 0.0,
    y_max: float = 100.0,
    values: Optional[List[float]] = None,
    ratios: Optional[List[float]] = None,
    bar_heights: Optional[List[float]] = None,
    data_values: Optional[List[float]] = None,
    bar_ratios: Optional[List[float]] = None,
    axis_min: float = 0.0,
    axis_max: float = 0.0,
) -> str:
    """
    막대/점에 표기된 숫자 레이블과 시각적 높이 비율이 일치하는지 검증합니다.

    Args:
        label_values: 차트에 표기된 숫자 리스트 (values, data_values 별칭 허용)
        visual_ratios: 막대 높이 비율 0~1 리스트 (ratios, bar_heights, bar_ratios 별칭 허용)
        y_min: Y축 최솟값 (axis_min 별칭 허용)
        y_max: Y축 최댓값 (axis_max 별칭 허용)
    """
    lv   = label_values  or values  or data_values  or []
    vr   = visual_ratios or ratios  or bar_heights  or bar_ratios or []
    ylo  = axis_min if axis_min != 0.0 else y_min
    yhi  = axis_max if axis_max != 0.0 else y_max
    if not lv:
        return json.dumps({"checked": False, "note": "label_values 데이터가 필요합니다."}, ensure_ascii=False)
    return json.dumps(
        MathTools.check_label_value_match(lv, vr, ylo, yhi),
        ensure_ascii=False,
    )


@tool
def tool_check_selective_annotation(
    annotated_indices: Optional[List[int]] = None,
    all_values: Optional[List[float]] = None,
    annotated_intervals: Optional[List[float]] = None,
    values: Optional[List[float]] = None,
) -> str:
    """
    주석(레이블/강조)이 특정 데이터에만 선택적으로 붙어 있는지 확인합니다.

    Args:
        annotated_indices: 주석이 있는 데이터 인덱스 리스트 (예: [0, 3, 7])
        all_values: 전체 데이터 값 리스트 (예: [45, 62, 38, 91, 55, 70, 48, 88])
        annotated_intervals: annotated_indices의 별칭
        values: all_values의 별칭
    """
    idx = annotated_indices
    if idx is None and annotated_intervals is not None:
        idx = list(range(len(annotated_intervals)))
    idx = idx or []

    vals = all_values or values
    if vals is None and annotated_intervals is not None:
        extracted = []
        for item in annotated_intervals:
            if isinstance(item, (int, float)):
                extracted.append(float(item))
        vals = extracted if extracted else []
    vals = vals or []

    return json.dumps(
        MathTools.check_selective_annotation(idx, vals),
        ensure_ascii=False,
    )


@tool
def tool_check_aspect_ratio(
    width_px: int = 0,
    height_px: int = 0,
    x_data_range: float = 0.0,
    y_data_range: float = 0.0,
    chart_width: Optional[Union[int, float]] = None,
    chart_height: Optional[Union[int, float]] = None,
    width: Optional[Union[int, float]] = None,
    height: Optional[Union[int, float]] = None,
    x_range: Optional[Union[int, float]] = None,
    y_range: Optional[Union[int, float]] = None,
    data_range_x: Optional[Union[int, float]] = None,
    data_range_y: Optional[Union[int, float]] = None,
    expected_ratio: Optional[Union[int, float]] = None,
) -> str:
    """
    차트의 가로·세로 비율이 Cleveland banking-to-45° 원칙을 위반하는지 확인합니다.

    Args:
        width_px: 차트 가로 픽셀 (chart_width, width 별칭 허용)
        height_px: 차트 세로 픽셀 (chart_height, height 별칭 허용)
        x_data_range: X축 데이터 범위 (x_range, data_range_x 별칭 허용)
        y_data_range: Y축 데이터 범위 (y_range, data_range_y 별칭 허용)
    """
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
    bin_edges: Optional[List[float]] = None,
    visual_widths_px: Optional[List[float]] = None,
    edges: Optional[List[float]] = None,
    widths: Optional[List[float]] = None,
    bin_boundaries: Optional[List[float]] = None,
    bar_widths_px: Optional[List[float]] = None,
) -> str:
    """
    히스토그램 구간(bin) 너비가 시각적으로 균등해 보이지만 실제 값 범위가 다른지 확인합니다.

    Args:
        bin_edges: 구간 경계값 리스트 (edges, bin_boundaries 별칭 허용)
        visual_widths_px: 각 구간의 시각적 픽셀 너비 리스트 (widths, bar_widths_px 별칭 허용)
    """
    be = bin_edges or edges or bin_boundaries or []
    wp = visual_widths_px or widths or bar_widths_px or []
    if not be:
        return json.dumps({"checked": False, "note": "bin_edges 데이터가 필요합니다."}, ensure_ascii=False)
    return json.dumps(
        MathTools.check_bin_widths(be, wp),
        ensure_ascii=False,
    )


@tool
def tool_check_baseline_alignment(
    bar_start_values: Optional[List[float]] = None,
    expected_baseline: float = 0.0,
    start_values: Optional[List[float]] = None,
    baseline_values: Optional[List[float]] = None,
    bar_starts: Optional[List[float]] = None,
    starts: Optional[List[float]] = None,
) -> str:
    """
    막대 차트에서 모든 막대가 동일한 기준선(보통 0)에서 시작하는지 확인합니다.

    Args:
        bar_start_values: 각 막대의 시작 Y값 리스트 (start_values, baseline_values, bar_starts, starts 별칭 허용)
        expected_baseline: 기대 기준선 (기본 0.0)
    """
    data = bar_start_values or start_values or baseline_values or bar_starts or starts or []
    if not data:
        return json.dumps({"checked": False, "note": "bar_start_values 데이터가 필요합니다."}, ensure_ascii=False)
    return json.dumps(
        MathTools.check_baseline_alignment(data, expected_baseline),
        ensure_ascii=False,
    )


@tool
def tool_check_data_gap(
    shown_x_values: Optional[List[float]] = None,
    expected_step: float = 1.0,
    x_values: Optional[List[float]] = None,
    x_labels: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    time_points: Optional[List[float]] = None,
    x_ticks: Optional[List[float]] = None,
) -> str:
    """
    X축에 표시된 값들 사이에 예상보다 큰 공백(누락 데이터)이 있는지 확인합니다.

    Args:
        shown_x_values: X축에 표시된 값 리스트 (x_values, x_labels, labels, time_points, x_ticks 별칭 허용)
        expected_step: 예상 간격 (예: 1년 단위면 1.0)
    """
    data = shown_x_values or x_values or x_labels or labels or time_points or x_ticks or []
    if not data:
        return json.dumps({"checked": False, "note": "x값 데이터가 필요합니다."}, ensure_ascii=False)
    return json.dumps(
        MathTools.check_data_gap_detection(data, expected_step),
        ensure_ascii=False,
    )


@tool
def tool_check_color_emphasis_bias(
    highlighted_indices: Optional[List[int]] = None,
    all_values: Optional[List[float]] = None,
    indices: Optional[List[int]] = None,
    values: Optional[List[float]] = None,
    emphasized_indices: Optional[List[int]] = None,
    data_values: Optional[List[float]] = None,
) -> str:
    """
    강조 색상(빨간/굵음/원색)이 붙은 데이터가 통계적으로 편향된 방향인지 확인합니다.

    Args:
        highlighted_indices: 강조 색상이 있는 데이터 인덱스 리스트
        all_values: 전체 데이터 값 리스트
        indices: highlighted_indices 별칭
        values: all_values 별칭
        emphasized_indices: highlighted_indices 별칭
        data_values: all_values 별칭
    """
    idx  = highlighted_indices or indices or emphasized_indices or []
    vals = all_values or values or data_values or []
    return json.dumps(
        MathTools.check_color_emphasis_bias(idx, vals),
        ensure_ascii=False,
    )


@tool
def tool_check_slope_distortion(
    values: Optional[List[float]] = None,
    visual_y_px: Optional[List[float]] = None,
    x_labels: Optional[List[str]] = None,
    data_values: Optional[List[float]] = None,
    y_positions: Optional[List[float]] = None,
    y_coords: Optional[List[float]] = None,
    y_pixels: Optional[List[float]] = None,
    actual_values: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
) -> str:
    """선 차트에서 실제 데이터 변화량과 시각적 기울기의 불일치를 탐지합니다.

    각 구간의 '픽셀당 값' 비율이 일정해야 정상입니다.
    작은 변화가 큰 변화보다 시각적으로 더 가파르게 그려진 경우를 감지합니다.

    Args:
        values: 시간 순서대로 나열한 실제 데이터 값 리스트 (data_values, actual_values 별칭 허용)
        visual_y_px: 각 데이터 포인트의 화면 Y 픽셀 좌표 위=0 (y_positions, y_coords, y_pixels 별칭 허용)
        x_labels: X축 레이블 리스트 (labels 별칭 허용)
    """
    vals = values or data_values or actual_values or []
    ypx  = visual_y_px or y_positions or y_coords or y_pixels or []
    xlbl = x_labels or labels or []
    if not vals:
        return json.dumps({"checked": False, "note": "values 데이터가 필요합니다."}, ensure_ascii=False)
    return json.dumps(
        MathTools.check_slope_distortion(vals, ypx, xlbl),
        ensure_ascii=False,
    )


@tool
def tool_check_misleading_annotation(
    annotated_intervals: Optional[List[str]] = None,
    annotations: Optional[List[str]] = None,
    intervals: Optional[List[str]] = None,
    annotation_data: Optional[List[str]] = None,
) -> str:
    """차트에 추가된 주석(화살표·괄호·강조선)이 실제 변화량을 왜곡 표현하는지 탐지합니다.

    탐지 패턴:
      1. 주석에 표시된 숫자와 실제 변화량 불일치 (수치 조작)
      2. 더 작은 변화에 더 강한 시각적 강조를 달아 오해 유발

    Args:
        annotated_intervals: 주석 구간 정보 JSON 문자열 리스트.
          각 항목(JSON 문자열): {"label":"7월→8월","actual_delta":-4.1,"displayed_text":"4.1 감소","visual_emphasis":1.0}
          visual_emphasis: 1.0=보통, 2.0=2배 강조
        annotations: annotated_intervals 별칭
        intervals: annotated_intervals 별칭
        annotation_data: annotated_intervals 별칭
    """
    raw = annotated_intervals or annotations or intervals or annotation_data or []
    if not raw:
        return json.dumps({"misleading": False, "note": "주석 데이터 없음"}, ensure_ascii=False)
    # JSON 문자열로 받은 경우 파싱
    data = []
    for item in raw:
        if isinstance(item, str):
            try:
                data.append(json.loads(item))
            except Exception:
                pass
        elif isinstance(item, dict):
            data.append(item)
    if not data:
        return json.dumps({"misleading": False, "note": "파싱 가능한 주석 없음"}, ensure_ascii=False)
    return json.dumps(
        MathTools.check_misleading_annotation(data),
        ensure_ascii=False,
    )


# ─────────────────────────────────────────────────────────
# 도구 목록
# ─────────────────────────────────────────────────────────

OBSERVER_TOOLS: list[Any] = [
    tool_search_misleader_evidence,
    tool_check_axis_truncation,
    # tool_check_pie_sum 제거 — tool_check_multi_pie_sum이 단일/다중 모두 처리
    tool_check_multi_pie_sum,
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
    tool_check_slope_distortion,
    tool_check_misleading_annotation,
]

ALL_TOOLS: list[Any] = OBSERVER_TOOLS + SUBTLE_TOOLS

OBSERVER_TOOLS_BY_NAME: dict[str, Any] = {t.name: t for t in ALL_TOOLS}
