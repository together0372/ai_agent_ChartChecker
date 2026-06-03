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
def tool_check_axis_truncation(
    y_min: float = None,
    y_max: float = None,
    data_min: float = None,
    axis_min: float = None,
    axis_max: float = None,
    min_value: float = None,
    chart_min: float = None,
    chart_max: float = None,
    start_value: float = None,
) -> str:
    """Y축이 0 이외 지점에서 시작해 차이를 과장하는지 수학적으로 검증합니다.

    Args:
        y_min: Y축 최솟값 (axis_min, chart_min, start_value 별칭 허용)
        y_max: Y축 최댓값 (axis_max, chart_max 별칭 허용)
        data_min: 데이터의 실제 최솟값 (min_value 별칭 허용)
    """
    lo   = y_min   if y_min   is not None else (axis_min  or chart_min  or start_value or 0)
    hi   = y_max   if y_max   is not None else (axis_max  or chart_max  or 100)
    dmin = data_min if data_min is not None else (min_value or lo)
    return json.dumps(MathTools.check_axis_truncation(float(lo), float(hi), float(dmin)), ensure_ascii=False)


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
        return json.dumps({"checked": False, "note": "percentages 리스트가 필요합니다."}, ensure_ascii=False)
    return json.dumps(MathTools.check_pie(pcts), ensure_ascii=False)


@tool
def tool_check_multi_pie_sum(
    all_pcts: list = None,
    chart_count: int = 0,
    pie_pcts: list = None,
    pcts: list = None,
    percentages: list = None,
    chart_index: int = None,
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
    ticks: list = None,
    y_ticks: list = None,
    tick_values: list = None,
    intervals: list = None,
    axis_ticks: list = None,
) -> str:
    """눈금 간격이 균등한지 검증합니다 (불규칙 눈금 왜곡 탐지).

    Args:
        ticks: 눈금 값 리스트 (y_ticks, tick_values, intervals, axis_ticks 별칭 허용)
    """
    data = ticks or y_ticks or tick_values or intervals or axis_ticks or []
    return json.dumps(MathTools.check_tick_intervals(data), ensure_ascii=False)


@tool
def tool_check_item_order(labels: list = None, x_labels: list = None) -> str:
    """레이블(특히 연도)의 순서가 올바른지 검증합니다.

    Args:
        labels: 차트 레이블 리스트 (예: ['2010년', '2008년', '2012년'])
        x_labels: labels의 별칭
    """
    items = labels or x_labels or []
    return json.dumps(MathTools.check_item_order(items), ensure_ascii=False)


@tool
def tool_check_pie_angles(
    pie_pcts: list = None,
    pie_angles_deg: list = None,
    angles: list = None,
    percentages: list = None,
    pcts: list = None,
    values: list = None,
    degrees: list = None,
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
        # 각도 없으면 퍼센트로 이론적 각도 계산 후 검증
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
        return json.dumps({"error": "[최솟값, 최댓값] 형식 필요"}, ensure_ascii=False)
    # math_tools와 동일한 기준으로 통일 (> 10배 또는 < 0.1배)
    result = MathTools.check_dual_axis(float(left[0]), float(left[1]),
                                       float(right[0]), float(right[1]))
    result["description"] = description
    return json.dumps(result, ensure_ascii=False)


@tool
def tool_check_log_scale(
    ticks: list = None,
    y_ticks: list = None,
    tick_values: list = None,
    axis_values: list = None,
) -> str:
    """Y축 눈금이 로그 스케일인지 감지합니다 (레이블 없으면 log_scale_unlabeled).

    Args:
        ticks: Y축 눈금 값 리스트 (y_ticks, tick_values, axis_values 별칭 허용)
    """
    data = ticks or y_ticks or tick_values or axis_values or []
    return json.dumps(MathTools.check_log_scale(data), ensure_ascii=False)


@tool
def tool_check_area_distortion(
    values: list = None,
    visual_areas: list = None,
    visual_ratios: list = None,
    data_values: list = None,
    areas: list = None,
    bubble_sizes: list = None,
    actual_values: list = None,
    circle_areas: list = None,
    size_ratios: list = None,
) -> str:
    """버블/산점도 차트에서 원 크기를 면적이 아닌 반지름에 비례시켜 왜곡하는지 검증합니다.

    Args:
        values: 실제 데이터 값 리스트 (data_values, actual_values 별칭 허용)
        visual_areas: 원들의 시각적 상대 크기 비율 리스트.
                      절대 픽셀값 불필요 — 상대 비율로 전달하면 됩니다.
                      예: [2.0, 1.0, 3.0] = 첫 번째 원이 두 번째의 2배, 세 번째의 2/3 크기
                      (visual_ratios, size_ratios, areas, bubble_sizes, circle_areas 별칭 허용)
    """
    vals = values or data_values or actual_values or []
    arrs = visual_areas or visual_ratios or size_ratios or areas or bubble_sizes or circle_areas or []
    if not vals:
        return json.dumps({"checked": False, "note": "values 데이터가 필요합니다."}, ensure_ascii=False)
    return json.dumps(MathTools.check_area_distortion(vals, arrs), ensure_ascii=False)


@tool
def tool_check_bar_scale_symmetry(
    left_values: list = None,
    left_px: list = None,
    right_values: list = None,
    right_px: list = None,
    left_bar_values: list = None,
    right_bar_values: list = None,
    left_bar_px: list = None,
    right_bar_px: list = None,
    left_lengths: list = None,
    right_lengths: list = None,
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
    label_values: list = None,
    visual_ratios: list = None,
    y_min: float = 0.0,
    y_max: float = 100.0,
    values: list = None,
    ratios: list = None,
    bar_heights: list = None,
    data_values: list = None,
    bar_ratios: list = None,
    axis_min: float = None,
    axis_max: float = None,
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
    ylo  = axis_min if axis_min is not None else y_min
    yhi  = axis_max if axis_max is not None else y_max
    if not lv:
        return json.dumps({"checked": False, "note": "label_values 데이터가 필요합니다."}, ensure_ascii=False)
    return json.dumps(
        MathTools.check_label_value_match(lv, vr, ylo, yhi),
        ensure_ascii=False,
    )


@tool
def tool_check_selective_annotation(
    annotated_indices: list = None,
    all_values: list = None,
    annotated_intervals: list = None,
    values: list = None,
) -> str:
    """
    주석(레이블/강조)이 특정 데이터에만 선택적으로 붙어 있는지 확인합니다.

    Args:
        annotated_indices: 주석이 있는 데이터 인덱스 리스트 (예: [0, 3, 7])
        all_values: 전체 데이터 값 리스트 (예: [45, 62, 38, 91, 55, 70, 48, 88])
        annotated_intervals: annotated_indices의 별칭
        values: all_values의 별칭
    """
    # annotated_intervals가 오면 인덱스 리스트로 변환
    idx = annotated_indices
    if idx is None and annotated_intervals is not None:
        idx = list(range(len(annotated_intervals)))
    idx = idx or []

    vals = all_values or values
    # annotated_intervals에서 값 추출 시도
    if vals is None and annotated_intervals is not None:
        extracted = []
        for item in annotated_intervals:
            if isinstance(item, dict):
                v = item.get("value") or item.get("pct") or item.get("percent") or 0
                extracted.append(float(v))
            elif isinstance(item, (int, float)):
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
    bin_edges: list = None,
    visual_widths_px: list = None,
    edges: list = None,
    widths: list = None,
    bin_boundaries: list = None,
    bar_widths_px: list = None,
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
    bar_start_values: list = None,
    expected_baseline: float = 0.0,
    start_values: list = None,
    baseline_values: list = None,
    bar_starts: list = None,
    starts: list = None,
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
    shown_x_values: list = None,
    expected_step: float = 1.0,
    x_values: list = None,
    x_labels: list = None,
    labels: list = None,
    time_points: list = None,
    x_ticks: list = None,
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
    highlighted_indices: list = None,
    all_values: list = None,
    indices: list = None,
    values: list = None,
    emphasized_indices: list = None,
    data_values: list = None,
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
    values: list = None,
    visual_y_px: list = None,
    x_labels: list = None,
    data_values: list = None,
    y_positions: list = None,
    y_coords: list = None,
    y_pixels: list = None,
    actual_values: list = None,
    labels: list = None,
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
def tool_check_cherry_pick_range(
    shown_values: list = None,
    full_start_label: str = "",
    full_end_label: str = "",
    shown_start_label: str = "",
    shown_end_label: str = "",
    values: list = None,
    data_values: list = None,
) -> str:
    """
    표시된 시간 범위가 유리한 구간만 선택(체리피킹)했는지 탐지합니다.

    Args:
        shown_values: 차트에 표시된 순서대로의 데이터 값 리스트 (values, data_values 별칭 허용)
        full_start_label: 전체 데이터의 시작 레이블 (예: '2000년')
        full_end_label: 전체 데이터의 끝 레이블 (예: '2023년')
        shown_start_label: 차트에 표시된 시작 레이블 (예: '2010년')
        shown_end_label: 차트에 표시된 끝 레이블 (예: '2020년')
    """
    vals = shown_values or values or data_values or []
    if not vals:
        return json.dumps({"checked": False, "note": "shown_values 데이터가 필요합니다."}, ensure_ascii=False)
    return json.dumps(
        MathTools.check_cherry_pick_range(vals, full_start_label, full_end_label,
                                          shown_start_label, shown_end_label),
        ensure_ascii=False,
    )


@tool
def tool_check_unit_switch(
    units: list = None,
    unit_labels: list = None,
    values: list = None,
) -> str:
    """
    같은 차트 내에서 단위가 일관성 없이 혼용되는지 탐지합니다.
    %(퍼센트) + %p(퍼센트포인트) 혼용, 억원 + 조원 혼합, % + 절댓값 혼합 등을 감지합니다.

    Args:
        units: 데이터 포인트별 단위 문자열 리스트 (unit_labels 별칭 허용)
               예: ['억원', '억원', '조원'] 또는 ['%', '%p', '%']
        values: 각 단위에 대응하는 데이터 값 리스트 (선택사항)
    """
    u = units or unit_labels or []
    if not u:
        return json.dumps({"checked": False, "note": "units 데이터가 필요합니다."}, ensure_ascii=False)
    v = values or []
    return json.dumps(
        MathTools.check_unit_switch(u, v if v else None),
        ensure_ascii=False,
    )


@tool
def tool_check_cumulative_vs_period(
    values: list = None,
    x_labels: list = None,
    declared_type: str = "period",
    data_values: list = None,
    labels: list = None,
) -> str:
    """
    누적(cumulative) 데이터를 기간(period) 데이터처럼 표시하거나 그 반대로 표시해
    성장을 과장하는지 탐지합니다.

    Args:
        values: 시간 순서대로의 데이터 값 리스트 (data_values 별칭 허용)
                예: 누적 가입자 [100, 250, 480, 800, 1200]
        x_labels: X축 레이블 리스트 (labels 별칭 허용, 선택사항)
        declared_type: 차트가 선언한 데이터 유형 ('period' 또는 'cumulative', 기본값 'period')
    """
    vals = values or data_values or []
    if not vals:
        return json.dumps({"checked": False, "note": "values 데이터가 필요합니다."}, ensure_ascii=False)
    xlbl = x_labels or labels or []
    return json.dumps(
        MathTools.check_cumulative_vs_period(vals, xlbl if xlbl else None, declared_type),
        ensure_ascii=False,
    )


@tool
def tool_check_misleading_annotation(
    annotated_intervals: list = None,
    annotations: list = None,
    intervals: list = None,
    annotation_data: list = None,
) -> str:
    """차트에 추가된 주석(화살표·괄호·강조선)이 실제 변화량을 왜곡 표현하는지 탐지합니다.

    탐지 패턴:
      1. 주석에 표시된 숫자와 실제 변화량 불일치 (수치 조작)
      2. 더 작은 변화에 더 강한 시각적 강조를 달아 오해 유발

    Args:
        annotated_intervals: 주석이 달린 구간 정보 리스트 (annotations, intervals, annotation_data 별칭 허용).
          각 항목: {
            "label": "7월→8월",
            "actual_delta": -4.1,
            "displayed_text": "4.1 감소",
            "visual_emphasis": 1.0
          }
          visual_emphasis: 1.0=보통, 2.0=2배 강조 (화살표 크기·색상 기준 추정값)
    """
    data = annotated_intervals or annotations or intervals or annotation_data or []
    if not data:
        return json.dumps({"misleading": False, "note": "주석 데이터 없음"}, ensure_ascii=False)
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
    tool_check_cherry_pick_range,
    tool_check_unit_switch,
    tool_check_cumulative_vs_period,
]

ALL_TOOLS: list[Any] = OBSERVER_TOOLS + SUBTLE_TOOLS

OBSERVER_TOOLS_BY_NAME: dict[str, Any] = {t.name: t for t in ALL_TOOLS}
