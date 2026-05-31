"""
수학적 검증 도구 모음 — 기본 + 교묘한 오류 탐지
"""
from __future__ import annotations

import math
import re
import statistics


class MathTools:

    # ─────────────────────────────────────────────────────
    # 기본 검증 (기존)
    # ─────────────────────────────────────────────────────

    @staticmethod
    def check_axis_truncation(y_min: float, y_max: float, data_min: float) -> dict:
        truncated = y_min > 0 and data_min >= 0
        exag = None
        if truncated and (y_max - y_min) > 0:
            exag = round(y_max / (y_max - y_min), 2)
        return {
            "truncated":         truncated,
            "y_axis_start":      y_min,
            "exaggeration_ratio": exag,
            "severity":          "높음" if (exag or 1) > 3 else "중간" if (exag or 1) > 1.5 else "낮음",
            "note":              f"Y축이 {y_min}에서 시작 → 실제 차이의 {exag}배 과장" if exag else "정상",
        }

    @staticmethod
    def check_tick_intervals(ticks: list[float]) -> dict:
        ticks = [v for v in ticks if isinstance(v, (int, float))]
        if len(ticks) < 3:
            return {"consistent": True, "note": "눈금 부족"}
        intervals = [round(ticks[i + 1] - ticks[i], 6) for i in range(len(ticks) - 1)]
        mean_i = sum(intervals) / len(intervals)
        cv = round(
            math.sqrt(sum((x - mean_i) ** 2 for x in intervals) / len(intervals))
            / (abs(mean_i) + 1e-9),
            4,
        )
        consistent = cv < 0.05
        return {
            "consistent": consistent,
            "intervals":  intervals,
            "cv":         cv,
            "severity":   "높음" if cv > 0.3 else "중간" if cv > 0.1 else "낮음",
            "note":       f"CV={cv:.3f} → {'불규칙(오류)' if not consistent else '균등(정상)'}",
        }

    @staticmethod
    def check_pie(values: list[float]) -> dict:
        values = [v for v in values if isinstance(v, (int, float))]
        if not values:
            return {"error": "값 없음"}
        total = sum(values)
        neg = any(v < 0 for v in values)
        return {
            "appropriate":   not neg and 98.5 <= total <= 100.5,
            "total":         round(total, 2),
            "has_negatives": neg,
            "verdict":       "부적절" if (neg or total > 100.5 or total < 98.5) else "정상",
            "reason": (
                "음수 포함" if neg
                else f"합계 {round(total, 1)}% (100% 아님)" if (total > 100.5 or total < 98.5)
                else "정상"
            ),
        }

    @staticmethod
    def check_dual_axis(l_min: float, l_max: float, r_min: float, r_max: float) -> dict:
        ls = l_max - l_min
        rs = r_max - r_min
        ratio = rs / ls if ls > 0 else float("inf")
        return {
            "scale_ratio": round(ratio, 2),
            "misleading":  ratio > 10 or ratio < 0.1,
            "note":        f"스케일 비율 {round(ratio, 1)}배 → {'조작 의심' if (ratio > 10 or ratio < 0.1) else '허용범위'}",
        }

    @staticmethod
    def check_proportion_accuracy(visual: list[float], actual: list[float]) -> dict:
        if not actual or sum(actual) == 0:
            return {"error": "값 없음"}
        total = sum(v for v in actual if v >= 0)
        ar = [v / total for v in actual]
        errors = [
            {"idx": i, "vis": round(v * 100, 1), "act": round(a * 100, 1), "diff": round(abs(v - a) * 100, 1)}
            for i, (v, a) in enumerate(zip(visual, ar))
            if abs(v - a) > 0.03
        ]
        return {
            "distortion_detected": bool(errors),
            "max_distortion_pct":  max((e["diff"] for e in errors), default=0),
            "errors":              errors,
        }

    @staticmethod
    def check_log_scale(ticks: list[float]) -> dict:
        ticks = [v for v in ticks if isinstance(v, (int, float)) and v > 0]
        if len(ticks) < 3:
            return {"is_log": False, "note": "판단 불가"}
        ratios = [ticks[i + 1] / ticks[i] for i in range(len(ticks) - 1)]
        mean_r = sum(ratios) / len(ratios)
        cv = math.sqrt(sum((r - mean_r) ** 2 for r in ratios) / len(ratios)) / (mean_r + 1e-9)
        is_log = cv < 0.1 and mean_r > 1.5
        return {
            "is_log": is_log,
            "ratio":  round(mean_r, 2),
            "cv":     round(cv, 4),
            "note":   f"눈금 비율 {round(mean_r, 1)}배 일정 → {'로그 스케일(표시 확인 필요)' if is_log else '선형 스케일'}",
        }

    @staticmethod
    def check_area_distortion(values: list[float], visual_areas: list[float]) -> dict:
        if len(values) < 2 or len(values) != len(visual_areas):
            return {"error": "데이터 부족"}
        errors = []
        base_v, base_a = values[0], visual_areas[0]
        for i in range(1, len(values)):
            expected_ratio = values[i] / base_v
            actual_ratio = visual_areas[i] / base_a
            if abs(actual_ratio - expected_ratio ** 2) < abs(actual_ratio - expected_ratio):
                errors.append({"idx": i, "suspicion": "반지름 비례 사용(면적 왜곡)"})
        return {
            "distortion_detected": bool(errors),
            "errors":              errors,
            "note":                "버블 크기가 값의 제곱에 비례 → 면적 왜곡" if errors else "정상",
        }

    @staticmethod
    def check_item_order(labels: list[str]) -> dict:
        yr = re.compile(r"\b(19|20)\d{2}\b")
        years = [int(yr.search(str(l)).group()) for l in labels if yr.search(str(l))]
        if len(years) < 2 or len(years) != len(labels):
            return {"checked": False, "note": "연도 레이블 부족"}
        asc = years == sorted(years)
        dsc = years == sorted(years, reverse=True)
        return {
            "correct": asc or dsc,
            "years":   years,
            "issue":   None if (asc or dsc) else f"시간 순서 불일치: {years}",
        }

    @staticmethod
    def check_pie_angles(pie_pcts: list[float], pie_angles_deg: list[float], tolerance_deg: float = 18.0) -> dict:
        vals_p = [v for v in pie_pcts if isinstance(v, (int, float))]
        vals_a = [v for v in pie_angles_deg if isinstance(v, (int, float))]
        if len(vals_p) < 2 or len(vals_p) != len(vals_a):
            return {"error": "데이터 부족 또는 개수 불일치"}
        total_pct = sum(vals_p)
        errors = []
        for i, (pct, ang) in enumerate(zip(vals_p, vals_a)):
            expected_deg = (pct / total_pct) * 360.0
            diff_deg = abs(ang - expected_deg)
            diff_pct = diff_deg / 360.0 * 100.0
            if diff_deg > tolerance_deg:
                errors.append({
                    "idx": i, "label_pct": round(pct, 2),
                    "expected_deg": round(expected_deg, 1), "actual_deg": round(ang, 1),
                    "diff_deg": round(diff_deg, 1), "diff_pct": round(diff_pct, 1),
                })
        max_diff = max((e["diff_pct"] for e in errors), default=0.0)
        severity = "높음" if max_diff > 10 else "중간" if max_diff > 5 else "낮음" if max_diff > 0 else "정상"
        return {
            "distortion_detected": bool(errors),
            "errors":              errors,
            "max_diff_pct":        round(max_diff, 1),
            "severity":            severity,
            "note": (
                f"최대 {round(max_diff, 1)}% 각도 왜곡 → misrepresentation" if errors
                else "각도-퍼센트 일치(정상)"
            ),
        }

    @staticmethod
    def check_bar_scale_symmetry(
        left_values: list[float], left_px: list[float],
        right_values: list[float], right_px: list[float],
        threshold: float = 0.20,
    ) -> dict:
        def _clean(vals, pxs):
            return [(v, p) for v, p in zip(vals, pxs)
                    if isinstance(v, (int, float)) and isinstance(p, (int, float)) and v > 0 and p > 0]
        lp = _clean(left_values, left_px)
        rp = _clean(right_values, right_px)

        # 픽셀 없이 값만으로 비교 (폴백): 최대값/평균으로 예상 비율 vs 시각적 최대 비율 비교
        if len(lp) < 2 or len(rp) < 2:
            lv = [v for v in left_values if isinstance(v, (int, float)) and v > 0]
            rv = [v for v in right_values if isinstance(v, (int, float)) and v > 0]
            if len(lv) < 2 or len(rv) < 2:
                return {"error": "데이터 부족"}
            # 값만으로: 두 축의 최대값 범위 비교
            # 동일 스케일이면 (max_left/max_right)가 시각적 비율과 같아야 함
            # 시각적 픽셀 없이 값 범위만으로 판단 불가 → 의심 플래그 반환
            max_l, max_r = max(lv), max(rv)
            avg_l, avg_r = sum(lv)/len(lv), sum(rv)/len(rv)
            # 좌/우 평균 비율 (같은 스케일이면 시각적으로 이 비율이 유지돼야 함)
            value_ratio = (avg_l / avg_r) if avg_r > 0 else 1.0
            return {
                "scale_manipulation": False,
                "note": f"픽셀 추정 없음. 왼쪽 평균:{round(avg_l,1)}, 오른쪽 평균:{round(avg_r,1)}, 비율:{round(value_ratio,2)}. 시각적 막대 길이를 직접 추정해 재호출 필요.",
                "left_max": max_l, "right_max": max_r,
                "suggestion": "시각적 막대 길이를 0~1 비율로 추정해 left_px/right_px에 전달하세요.",
            }

        l_ratios = [p / v for v, p in lp]
        r_ratios = [p / v for v, p in rp]
        scale_l = sum(l_ratios) / len(l_ratios)
        scale_r = sum(r_ratios) / len(r_ratios)
        ratio = scale_l / scale_r if scale_r > 0 else float("inf")
        manipulated = abs(ratio - 1.0) > threshold
        bias = (
            "left_exaggerated" if ratio > 1.0 + threshold
            else "right_exaggerated" if ratio < 1.0 - threshold
            else "symmetric"
        )
        severity = (
            "높음" if abs(ratio - 1.0) > 0.40 else "중간" if abs(ratio - 1.0) > 0.20
            else "낮음" if manipulated else "정상"
        )
        return {
            "scale_manipulation": manipulated,
            "scale_left":  round(scale_l, 4),
            "scale_right": round(scale_r, 4),
            "ratio":       round(ratio, 3),
            "bias_direction": bias,
            "severity":    severity,
            "note": (
                f"좌우 스케일 비율 {round(ratio, 2)}배 차이 → {bias} (비대칭 왜곡)"
                if manipulated else "좌우 스케일 대칭(정상)"
            ),
        }

    # ─────────────────────────────────────────────────────
    # 교묘한 오류 탐지 (신규)
    # ─────────────────────────────────────────────────────

    @staticmethod
    def check_label_value_match(
        label_values: list[float],
        visual_ratios: list[float],
        y_min: float = 0.0,
        y_max: float = 100.0,
        tolerance: float = 0.05,
    ) -> dict:
        if not label_values or len(label_values) != len(visual_ratios):
            return {"error": "데이터 개수 불일치"}
        y_span = y_max - y_min
        if y_span <= 0:
            return {"error": "Y축 범위 오류"}
        errors = []
        for i, (label, ratio) in enumerate(zip(label_values, visual_ratios)):
            expected_ratio = (label - y_min) / y_span
            actual_ratio   = max(0.0, min(1.0, ratio))
            diff = abs(actual_ratio - expected_ratio)
            diff_pct = round(diff * 100, 1)
            if diff > tolerance:
                errors.append({
                    "idx":            i,
                    "label_value":    label,
                    "expected_ratio": round(expected_ratio, 3),
                    "actual_ratio":   round(actual_ratio, 3),
                    "diff_pct":       diff_pct,
                })
        max_diff = max((e["diff_pct"] for e in errors), default=0.0)
        severity = "높음" if max_diff > 10 else "중간" if max_diff > 5 else "낮음" if errors else "정상"
        return {
            "distortion_detected": bool(errors),
            "errors":              errors,
            "max_diff_pct":        round(max_diff, 1),
            "severity":            severity,
            "note": (
                f"레이블-시각 불일치 최대 {round(max_diff,1)}% → data_visual_disproportion/manipulated_annotation"
                if errors else "레이블-시각 일치(정상)"
            ),
        }

    @staticmethod
    def check_selective_annotation(
        annotated_indices: list[int],
        all_values: list[float],
        direction_bias: str = "auto",
    ) -> dict:
        if not all_values or not annotated_indices:
            return {"checked": False, "note": "데이터 부족"}
        n = len(all_values)
        ann_vals   = [all_values[i] for i in annotated_indices if 0 <= i < n]
        unann_vals = [all_values[i] for i in range(n) if i not in annotated_indices]
        if not ann_vals or not unann_vals:
            return {"checked": True, "biased": False, "note": "모든 데이터 주석 있음 또는 없음"}
        ann_mean   = statistics.mean(ann_vals)
        unann_mean = statistics.mean(unann_vals)
        total_mean = statistics.mean(all_values)
        total_std  = statistics.stdev(all_values) if len(all_values) > 1 else 0
        z_score = (ann_mean - total_mean) / total_std if total_std > 0 else 0.0
        biased = abs(z_score) > 0.8
        if z_score > 0.8:
            bias_dir = "상위값만 주석 (긍정적 강조 의심)"
        elif z_score < -0.8:
            bias_dir = "하위값만 주석 (부정적 강조 의심)"
        else:
            bias_dir = "편향 없음"

        # 핵심 추가: 주석이 붙은 값이 최대값이 아닌 경우 → 시선 유도 의심
        # (예: 도넛 차트에서 가장 큰 조각이 아닌 특정 조각에만 큰 숫자 표시)
        max_val = max(all_values)
        annotated_non_max = any(all_values[i] < max_val for i in annotated_indices if 0 <= i < n)
        if annotated_non_max and not biased:
            biased = True
            bias_dir = f"최대값({max_val})이 아닌 값을 주석으로 강조 → 시선 유도 조작 의심"

        # 주석이 없는 값 중 주석 있는 값보다 더 큰 값이 존재하면 확실한 편향
        if unann_vals and ann_vals and max(unann_vals) > max(ann_vals):
            biased = True
            bias_dir = (
                f"미주석 값 중 최대({max(unann_vals):.1f})가 주석 최대({max(ann_vals):.1f})보다 큼 "
                f"→ 더 중요한 값 은폐"
            )

        coverage = round(len(annotated_indices) / n * 100, 1)
        return {
            "biased":            biased,
            "bias_direction":    bias_dir,
            "annotated_mean":    round(ann_mean, 2),
            "unannotated_mean":  round(unann_mean, 2),
            "z_score":           round(z_score, 2),
            "annotation_coverage_pct": coverage,
            "annotated_non_max": annotated_non_max,
            "note": (
                f"주석 편향 감지: {bias_dir} (z={round(z_score,2)}) → selective_emphasis"
                if biased else f"주석 편향 없음 (커버리지:{coverage}%)"
            ),
        }

    @staticmethod
    def check_aspect_ratio(
        width_px: int,
        height_px: int,
        x_data_range: float,
        y_data_range: float,
    ) -> dict:
        if width_px <= 0 or height_px <= 0 or x_data_range <= 0 or y_data_range <= 0:
            return {"error": "유효하지 않은 입력"}
        px_per_x = width_px  / x_data_range
        px_per_y = height_px / y_data_range
        ratio    = px_per_y / px_per_x
        manipulated = ratio > 3.0 or ratio < 0.3
        severity = (
            "높음" if (ratio > 5.0 or ratio < 0.2) else
            "중간" if manipulated else
            "정상"
        )
        direction = (
            "세로 과장 (작은 변화가 급격해 보임)" if ratio > 3.0 else
            "가로 과장 (큰 변화가 평탄해 보임)" if ratio < 0.3 else
            "적절한 종횡비"
        )
        return {
            "aspect_manipulated": manipulated,
            "px_per_x_unit":     round(px_per_x, 2),
            "px_per_y_unit":     round(px_per_y, 2),
            "ratio":             round(ratio, 2),
            "severity":          severity,
            "direction":         direction,
            "note": (
                f"종횡비 {round(ratio,1)}배 편향 → {direction} (aspect_ratio_manipulation)"
                if manipulated else f"종횡비 정상 (비율:{round(ratio,1)})"
            ),
        }

    @staticmethod
    def check_bin_widths(
        bin_edges: list[float],
        visual_widths_px: list[float],
        tolerance: float = 0.10,
    ) -> dict:
        if len(bin_edges) < 3:
            return {"error": "구간 경계값 부족"}
        actual_widths = [bin_edges[i + 1] - bin_edges[i] for i in range(len(bin_edges) - 1)]
        if visual_widths_px and len(visual_widths_px) == len(actual_widths):
            vis_cv = _cv(visual_widths_px)
            act_cv = _cv(actual_widths)
            visual_uniform   = vis_cv < 0.05
            actual_uniform   = act_cv < 0.05
            inconsistent = visual_uniform and not actual_uniform
            errors = []
            if len(set(round(w, 2) for w in actual_widths)) > 1:
                expected = actual_widths[0]
                for i, w in enumerate(actual_widths):
                    if abs(w - expected) / (expected + 1e-9) > tolerance:
                        errors.append({"idx": i, "expected_width": expected, "actual_width": w})
            return {
                "inconsistent":         inconsistent,
                "actual_widths":        [round(w, 2) for w in actual_widths],
                "visual_widths_px":     [round(w, 1) for w in visual_widths_px],
                "visual_cv":            round(vis_cv, 3),
                "actual_cv":            round(act_cv, 3),
                "unequal_bin_errors":   errors,
                "note": (
                    f"시각적으로 균등(CV={vis_cv:.2f})하지만 실제 너비 불균등(CV={act_cv:.2f}) → inconsistent_binning"
                    if inconsistent else "구간 너비 정상"
                ),
            }
        else:
            act_cv = _cv(actual_widths)
            return {
                "inconsistent":   act_cv > 0.05,
                "actual_widths":  [round(w, 2) for w in actual_widths],
                "actual_cv":      round(act_cv, 3),
                "note": f"실제 구간 너비 CV={act_cv:.3f} → {'불균등 의심' if act_cv > 0.05 else '균등'}",
            }

    @staticmethod
    def check_baseline_alignment(
        bar_start_values: list[float],
        expected_baseline: float = 0.0,
        tolerance: float = 0.02,
    ) -> dict:
        if not bar_start_values:
            return {"error": "데이터 없음"}
        errors = []
        for i, start in enumerate(bar_start_values):
            if abs(start - expected_baseline) > tolerance:
                errors.append({
                    "idx":              i,
                    "actual_start":     round(start, 3),
                    "expected_start":   expected_baseline,
                    "diff":             round(abs(start - expected_baseline), 3),
                })
        return {
            "misaligned":     bool(errors),
            "errors":         errors,
            "expected_base":  expected_baseline,
            "note": (
                f"기준선 불일치 {len(errors)}개 → non_aligned_baseline"
                if errors else "모든 막대 기준선 동일(정상)"
            ),
        }

    @staticmethod
    def check_data_gap_detection(
        shown_x_values: list,
        expected_step: float = 1.0,
        is_time_series: bool = True,
    ) -> dict:
        if len(shown_x_values) < 2:
            return {"checked": False, "note": "데이터 부족"}
        numeric = []
        for v in shown_x_values:
            try:
                numeric.append(float(str(v).replace("년", "").replace("Q", ".").strip()))
            except ValueError:
                pass
        if len(numeric) < 2:
            return {"checked": False, "note": "수치 변환 불가"}
        gaps = [numeric[i + 1] - numeric[i] for i in range(len(numeric) - 1)]
        min_gap = min(gaps)
        max_gap = max(gaps)
        suspicious_gaps = []
        for i, g in enumerate(gaps):
            if g > min_gap * 2.5:
                missing_count = round(g / min_gap) - 1
                suspicious_gaps.append({
                    "between_idx": i,
                    "from": numeric[i],
                    "to":   numeric[i + 1],
                    "gap":  round(g, 2),
                    "expected_gap": round(min_gap, 2),
                    "estimated_missing_points": missing_count,
                })
        return {
            "gaps_detected":    bool(suspicious_gaps),
            "suspicious_gaps":  suspicious_gaps,
            "min_gap":          round(min_gap, 2),
            "max_gap":          round(max_gap, 2),
            "gap_ratio":        round(max_gap / (min_gap + 1e-9), 2),
            "note": (
                f"데이터 공백 {len(suspicious_gaps)}곳 (최대:{round(max_gap,1)}/최소:{round(min_gap,1)}) "
                f"→ cherry_picking/cropped_visual_context"
                if suspicious_gaps else "데이터 공백 없음(정상)"
            ),
        }

    @staticmethod
    def check_color_emphasis_bias(
        highlighted_indices: list[int],
        all_values: list[float],
        context: str = "",
    ) -> dict:
        if not all_values or not highlighted_indices:
            return {"checked": False, "note": "데이터 없음"}
        n = len(all_values)
        hi_vals = [all_values[i] for i in highlighted_indices if 0 <= i < n]
        lo_vals = [all_values[i] for i in range(n) if i not in highlighted_indices]
        if not hi_vals:
            return {"checked": False, "note": "강조 데이터 없음"}
        hi_mean  = statistics.mean(hi_vals)
        all_mean = statistics.mean(all_values)
        all_std  = statistics.stdev(all_values) if len(all_values) > 1 else 1e-9
        z = (hi_mean - all_mean) / all_std
        biased = abs(z) > 0.8
        direction = (
            "유리한(높은) 값만 강조" if z > 0.8 else
            "불리한(낮은) 값만 강조" if z < -0.8 else
            "편향 없음"
        )
        return {
            "emphasis_biased":   biased,
            "bias_direction":    direction,
            "highlighted_mean":  round(hi_mean, 2),
            "overall_mean":      round(all_mean, 2),
            "z_score":           round(z, 2),
            "note": (
                f"강조 편향 감지: {direction} (z={round(z,2)}) → visual_saliency_hacking"
                if biased else "색상 강조 편향 없음"
            ),
        }


# ─────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────

def _cv(values: list[float]) -> float:
    """변동계수 (Coefficient of Variation)"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    return std / abs(mean)
