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
        y_span = y_max - y_min
        if y_span <= 0:
            return {"truncated": False, "note": "Y축 범위 오류"}

        # 절단 판정: y축 시작이 0이 아니고, 데이터 최솟값이 0 이상인데 축이 0 위에서 시작
        # 또는: 데이터 최솟값이 음수인데 축 시작이 그보다 훨씬 위인 경우
        if data_min >= 0:
            # 양수 데이터: y_min > 0 이면 절단
            truncated = y_min > 0
        else:
            # 음수 데이터: y_min이 data_min보다 의미있게 위에서 시작하면 절단
            # (data_min의 10% 이상 잘려나간 경우)
            truncated = y_min > data_min and abs(y_min - data_min) > abs(data_min) * 0.1

        exag = None
        if truncated and y_span > 0:
            # 과장 비율: 전체 데이터 범위 대비 표시 범위
            full_range = y_max - data_min if data_min < y_min else y_max
            if full_range > 0:
                exag = round(full_range / y_span, 2)

        return {
            "truncated":          truncated,
            "y_axis_start":       y_min,
            "data_min":           data_min,
            "exaggeration_ratio": exag,
            "severity":           "높음" if (exag or 1) > 3 else "중간" if (exag or 1) > 1.5 else "낮음",
            "note": (
                f"Y축이 {y_min}에서 시작 (데이터 최솟값 {data_min}) → 실제 차이의 {exag}배 과장"
                if truncated and exag is not None
                else "Y축이 잘렸으나 과장 비율 계산 불가"
                if truncated
                else "정상 (Y축이 데이터 범위를 포함)"
            ),
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
            return {"checked": False, "note": "데이터 없음"}
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
    def check_multi_pie_sum(all_pcts: list[float], chart_count: int = 0) -> dict:
        """
        한 이미지에 파이/도넛 차트가 여러 개 있을 때 합계를 개별 검증.

        chart_count=0 이면 자동 감지:
          연속된 값들을 그룹화해 각 그룹이 ~100%가 되는 분할을 탐색.
        """
        values = [v for v in all_pcts if isinstance(v, (int, float)) and v >= 0]
        if not values:
            return {"checked": False, "note": "데이터 없음"}

        total = sum(values)

        # 자동 감지: 누적합이 ~100%가 될 때마다 새 그룹으로 분리 (항상 먼저 시도)
        def _auto_detect(vals):
            grps = []
            cur: list[float] = []
            run = 0.0
            for v in vals:
                cur.append(v)
                run += v
                if 98.0 <= run <= 102.0:
                    grps.append(cur)
                    cur = []
                    run = 0.0
            if cur:
                grps.append(cur)
            return grps

        groups = _auto_detect(values)

        # chart_count >= 2 이고 자동 감지 결과가 기대 개수와 다를 때만 등분 폴백
        if chart_count >= 2 and len(groups) != chart_count:
            n    = chart_count
            size = len(values) // n
            fallback = [values[i * size:(i + 1) * size] for i in range(n)]
            remainder = values[n * size:]
            if remainder:
                fallback[-1].extend(remainder)
            # 자동 감지보다 등분이 더 나은 경우(각 그룹 합계가 100에 더 가까울 때)만 사용
            auto_err = sum(abs(sum(g) - 100) for g in groups)
            fall_err = sum(abs(sum(g) - 100) for g in fallback)
            if fall_err < auto_err:
                groups = fallback

        # 각 그룹 검증
        group_results = []
        all_valid = True
        for i, grp in enumerate(groups):
            s = round(sum(grp), 2)
            ok = 98.0 <= s <= 102.0
            if not ok:
                all_valid = False
            group_results.append({
                "chart_index": i + 1,
                "values": grp,
                "sum": s,
                "valid": ok,
            })

        detected_count = len(groups)
        return {
            "detected_charts": detected_count,
            "total_sum": round(total, 2),
            "each_valid": all_valid,
            "groups": group_results,
            "verdict": "정상" if all_valid else "오류",
            "reason": (
                f"{detected_count}개 차트 각각 합계 정상"
                if all_valid
                else f"{detected_count}개 차트 중 일부 합계 이상: "
                     + ", ".join(
                         f"차트{r['chart_index']}={r['sum']}%"
                         for r in group_results if not r["valid"]
                     )
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
            return {"checked": False, "note": "데이터 없음"}
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
            return {"checked": False, "is_log": False, "note": "눈금 데이터 부족 (3개 이상 필요)"}
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
    def check_area_distortion(values: list[float], visual_sizes: list[float]) -> dict:
        """
        버블/산점도에서 원 크기가 면적이 아닌 반지름에 비례하는지 검증.

        visual_sizes: 절대 픽셀 불필요 — 상대 비율로 전달
                      예) [2.0, 1.0, 3.0] = 첫 번째 원이 두 번째의 2배, 세 번째의 2/3
                      visual_ratios, visual_areas 어느 쪽으로 전달해도 동작
        """
        if len(values) < 2:
            return {"checked": False, "note": "데이터 부족 (values 2개 이상 필요)"}
        if len(visual_sizes) < 2 or len(visual_sizes) != len(values):
            return {"checked": False, "note": "visual_sizes 데이터 부족 또는 길이 불일치"}

        base_v = values[0]
        base_a = visual_sizes[0]
        if base_v == 0 or base_a == 0:
            return {"checked": False, "note": "기준값(첫 번째 요소)이 0 — 비교 불가"}

        errors = []
        for i in range(1, len(values)):
            expected_ratio = values[i] / base_v       # 값 비율 (올바른 면적 비례 시 시각 비율과 같아야 함)
            actual_ratio   = visual_sizes[i] / base_a  # 실제 시각 크기 비율

            if expected_ratio == 1.0:
                continue  # 값이 동일하면 비교 의미 없음

            # 올바른 방식 (면적 비례): visual_area ∝ value   → actual_ratio ≈ expected_ratio
            # 잘못된 방식 (반지름 비례): radius ∝ value → area ∝ value² → actual_ratio ≈ expected_ratio²
            dist_correct = abs(actual_ratio - expected_ratio)
            dist_radius  = abs(actual_ratio - expected_ratio ** 2)

            if dist_radius < dist_correct:
                errors.append({
                    "idx":          i,
                    "value_ratio":  round(expected_ratio, 3),
                    "visual_ratio": round(actual_ratio, 3),
                    "suspicion":    "반지름 비례 사용 — 큰 값이 실제보다 과장되어 보임",
                })

        return {
            "distortion_detected": bool(errors),
            "errors":              errors,
            "note": (
                "원 크기가 값의 제곱에 비례(반지름 비례) → 큰 값일수록 과장"
                if errors else "정상 (면적 비례)"
            ),
        }

    @staticmethod
    def check_item_order(labels: list[str]) -> dict:
        # ── 1순위: 4자리 연도 ──────────────────────────────
        # \b 대신 숫자 경계 사용 (한글 '년' 등 앞뒤로도 매칭)
        yr = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
        years = [int(yr.search(str(l)).group()) for l in labels if yr.search(str(l))]
        if len(years) >= 2 and len(years) == len(labels):
            asc = years == sorted(years)
            dsc = years == sorted(years, reverse=True)
            return {
                "checked": True, "type": "year",
                "correct": asc or dsc,
                "values":  years,
                "issue":   None if (asc or dsc) else f"연도 순서 불일치: {years}",
            }

        # ── 2순위: 분기 (Q1~Q4 또는 1Q~4Q) ────────────────
        qr = re.compile(r"[Qq](\d)|(\d)[Qq]")
        quarters = []
        for l in labels:
            m = qr.search(str(l))
            if m:
                quarters.append(int(m.group(1) or m.group(2)))
        if len(quarters) >= 2 and len(quarters) == len(labels):
            asc = quarters == sorted(quarters)
            dsc = quarters == sorted(quarters, reverse=True)
            return {
                "checked": True, "type": "quarter",
                "correct": asc or dsc,
                "values":  quarters,
                "issue":   None if (asc or dsc) else f"분기 순서 불일치: {quarters}",
            }

        # ── 3순위: 월 (숫자 또는 한국어/영어 월명) ──────────
        month_map = {
            "1월":1,"2월":2,"3월":3,"4월":4,"5월":5,"6월":6,
            "7월":7,"8월":8,"9월":9,"10월":10,"11월":11,"12월":12,
            "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
            "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
            "january":1,"february":2,"march":3,"april":4,"june":6,
            "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
        }
        months = []
        for l in labels:
            key = str(l).strip().lower()
            if key in month_map:
                months.append(month_map[key])
            else:
                m = re.search(r"\b(\d{1,2})월\b", str(l))
                if m:
                    months.append(int(m.group(1)))
        if len(months) >= 2 and len(months) == len(labels):
            asc = months == sorted(months)
            dsc = months == sorted(months, reverse=True)
            return {
                "checked": True, "type": "month",
                "correct": asc or dsc,
                "values":  months,
                "issue":   None if (asc or dsc) else f"월 순서 불일치: {months}",
            }

        # ── 4순위: 순수 숫자 레이블 ──────────────────────────
        nums = []
        for l in labels:
            try:
                nums.append(float(str(l).replace(",", "").strip()))
            except ValueError:
                pass
        if len(nums) >= 2 and len(nums) == len(labels):
            asc = nums == sorted(nums)
            dsc = nums == sorted(nums, reverse=True)
            return {
                "checked": True, "type": "numeric",
                "correct": asc or dsc,
                "values":  nums,
                "issue":   None if (asc or dsc) else f"숫자 순서 불일치: {nums}",
            }

        return {"checked": False, "note": f"순서 판별 불가 (레이블: {labels[:5]})"}

    @staticmethod
    def check_pie_angles(pie_pcts: list[float], pie_angles_deg: list[float], tolerance_deg: float = 18.0) -> dict:
        vals_p = [v for v in pie_pcts if isinstance(v, (int, float))]
        vals_a = [v for v in pie_angles_deg if isinstance(v, (int, float))]
        if len(vals_p) < 2 or len(vals_p) != len(vals_a):
            return {"checked": False, "note": "데이터 부족 또는 개수 불일치"}
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
                return {"checked": False, "note": "데이터 부족"}
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
            return {"checked": False, "note": "데이터 개수 불일치"}
        y_span = y_max - y_min
        if y_span <= 0:
            return {"checked": False, "note": "Y축 범위 오류"}
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
            return {"checked": False, "note": "유효하지 않은 입력"}
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
            return {"checked": False, "note": "구간 경계값 부족 (3개 이상 필요)"}
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
    def check_slope_distortion(
        values: list[float],
        visual_y_px: list[float],
        x_labels: list[str] | None = None,
        tolerance: float = 0.25,
    ) -> dict:
        """
        선 차트에서 실제 데이터 변화량과 시각적 기울기의 불일치를 탐지.

        핵심 원리:
          - 실제 변화량 Δv[i] = values[i+1] - values[i]
          - 시각적 변화량 Δpx[i] = visual_y_px[i] - visual_y_px[i+1]
            (화면 Y좌표는 위가 0이므로 값 감소 = px 감소)
          - 각 구간의 '픽셀/값' 비율(scale_ratio)이 일정해야 정상
          - 어떤 구간이 다른 구간보다 비율이 크면 → 해당 구간을 시각적으로 과장

        예: 4.1 감소 구간보다 3.5 감소 구간이 시각적으로 더 가파른 경우 감지
        """
        n = len(values)
        if n < 3 or len(visual_y_px) != n:
            return {
                "distorted": False,
                "note": f"데이터 부족 (값:{n}개, 픽셀:{len(visual_y_px)}개) — 최소 3개 필요",
            }

        labels = x_labels or [str(i) for i in range(n)]

        # 구간별 실제/시각 변화량 계산
        actual_deltas  = [round(values[i + 1] - values[i], 4) for i in range(n - 1)]
        visual_deltas  = [round(visual_y_px[i] - visual_y_px[i + 1], 2) for i in range(n - 1)]

        # 각 구간 스케일 비율 (px per unit) — 값이 0인 구간은 스킵
        scale_ratios = []
        for i, (av, vv) in enumerate(zip(actual_deltas, visual_deltas)):
            if abs(av) < 1e-9:
                scale_ratios.append(None)
            else:
                scale_ratios.append(round(vv / av, 4))

        valid_ratios = [r for r in scale_ratios if r is not None]
        if len(valid_ratios) < 2:
            return {"distorted": False, "note": "유효 구간 부족"}

        mean_ratio = sum(valid_ratios) / len(valid_ratios)

        # 개별 구간이 평균 대비 얼마나 벗어나는지
        deviations = []
        distorted_intervals = []
        for i, (sr, av, vv) in enumerate(zip(scale_ratios, actual_deltas, visual_deltas)):
            if sr is None:
                continue
            dev = (sr - mean_ratio) / (abs(mean_ratio) + 1e-9)
            deviations.append(round(dev, 4))
            interval_label = f"{labels[i]}→{labels[i+1]}"
            if abs(dev) > tolerance:
                direction = "과장" if dev > 0 else "축소"
                distorted_intervals.append({
                    "interval":      interval_label,
                    "actual_delta":  av,
                    "visual_delta_px": vv,
                    "scale_ratio":   round(sr, 4),
                    "deviation":     round(dev, 4),
                    "direction":     direction,
                    "note": (
                        f"실제 변화 {av:+.2f} 이지만 시각적으로 {direction} "
                        f"(스케일비율 평균대비 {dev:+.0%})"
                    ),
                })

        distorted = len(distorted_intervals) > 0

        # 특수 패턴: 작은 변화가 큰 변화보다 시각적으로 더 가파른 경우
        misleading_pairs = []
        for i in range(len(actual_deltas) - 1):
            for j in range(i + 1, len(actual_deltas)):
                abs_i, abs_j = abs(actual_deltas[i]), abs(actual_deltas[j])
                vabs_i = abs(visual_deltas[i])
                vabs_j = abs(visual_deltas[j])
                if abs_i > 0 and abs_j > 0:
                    # 실제로는 i구간이 더 크지만 시각적으로 j구간이 더 가파른 경우
                    if abs_i > abs_j and vabs_j > vabs_i * (1 + tolerance):
                        misleading_pairs.append({
                            "larger_actual_interval":  f"{labels[i]}→{labels[i+1]} ({actual_deltas[i]:+.2f})",
                            "smaller_actual_interval": f"{labels[j]}→{labels[j+1]} ({actual_deltas[j]:+.2f})",
                            "note": (
                                f"실제 변화는 [{labels[i]}→{labels[i+1]}]({abs_i:.2f}) > "
                                f"[{labels[j]}→{labels[j+1]}]({abs_j:.2f}) 이지만 "
                                f"시각적으로는 반대 → 추세 둔화를 가속처럼 보이게 왜곡"
                            ),
                        })
                        distorted = True

        severity = "높음" if len(distorted_intervals) >= 2 or misleading_pairs else \
                   "중간" if distorted_intervals else "낮음"

        return {
            "distorted":            distorted,
            "actual_deltas":        actual_deltas,
            "visual_deltas_px":     visual_deltas,
            "scale_ratios":         scale_ratios,
            "mean_scale_ratio":     round(mean_ratio, 4),
            "distorted_intervals":  distorted_intervals,
            "misleading_pairs":     misleading_pairs,
            "severity":             severity,
            "note": (
                f"기울기 왜곡 감지: {len(distorted_intervals)}개 구간 과장, "
                f"추세 역전 {len(misleading_pairs)}건"
                if distorted else "기울기 정상 (시각적 기울기와 실제 변화량 일치)"
            ),
        }

    @staticmethod
    def check_misleading_annotation(
        annotated_intervals: list[dict],
        tolerance: float = 0.15,
    ) -> dict:
        """
        주석(화살표·괄호·강조선)이 실제 변화량을 왜곡하여 표현하는지 탐지.

        annotated_intervals 형식:
          [{"label": "7월→8월", "actual_delta": -4.1, "displayed_text": "4.1 감소"},
           {"label": "8월→9월", "actual_delta": -3.5, "displayed_text": "3.5 감소"}]

        탐지 패턴:
          1. 표시 숫자와 실제 변화량 불일치 (값 조작)
          2. 같은 방향 변화인데 한 쪽만 주석 → 선택적 강조
          3. 더 작은 변화에 더 큰 시각적 강조(화살표 크기·색상)를 달아 오해 유발
        """
        if not annotated_intervals:
            return {"misleading": False, "note": "주석 없음"}

        errors = []
        for item in annotated_intervals:
            label    = item.get("label", "?")
            actual   = item.get("actual_delta", None)
            text     = item.get("displayed_text", "")
            emphasis = item.get("visual_emphasis", 1.0)  # 1.0=보통, 2.0=2배 강조

            # 패턴 1: 표시 숫자 ≠ 실제 변화량
            if actual is not None and text:
                nums = re.findall(r"[-+]?\d+\.?\d*", text)
                if nums:
                    displayed_val = float(nums[0])
                    if abs(abs(displayed_val) - abs(actual)) > abs(actual) * tolerance:
                        errors.append({
                            "interval": label,
                            "type": "value_mismatch",
                            "actual_delta": actual,
                            "displayed_value": displayed_val,
                            "note": f"표시값({displayed_val}) ≠ 실제변화({actual:.2f}) → 수치 조작 의심",
                        })

        # 패턴 2: 실제보다 작은 변화에 더 강한 강조가 붙어 있는 경우
        items_with_emphasis = [
            i for i in annotated_intervals
            if i.get("actual_delta") is not None and i.get("visual_emphasis") is not None
        ]
        if len(items_with_emphasis) >= 2:
            for i in range(len(items_with_emphasis)):
                for j in range(i + 1, len(items_with_emphasis)):
                    a, b = items_with_emphasis[i], items_with_emphasis[j]
                    da = abs(a["actual_delta"])
                    db = abs(b["actual_delta"])
                    ea = a.get("visual_emphasis", 1.0)
                    eb = b.get("visual_emphasis", 1.0)
                    if da > db and eb > ea * (1 + tolerance):
                        errors.append({
                            "type": "emphasis_mismatch",
                            "larger_change": f"{a['label']}({da:.2f})",
                            "smaller_change": f"{b['label']}({db:.2f})",
                            "note": (
                                f"실제 변화는 [{a['label']}]({da:.2f})가 크지만 "
                                f"[{b['label']}]({db:.2f})에 더 강한 시각적 강조 → "
                                f"작은 변화를 더 중요하게 보이도록 오도"
                            ),
                        })

        misleading = len(errors) > 0
        return {
            "misleading":   misleading,
            "error_count":  len(errors),
            "errors":       errors,
            "severity":     "높음" if len(errors) >= 2 else "중간" if errors else "낮음",
            "note": (
                f"주석 왜곡 {len(errors)}건 감지"
                if misleading else "주석 정상 (표시값과 실제값 일치)"
            ),
        }

    @staticmethod
    def check_baseline_alignment(
        bar_start_values: list[float],
        expected_baseline: float = 0.0,
        tolerance: float = 0.02,
    ) -> dict:
        if not bar_start_values:
            return {"checked": False, "note": "데이터 없음"}
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
        _qr = re.compile(r"(?:(\d{4})\s*[Qq](\d)|[Qq](\d)\s*(\d{4}))")
        for v in shown_x_values:
            s = str(v).strip()
            # 분기 형식 우선 처리: "2024Q3", "Q3 2024", "3Q2024" 등
            qm = _qr.search(s)
            if qm:
                year = qm.group(1) or qm.group(4)
                qnum = qm.group(2) or qm.group(3)
                numeric.append(float(f"{year}.{qnum}"))
                continue
            try:
                numeric.append(float(s.replace("년", "").replace(",", "").strip()))
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


    @staticmethod
    def check_cherry_pick_range(
        shown_values: list[float],
        full_start_label: str = "",
        full_end_label: str = "",
        shown_start_label: str = "",
        shown_end_label: str = "",
    ) -> dict:
        """
        표시된 시간 범위가 유리한 구간만 선택(체리피킹)했는지 탐지.

        탐지 패턴:
          1. 시작점 ≈ 최저점 & 끝점 ≈ 최고점 → 상승 narrative 조작
          2. 시작점 ≈ 최고점 & 끝점 ≈ 최저점 → 하락 narrative 조작
          3. 전체 범위 대비 표시 범위가 지나치게 짧음 (레이블 제공 시)
          4. 내부에 큰 반등/하락이 있지만 시작~끝만 보면 단순 추세처럼 보이는 경우
        """
        vals = [v for v in (shown_values or []) if isinstance(v, (int, float))]
        if len(vals) < 3:
            return {"checked": False, "note": "데이터 부족 (최소 3개 필요)"}

        v_min   = min(vals)
        v_max   = max(vals)
        v_range = v_max - v_min
        if v_range == 0:
            return {"checked": True, "cherry_picked": False, "note": "데이터 변화 없음"}

        start_val = vals[0]
        end_val   = vals[-1]

        # 15% 이내이면 "근처"로 판정
        start_near_min = (start_val - v_min) / v_range < 0.15
        end_near_max   = (v_max - end_val)   / v_range < 0.15
        start_near_max = (v_max - start_val) / v_range < 0.15
        end_near_min   = (end_val - v_min)   / v_range < 0.15

        if start_near_min and end_near_max:
            cherry_picked = True
            direction     = "상승 체리피킹 (최저점 시작 → 최고점 끝)"
        elif start_near_max and end_near_min:
            cherry_picked = True
            direction     = "하락 체리피킹 (최고점 시작 → 최저점 끝)"
        else:
            cherry_picked = False
            direction     = "편향 없음"

        # 내부 변동성 은폐 탐지 — 최대 구간 변화가 전체 변화의 2배 이상이면 의심
        trend_actual     = abs(end_val - start_val)
        max_single_swing = max(abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1))
        hidden_volatility = max_single_swing > trend_actual * 2.0

        # 전체 범위 레이블 정보 (있을 때만)
        range_info = ""
        if full_start_label and full_end_label and shown_start_label and shown_end_label:
            range_info = (
                f"전체 범위 {full_start_label}~{full_end_label} 중 "
                f"{shown_start_label}~{shown_end_label}만 표시"
            )

        severity = "높음" if cherry_picked else "낮음"
        note_parts = []
        if cherry_picked:
            note_parts.append(f"체리피킹 감지: {direction}")
        if range_info:
            note_parts.append(range_info)
        if hidden_volatility:
            note_parts.append("내부 변동성 은폐 의심")

        return {
            "cherry_picked":     cherry_picked,
            "bias_direction":    direction,
            "start_value":       round(start_val, 3),
            "end_value":         round(end_val, 3),
            "min_value":         round(v_min, 3),
            "max_value":         round(v_max, 3),
            "hidden_volatility": hidden_volatility,
            "range_info":        range_info,
            "severity":          severity,
            "note": (
                " | ".join(note_parts)
                if note_parts else "정상 (시작/끝점 편향 없음)"
            ),
        }

    @staticmethod
    def check_unit_switch(
        units: list[str],
        values: list[float] = None,
    ) -> dict:
        """
        같은 차트 내에서 단위가 일관성 없이 혼용되는지 탐지.

        탐지 패턴:
          1. 화폐 스케일 혼합 (억원 + 조원, 달러 + 원)
          2. 이질 단위 혼합 (% + 절댓값, 명 + 건)
          3. %와 %p(퍼센트포인트) 혼용
        """
        if not units:
            return {"checked": False, "note": "단위 데이터 없음"}

        cleaned = [str(u).strip() for u in units if u is not None and str(u).strip()]
        if not cleaned:
            return {"checked": False, "note": "유효한 단위 없음"}

        # ── 단위 카테고리 분류 ────────────────────────────────
        KR_MONEY_SCALES = ["조원", "억원", "만원", "원", "billion", "million", "trillion",
                           "조달러", "억달러", "만달러", "달러", "$"]
        PERCENT       = {"%", "퍼센트", "percent", "pct"}
        PERCENT_POINT = {"%p", "%포인트", "pp", "percentage point", "퍼센트포인트"}
        COUNT_UNITS   = ["억명", "만명", "명", "만건", "건", "억개", "만개", "개", "회", "곳"]

        def _categorize(u: str) -> str:
            ul = u.lower()
            if ul in PERCENT:       return "percent"
            if ul in PERCENT_POINT: return "percent_point"
            for s in KR_MONEY_SCALES:
                if ul == s.lower() or ul.endswith(s.lower()):
                    return "money"
            for s in COUNT_UNITS:
                if ul.endswith(s):
                    return "count"
            return "other"

        categories   = [_categorize(u) for u in cleaned]
        unique_cats  = set(c for c in categories if c != "other")

        # ── 화폐 스케일 혼합 체크 ─────────────────────────────
        money_scales = []
        scale_order  = ["조원", "억원", "만원", "원"]
        for u in cleaned:
            ul = u.lower()
            for s in scale_order:
                if ul.endswith(s.lower()):
                    money_scales.append(s)
                    break
        scale_mixed = len(set(money_scales)) > 1

        # ── 카테고리 혼합 체크 ───────────────────────────────
        # percent + percent_point 혼용은 별도 플래그
        pct_mixed = "percent" in unique_cats and "percent_point" in unique_cats
        # 완전히 다른 단위 카테고리 혼합
        cat_mixed = len(unique_cats - {"percent", "percent_point"}) > 0 and (
            "percent" in unique_cats or "percent_point" in unique_cats
        ) or len({c for c in unique_cats if c in ("money", "count")}) > 1

        mixed = scale_mixed or cat_mixed or pct_mixed

        if pct_mixed:
            issue = "%(퍼센트)와 %p(퍼센트포인트) 혼용 → 의미가 다른 단위를 같이 표시"
        elif cat_mixed:
            issue = f"이질 단위 혼합: {', '.join(sorted(unique_cats))}"
        elif scale_mixed:
            issue = f"화폐 스케일 혼합: {', '.join(sorted(set(money_scales)))}"
        else:
            issue = "단위 일관성 있음"

        severity = "높음" if cat_mixed else "중간" if (scale_mixed or pct_mixed) else "정상"

        return {
            "unit_switch_detected": mixed,
            "units_found":          list(set(cleaned)),
            "categories":           list(unique_cats) if unique_cats else ["other"],
            "scale_mixed":          scale_mixed,
            "category_mixed":       cat_mixed,
            "pct_vs_pctp_mixed":    pct_mixed,
            "severity":             severity,
            "note": (
                f"단위 불일치 감지: {issue} → unit_switch"
                if mixed else "단위 일관성 정상"
            ),
        }

    @staticmethod
    def check_cumulative_vs_period(
        values: list[float],
        x_labels: list[str] = None,
        declared_type: str = "period",
    ) -> dict:
        """
        누적 데이터를 기간 데이터처럼 표시(또는 반대)하여 성장을 과장하는지 탐지.

        탐지 패턴:
          1. 값이 단조 증가(감소 없음)인데 기간 데이터로 선언 → 누적 의심
          2. 평균 변화량이 총계의 20% 미만 → 누적 데이터 특성
          3. 연속적인 값이 합산 가능하지 않음에도 누적 표시 사용
        """
        vals = [v for v in (values or []) if isinstance(v, (int, float))]
        if len(vals) < 3:
            return {"checked": False, "note": "데이터 부족 (최소 3개 필요)"}

        n = len(vals)

        # ── 단조성 판정 ───────────────────────────────────────
        decreases       = sum(1 for i in range(n - 1) if vals[i + 1] < vals[i])
        increases       = sum(1 for i in range(n - 1) if vals[i + 1] > vals[i])
        decrease_ratio  = decreases / (n - 1)
        non_decreasing  = decreases == 0
        non_increasing  = increases == 0
        monotonic       = non_decreasing or non_increasing

        # ── 변화량 분석 ───────────────────────────────────────
        deltas = [abs(vals[i + 1] - vals[i]) for i in range(n - 1)]
        avg_delta = sum(deltas) / len(deltas) if deltas else 0
        final_val = vals[-1] if vals[-1] != 0 else 1.0
        delta_to_total = avg_delta / abs(final_val) if final_val != 0 else 0

        # ── 누적 가능성 점수 (0 ~ 1.0) ──────────────────────
        score = 0.0
        if non_decreasing:                    score += 0.50  # 한 번도 감소 안 함
        elif decrease_ratio < 0.15:           score += 0.25  # 거의 감소 없음
        if delta_to_total < 0.20:             score += 0.30  # 변화량이 총계 대비 소
        if vals[0] >= 0 and non_decreasing:   score += 0.20  # 0 이상에서 단조 증가

        looks_cumulative = score >= 0.70

        # ── declared_type과 비교 → 오류 판정 ─────────────────
        if declared_type == "period" and looks_cumulative:
            misleading = True
            issue = (
                "기간(period) 데이터로 표시됐지만 누적 패턴 강하게 의심 "
                "→ cumulative_misrepresentation"
            )
        elif declared_type == "cumulative" and not looks_cumulative:
            misleading = True
            issue = "누적 데이터로 표시됐지만 기간 데이터 패턴"
        else:
            misleading = False
            issue = ""

        # 급격한 성장률 일관성 체크 — 누적이면 구간별 성장률이 감소하는 경향
        growth_rates = []
        for i in range(n - 1):
            if vals[i] != 0:
                growth_rates.append((vals[i + 1] - vals[i]) / abs(vals[i]))
        avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0

        severity = (
            "높음" if (misleading and score >= 0.80)
            else "중간" if misleading
            else "정상"
        )

        return {
            "misleading":           misleading,
            "declared_type":        declared_type,
            "looks_cumulative":     looks_cumulative,
            "cumulative_score":     round(score, 2),
            "is_monotonic":         monotonic,
            "decrease_ratio":       round(decrease_ratio, 2),
            "delta_to_total_ratio": round(delta_to_total, 3),
            "avg_period_growth_rate": round(avg_growth, 4),
            "severity":             severity,
            "note": (
                issue if misleading
                else (
                    f"정상 ({declared_type} 패턴 확인, "
                    f"누적 점수:{score:.1f}, 감소 비율:{decrease_ratio:.0%})"
                )
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
