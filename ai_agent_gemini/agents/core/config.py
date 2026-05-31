"""
설정값, 분류체계 상수, 정규화 유틸

MISLEADER_TAXONOMY 설계 원칙 (v3, 29종):
  - MisViz 공식 12종 전체 포함 (데이터셋 호환)
  - 겹치는 범주 통합: 38종 → 29종
  - 각 항목이 서로 명확히 구분 가능
  - 검출 도구(lc_tools.py)와 1:1 대응 원칙

참고 문헌:
  - Lo et al. (2022) "Misinformed by Visualization" 12-type taxonomy
  - MisViz dataset (UKPLab/misviz)
  - Misleading Chart Deception taxonomy (2025 papers)
"""
from __future__ import annotations

import os
import re

# ─────────────────────────────────────────────────────────
# LLM / 이미지 설정
# ─────────────────────────────────────────────────────────
LLM_NAME     = os.environ.get("CHART_MODEL", "gemini-3.5-flash")
# ── Gemini 3.5 계열 (2025, 최신) ─────────────────────────
# gemini-3.5-flash      → GA. 100만 토큰 컨텍스트, 65K 출력 (기본값)
# gemini-3.1-pro-preview→ 3.1 Pro 프리뷰
# gemini-3.1-flash-lite → 경량 Flash
# gemini-3-flash-preview→ 3.0 Flash 프리뷰
# ── Gemini 2.5 계열 (안정) ───────────────────────────────
# gemini-2.5-pro        → 최고 정확도 (비용 높음)
# gemini-2.5-flash      → 이전 Flash (안정)
# gemini-2.5-flash-lite → 경량
MODEL        = LLM_NAME
IMAGE_TOKENS = int(os.environ.get("CHART_IMAGE_TOKENS", "1120"))

# ── Gemini 3.5 Flash 전용 설정 ───────────────────────────
# temperature/top_p/top_k 는 3.5-flash 에서 deprecated
# 대신 thinking_level 로 품질/속도 트레이드오프 조정
# "minimal" | "low" | "medium"(기본) | "high"
THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "minimal")

_RESPONSE_CACHE: dict[str, str] = {}

# ─────────────────────────────────────────────────────────
# MisViz RAG 설정
# ─────────────────────────────────────────────────────────
HF_TOKEN             = os.environ.get("HF_TOKEN", "")
MISVIZ_DATASET       = "UKPLab/misviz"
MISVIZ_DB_DIR        = os.environ.get("MISVIZ_DB_DIR", "./misviz_vectordb")
MISVIZ_IMG_DIR       = os.environ.get("MISVIZ_IMG_DIR", "./misviz_images")
MISVIZ_EMBED         = os.environ.get(
    "MISVIZ_EMBED",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
MISVIZ_CONTENT_SCHEMA = "v2-taxonomy-enriched"

# ─────────────────────────────────────────────────────────
# Misleader 분류체계 (29종, v3)
#
# 카테고리 구성:
#   A. 축 조작 (5)   : truncated_axis, inverted_axis, dual_axis,
#                       log_scale_unlabeled, inappropriate_range
#   B. 스케일 조작 (3): inconsistent_tick, aspect_ratio_manipulation,
#                       non_aligned_baseline
#   C. 시각 인코딩 왜곡 (5): data_visual_disproportion, misrepresentation,
#                              area_distortion, distorted_projection, 3d_effect
#   D. 데이터 선택 조작 (4): cherry_picking, inappropriate_order,
#                              inconsistent_binning, cropped_visual_context
#   E. 차트 유형 오용 (3): inappropriate_line,
#                           discretized_continuous, stacked_chart_misuse
#   F. 강조 조작 (3)  : selective_emphasis, misleading_color, manipulated_annotation
#   G. 정보 은폐 (4)  : missing_label, hidden_uncertainty,
#                        sample_size_hidden, decorative_chartjunk
#   H. 특수 (1)       : false_correlation
# ─────────────────────────────────────────────────────────
MISLEADER_TAXONOMY: dict[str, dict] = {

    # ── A. 축 조작 ────────────────────────────────────────
    "truncated_axis": {
        "name": "축 절단 (Truncated Axis)",
        "desc": "Y축이 0이 아닌 값에서 시작해 미세한 차이를 극적으로 과장. "
                "물결선(~) 또는 축 시작값이 데이터 최솟값 근처.",
        "detection": "Y축 시작값 확인. 0이 아니면 절단 의심. "
                     "도구: tool_check_axis_truncation(y_min, y_max, data_min)",
        "example": "Y축 83→87, 실제 차이 1.2%인데 막대가 2.5배 차이로 보임",
        "chart_types": ["막대차트", "선차트", "영역차트"],
        "severity_weight": 0.90,
    },
    "inverted_axis": {
        "name": "역전 축 (Inverted Axis)",
        "desc": "Y축이 관례와 반대 방향(위→아래로 값 증가). "
                "사망자 증가가 그래프에서 하락처럼 보이는 Florida 총기 사례.",
        "detection": "Y축 눈금이 위=큰값, 아래=작은값이면 역전 아님. "
                     "위=작은값(0), 아래=큰값이면 역전.",
        "example": "Florida 총기 사망자 그래프: 총기법 시행 후 사망자 급증인데 막대가 줄어드는 방향",
        "chart_types": ["막대차트", "선차트", "산점도"],
        "severity_weight": 0.85,
    },
    "dual_axis": {
        "name": "이중 축 조작 (Dual Axis Manipulation)",
        "desc": "독립적인 두 Y축의 스케일을 조작해 상관관계나 인과관계를 제조. "
                "두 축의 범위 비율을 바꿔 원하는 교차점을 만듦.",
        "detection": "차트에 왼쪽/오른쪽 Y축이 별도로 있는지 확인. "
                     "두 선이 교차하도록 스케일 맞춰진 경우 의심.",
        "example": "아이스크림 판매량 + 익사 사망자를 이중 축으로 완벽 교차하도록 설정",
        "chart_types": ["선차트", "막대차트", "혼합차트"],
        "severity_weight": 0.85,
    },
    "log_scale_unlabeled": {
        "name": "로그 스케일 미표시 (Unlabeled Log Scale)",
        "desc": "로그 스케일을 사용하면서 'log scale'이나 지수 표기를 하지 않아 "
                "독자가 선형으로 오해. 눈금이 1, 10, 100, 1000처럼 등비수열.",
        "detection": "Y 눈금값이 10배씩 증가하는데 시각 간격이 균등하면 로그. "
                     "레이블에 'log', '로그', 지수 표기 없으면 미표시.",
        "example": "코로나 확진자 수 1→10→100→1000이 동일 간격, 축 레이블에 설명 없음",
        "chart_types": ["선차트", "막대차트", "산점도"],
        "severity_weight": 0.80,
    },
    "inappropriate_range": {
        "name": "부적절한 축 범위 (Inappropriate Axis Range)",
        "desc": "축 범위를 데이터 실제 변동폭보다 훨씬 넓게 설정해 큰 변화를 평탄하게 보이게 하거나, "
                "반대로 너무 좁게 설정해 작은 변화를 극적으로 보이게 함.",
        "detection": "데이터 범위 대비 축 범위 비율 확인. "
                     "데이터가 10-12인데 0-1000까지 표시 → 평탄화. "
                     "데이터가 0-100인데 95-100만 표시 → truncated_axis와 구분.",
        "example": "0.01% 차이 데이터를 0~0.015% 범위로 확대해 극적으로 보이게",
        "chart_types": ["선차트", "막대차트", "산점도", "영역차트"],
        "severity_weight": 0.75,
    },

    # ── B. 스케일 조작 ────────────────────────────────────
    "inconsistent_tick": {
        "name": "불규칙 눈금 (Inconsistent Tick Intervals)",
        "desc": "눈금 간격이 시각적으로는 균등하지만 실제 데이터 값 간격이 다름. "
                "또는 시간축에서 1990, 2000, 2005, 2006이 동일 픽셀 폭.",
        "detection": "연속된 눈금값 차이 목록에서 최대/최소 비율 > 1.5이면 불규칙. "
                     "도구: tool_check_tick_intervals(tick_values)",
        "example": "X축: 1990, 2000, 2005, 2006이 같은 픽셀 간격 → 10년=5년=1년=4년처럼 보임",
        "chart_types": ["선차트", "막대차트", "산점도"],
        "severity_weight": 0.80,
    },
    "aspect_ratio_manipulation": {
        "name": "종횡비 조작 (Aspect Ratio Manipulation)",
        "desc": "차트를 세로로 매우 길게 만들어 작은 변화가 급등처럼 보이게 하거나, "
                "가로로 넓게 만들어 큰 변화가 평탄해 보이게 함.",
        "detection": "차트 높이/폭 비율이 2:1 이상이면 의심. "
                     "기울기가 데이터 값 변화에 비해 극단적으로 가파르면 조작 의심.",
        "example": "월별 0.5% 성장률을 세로 4000px 차트로 그려 수직 급등처럼 표현",
        "chart_types": ["선차트", "영역차트", "막대차트"],
        "severity_weight": 0.65,
    },
    "non_aligned_baseline": {
        "name": "기준선 불일치 (Non-aligned Baseline)",
        "desc": "막대들의 시작 위치(기준선)가 서로 달라 직접 높이 비교가 불가능. "
                "부분 막대를 띄워서(floating bar) 차이를 왜곡.",
        "detection": "모든 막대가 동일한 y=0 또는 동일한 기준값에서 시작하는지 확인. "
                     "시작값이 다른 막대가 있으면 해당.",
        "example": "A막대는 y=0에서, B막대는 y=20에서 시작 → B가 실제보다 훨씬 커 보임",
        "chart_types": ["막대차트", "영역차트", "스택차트"],
        "severity_weight": 0.75,
    },

    # ── C. 시각 인코딩 왜곡 ───────────────────────────────
    "data_visual_disproportion": {
        "name": "데이터-시각 불비례 (Data-Visual Disproportion)",
        "desc": "시각적 크기(막대 높이·길이·선 위치)와 레이블 수치의 비율 관계가 맞지 않음. "
                "레이블은 100인데 막대가 시각적으로 200처럼 보이는 경우.",
        "detection": "막대A/막대B 시각 비율 vs 값A/값B 비율 비교. "
                     "도구: tool_check_label_value_match(labeled_values, visual_ratios)",
        "example": "A=50, B=60인데 시각적으로 B가 A의 3배 크기",
        "chart_types": ["막대차트", "선차트", "산점도", "영역차트"],
        "severity_weight": 0.90,
    },
    "misrepresentation": {
        "name": "시각 인코딩 오류 (Misrepresentation)",
        "desc": "레이블/데이터와 시각 요소가 구조적으로 불일치. "
                "파이 조각 각도가 퍼센트와 다르거나, 막대 레이블 값이 실제 막대 높이와 다른 숫자.",
        "detection": "파이차트: 각도×100/360 ≠ 퍼센트. 막대차트: 레이블 숫자 ≠ 막대 높이 수치. "
                     "도구: tool_check_pie_angles, tool_check_label_value_match",
        "example": "파이 조각이 45°(12.5%)인데 레이블은 30%로 표시",
        "chart_types": ["파이차트", "도넛차트", "막대차트", "버블차트"],
        "severity_weight": 0.90,
    },
    "area_distortion": {
        "name": "면적·크기 왜곡 (Area/Size Distortion)",
        "desc": "버블·원·아이콘의 크기를 값에 비례시킬 때 면적 대신 반지름에 비례시켜 "
                "차이를 제곱으로 과장. 값 2배 → 면적 4배.",
        "detection": "원형 요소의 반지름 비율 vs 값 비율 비교. "
                     "반지름 비율 = 값 비율이면 면적 왜곡. "
                     "도구: tool_check_area_distortion(data_values, visual_areas)",
        "example": "GDP A=100, B=200인데 B 버블 반지름이 2배 → 시각 면적은 4배",
        "chart_types": ["버블차트", "픽토그램", "산점도"],
        "severity_weight": 0.80,
    },
    "distorted_projection": {
        "name": "왜곡된 투영·원근 (Distorted Projection)",
        "desc": "2D 데이터를 비표준 3D 투영·사선·기울기로 렌더링해 시각적 착시 유발. "
                "카메라 각도에 따라 앞쪽 요소가 실제보다 크게 보임.",
        "detection": "차트가 기울어져 있거나 사선 배치된 경우. "
                     "3D 효과로 앞/뒤 막대 크기가 다르게 보이는 경우.",
        "example": "3D 파이차트에서 앞 조각이 45°로 기울어져 같은 크기의 뒤 조각보다 2배 커 보임",
        "chart_types": ["파이차트", "막대차트", "영역차트", "3D차트"],
        "severity_weight": 0.80,
    },
    "3d_effect": {
        "name": "3D 효과 왜곡 (3D Effect)",
        "desc": "불필요한 3D 막대·입체 효과로 데이터 정확한 비교 방해. "
                "distorted_projection이 렌더링 왜곡이라면 3d_effect는 스타일 선택 왜곡.",
        "detection": "막대/원이 입체적으로 그려져 있는가. "
                     "상단면·측면이 보이는 막대 = 3D 효과.",
        "example": "입체 막대차트에서 측면 면적이 막대 높이에 추가되어 실제보다 커 보임",
        "chart_types": ["막대차트", "파이차트", "도넛차트"],
        "severity_weight": 0.75,
    },

    # ── D. 데이터 선택 조작 ──────────────────────────────
    "cherry_picking": {
        "name": "체리피킹 (Cherry Picking / Selective Time Range)",
        "desc": "전체 데이터 중 주장에 유리한 기간·범위만 선택. "
                "전체 추세는 하락인데 일시적 상승 구간만 표시.",
        "detection": "표시된 기간이 전체 데이터 기간의 일부인지 확인. "
                     "시작점과 끝점이 주장에 유리하게 선택된 것은 아닌지.",
        "example": "2000-2023년 데이터 중 1998-2012년만 사용해 온난화 정체처럼 보이게",
        "chart_types": ["선차트", "막대차트", "영역차트"],
        "severity_weight": 0.80,
    },
    "inappropriate_order": {
        "name": "부적절한 순서 (Inappropriate Item Order)",
        "desc": "날짜·알파벳 등 기대되는 자연 순서를 무시해 패턴을 왜곡하거나 "
                "특정 값이 두드러지도록 임의 정렬.",
        "detection": "X축 레이블이 날짜/숫자면 오름차순·시간순인지 확인. "
                     "도구: tool_check_item_order(x_labels)",
        "example": "월별 데이터를 1,3,2,6,4,5월 순서로 정렬해 패턴 숨김",
        "chart_types": ["막대차트", "선차트", "히스토그램"],
        "severity_weight": 0.70,
    },
    "inconsistent_binning": {
        "name": "불균등 구간 (Inconsistent Binning)",
        "desc": "히스토그램 구간 너비가 시각적으로 균등해 보이지만 실제 범위가 다름. "
                "좁은 구간에 많은 데이터가 있어 빈도가 과장되거나 축소됨.",
        "detection": "히스토그램 구간 경계값 차이가 일정한지 확인. "
                     "도구: tool_check_bin_widths(bin_edges, visual_widths_px)",
        "example": "구간 너비: 0-10, 10-20, 20-50, 50-100이 같은 픽셀 폭으로 표시",
        "chart_types": ["히스토그램"],
        "severity_weight": 0.75,
    },
    "cropped_visual_context": {
        "name": "맥락 절단 (Cropped Visual Context)",
        "desc": "차트가 전체 데이터 범위를 보여주지 않고 일부만 잘라 표시. "
                "cherry_picking이 시간축 선택이라면, 이것은 공간적 클리핑.",
        "detection": "데이터 포인트가 차트 경계에서 갑자기 끊기는 경우. "
                     "스케일과 맞지 않게 일부 막대/선이 잘린 경우.",
        "example": "100을 넘는 막대가 있는데 Y축이 80까지만 표시 → 일부 막대 상단 잘림",
        "chart_types": ["막대차트", "선차트", "영역차트"],
        "severity_weight": 0.70,
    },

    # ── E. 차트 유형 오용 ────────────────────────────────
    "inappropriate_line": {
        "name": "부적절한 선차트 (Inappropriate Line Chart)",
        "desc": "연속성이 없는 범주형·이산형 데이터에 꺾은선 그래프 사용. "
                "존재하지 않는 중간값을 암시하는 연결선.",
        "detection": "X축이 순서 없는 범주(국가명, 브랜드명 등)인데 선으로 연결된 경우.",
        "example": "국가별 GDP(범주)를 꺾은선으로 연결해 '추세'처럼 보이게",
        "chart_types": ["선차트"],
        "severity_weight": 0.70,
    },
    "discretized_continuous": {
        "name": "연속변수 임의 이산화 (Discretized Continuous Variable)",
        "desc": "연속 변수를 임의 기준으로 나눠 경계 근처 데이터의 차이를 극대화. "
                "컷오프 기준을 유리한 위치에 설정.",
        "detection": "실제 연속 데이터(시험 점수, 소득 등)를 막대로 표시하면서 "
                     "구간 기준이 자의적인 경우.",
        "example": "연속 소득 데이터를 '중하층/중층/중상층'으로 나눠 극적 차이처럼 보이게",
        "chart_types": ["막대차트", "히스토그램"],
        "severity_weight": 0.65,
    },
    "stacked_chart_misuse": {
        "name": "누적 차트 오용 (Stacked Chart Misuse)",
        "desc": "누적 막대/영역 차트에서 중간 계층 값 비교가 불가능하게 기준선이 다름. "
                "또는 합산이 의미 없는 데이터를 누적으로 표시.",
        "detection": "누적 막대에서 두 번째 이상 계층의 값을 직접 비교하려면 "
                     "시작점이 달라 불가능한지 확인.",
        "example": "A국(기저10+추가30)과 B국(기저40+추가10)을 누적으로 보여주면 추가분 비교 불가",
        "chart_types": ["누적막대차트", "영역차트", "스택차트"],
        "severity_weight": 0.70,
    },

    # ── F. 강조 조작 ──────────────────────────────────────
    "selective_emphasis": {
        "name": "선택적 강조 (Selective Emphasis)",
        "desc": "특정 데이터에만 ① 색상 강조(빨간색/원색) ② 주석·화살표 ③ 크기 확대 "
                "④ 특정 막대만 다른 색 등을 적용해 나머지 데이터를 무시하게 유도.",
        "detection": "막대 색상이 대부분 회색인데 특정 막대만 강조색인 경우. "
                     "일부 데이터에만 레이블이 붙어있는 경우. "
                     "도구: tool_check_color_emphasis_bias, tool_check_selective_annotation",
        "example": "12개 막대 중 가장 높은 1개만 빨간색으로 강조하고 레이블 부착",
        "chart_types": ["막대차트", "선차트", "파이차트"],
        "severity_weight": 0.65,
    },
    "misleading_color": {
        "name": "오도적 색상 (Misleading Color Encoding)",
        "desc": "① 색상 스케일이 선형 데이터에 비선형으로 적용 "
                "② 관례와 반대 색상(빨강=낮음, 초록=높음) "
                "③ 히트맵에서 중간값을 임의 경계로 설정.",
        "detection": "색상 범례가 있는 경우 색상 스케일의 간격이 균등한지 확인. "
                     "적/녹 색상이 반직관적으로 사용됐는지.",
        "example": "히트맵에서 중간값 50을 흰색으로 설정해 49와 51이 극적으로 달라 보이게",
        "chart_types": ["히트맵", "코로플레스맵", "버블차트", "산점도"],
        "severity_weight": 0.65,
    },
    "manipulated_annotation": {
        "name": "조작된 주석·레이블 (Manipulated Annotation)",
        "desc": "수치 레이블, 범례, 제목을 실제 데이터와 다르게 기재하거나 "
                "범례 색상 순서를 바꿔 잘못된 대응을 유도.",
        "detection": "막대 위의 숫자가 막대 높이와 비례하는지 확인. "
                     "범례 항목과 차트 색상의 대응이 올바른지 확인.",
        "example": "A=70, B=80인데 막대 위 레이블은 반대로 A=80, B=70으로 기재",
        "chart_types": ["막대차트", "파이차트", "선차트"],
        "severity_weight": 0.85,
    },

    # ── G. 정보 은폐 ──────────────────────────────────────
    "missing_label": {
        "name": "레이블·단위 누락 (Missing Label / Unit)",
        "desc": "① Y축 레이블·단위 없음 ② 출처 없음 ③ 기준 시점 없음 "
                "④ 범례 없음. 독자가 자의적으로 단위를 해석하게 유도.",
        "detection": "Y축에 단위(%, 원, 달러, 명 등)가 있는지 확인. "
                     "제목, 출처, 기준년도가 명시됐는지.",
        "example": "Y축 숫자만 있고 단위 없음 → 독자가 억원인지 만원인지 모름",
        "chart_types": ["모든 차트"],
        "severity_weight": 0.60,
    },
    "hidden_uncertainty": {
        "name": "불확실성 은폐 (Hidden Uncertainty)",
        "desc": "오차막대(error bar), 신뢰구간, 표준편차가 없어 결과가 확실한 것처럼 보임. "
                "예측·추정 데이터를 실측처럼 표시.",
        "detection": "막대/선 차트에 오차막대가 있는지. "
                     "예측선이 실선으로 표시돼 실측과 구분 불가한지.",
        "example": "표본오차 ±5%인데 오차막대 없이 정확한 막대만 표시",
        "chart_types": ["막대차트", "선차트", "산점도"],
        "severity_weight": 0.65,
    },
    "sample_size_hidden": {
        "name": "표본 크기 은폐 (Hidden Sample Size)",
        "desc": "표본 n이 매우 작거나 통계적 유의성이 없음에도 숨기고 결과만 표시. "
                "n=3~5인 실험 결과를 대규모 연구처럼 제시.",
        "detection": "차트 근처에 표본 크기(n=?) 명시 여부. "
                     "오차막대가 매우 크면 표본 작을 가능성.",
        "example": "신약 효과 막대그래프 (n=4, 오차막대 없음, 출처 없음)",
        "chart_types": ["막대차트", "선차트", "산점도"],
        "severity_weight": 0.70,
    },
    "decorative_chartjunk": {
        "name": "과도한 장식 (Decorative Chartjunk)",
        "desc": "불필요한 아이콘·배경그림·격자선·그림자·그라디언트로 데이터 해석을 방해. "
                "시각적 소음이 데이터 신호보다 강해지는 상태.",
        "detection": "차트 요소 중 데이터 정보를 담지 않는 장식 요소의 비중. "
                     "배경이 단색 흰색/회색이 아닌 경우.",
        "example": "막대 배경에 빌딩 사진, 격자선 20개, 그라디언트 막대로 정확한 높이 파악 불가",
        "chart_types": ["인포그래픽", "막대차트", "파이차트"],
        "severity_weight": 0.45,
    },

    # ── H. 특수 ───────────────────────────────────────────
    "false_correlation": {
        "name": "허위 상관관계 (False Correlation / Spurious Causation)",
        "desc": "산점도나 이중 축 스케일 조작으로 인과관계 없는 두 변수를 상관관계처럼 표현. "
                "니콜라스 케이지 영화 수 vs 수영장 익사자 수.",
        "detection": "이중 축에서 두 선이 완벽 일치하는 경우. "
                     "산점도에서 관계 없는 두 변수의 상관계수가 높게 표시.",
        "example": "아이스크림 판매량 ↑ + 익사 사망 ↑ → 이중 축으로 완벽 일치 = 허위 인과",
        "chart_types": ["산점도", "선차트", "이중축차트"],
        "severity_weight": 0.75,
    },

    # ── I. 픽토그램 오류 (통합) ───────────────────────────
    "pictogram_distortion": {
        "name": "픽토그램 왜곡 (Pictogram / Icon Distortion)",
        "desc": "아이콘·그림으로 데이터를 표현할 때: "
                "① 크기와 개수를 동시에 키워 이중 왜곡 "
                "② 반지름에 비례시켜 면적 과장 "
                "③ 국기·사람 등 상징 이미지 비례 오류.",
        "detection": "아이콘 크기 변화와 개수 변화가 동시에 일어나는지. "
                     "아이콘 면적이 데이터에 정비례하는지.",
        "example": "인구 2배 → 사람 아이콘 크기 2배 + 개수 2배 → 시각적으로 4배 차이",
        "chart_types": ["픽토그램", "인포그래픽"],
        "severity_weight": 0.75,
    },

    # ── G. 기울기·주석 왜곡 ──────────────────────────────
    "slope_distortion": {
        "name": "기울기 왜곡 (Slope Distortion)",
        "desc": "선 차트에서 실제 데이터 변화량과 시각적 기울기가 불일치. "
                "더 작은 변화 구간이 더 큰 변화 구간보다 가파르게 그려져 "
                "추세 둔화를 가속처럼 보이게 하거나 그 반대로 표현.",
        "detection": "각 구간의 픽셀-값 비율(px/unit)이 일정한지 확인. "
                     "비율 편차가 25% 이상이면 왜곡으로 판정.",
        "example": "7→8월: 4.1pp 감소인데 시각적으로 완만, "
                   "8→9월: 3.5pp 감소인데 시각적으로 더 가파르게 표현 → 둔화를 가속으로 오도",
        "chart_types": ["선차트", "영역차트"],
        "severity_weight": 0.80,
    },
    "misleading_annotation": {
        "name": "주석 왜곡 (Misleading Annotation)",
        "desc": "차트에 추가된 화살표·괄호·강조선 등 주석이 실제 변화량을 "
                "왜곡하거나 더 작은 변화를 더 극적으로 강조하여 오해 유발.",
        "detection": "주석에 표시된 숫자와 실제 변화량 불일치, "
                     "또는 작은 변화에 더 강한 시각적 강조가 붙어 있는 경우.",
        "example": "4.1 감소보다 3.5 감소에 더 굵은 빨간 화살표를 달아 "
                   "감소폭이 커지는 것처럼 오도",
        "chart_types": ["선차트", "막대차트", "영역차트"],
        "severity_weight": 0.75,
    },
}

MISLEADER_KEYS: set[str] = set(MISLEADER_TAXONOMY.keys())

# ─────────────────────────────────────────────────────────
# MisViz 데이터셋 라벨 → 내부 키 매핑
# ─────────────────────────────────────────────────────────
MISVIZ_LABEL_TO_KEY: dict[str, str] = {
    # MisViz 공식 라벨
    "misrepresentation":               "misrepresentation",
    "3d":                              "3d_effect",
    "3d_effect":                       "3d_effect",
    "truncated_axis":                  "truncated_axis",
    "truncated_y_axis":                "truncated_axis",
    "inconsistent_tick_intervals":     "inconsistent_tick",
    "inconsistent_tick":               "inconsistent_tick",
    "dual_axis":                       "dual_axis",
    "inconsistent_binning_size":       "inconsistent_binning",
    "inconsistent_binning":            "inconsistent_binning",
    "discretized_continuous_variable": "discretized_continuous",
    "discretized_continuous":          "discretized_continuous",
    "inappropriate_use_of_line_chart": "inappropriate_line",
    "inappropriate_line":              "inappropriate_line",
    "inappropriate_item_order":        "inappropriate_order",
    "inappropriate_order":             "inappropriate_order",
    "inverted_axis":                   "inverted_axis",
    "inappropriate_axis_range":        "inappropriate_range",
    "inappropriate_range":             "inappropriate_range",
    "distorted_projection":            "distorted_projection",
    "data_visual_disproportion":       "data_visual_disproportion",
    "data-visual_disproportion":       "data_visual_disproportion",
    "manipulated_annotation":          "manipulated_annotation",
    "manipulated_annotations":         "manipulated_annotation",
    "none":                            "",
    "no_misleader":                    "",
    # 구 키 → 새 키 호환성 유지
    "visual_saliency_hacking":         "selective_emphasis",
    "annotation_bias":                 "selective_emphasis",
    "selective_highlighting":          "selective_emphasis",
    "foreground_emphasis":             "selective_emphasis",
    "perspective_exaggeration":        "distorted_projection",
    "bar_width_distortion":            "data_visual_disproportion",
    "icon_quantity_conflict":          "pictogram_distortion",
    "symbolic_area_mismatch":          "pictogram_distortion",
    "uneven_spacing":                  "inconsistent_tick",
    "axis_obfuscation":                "missing_label",
    "cropped_visual_context":          "cropped_visual_context",
    "shape_encoding_distortion":       "area_distortion",
    "stacked_chart_misuse":            "stacked_chart_misuse",
}

# ─────────────────────────────────────────────────────────
# 차트 유형 정규화 매핑
# ─────────────────────────────────────────────────────────
CHART_TYPE_ALIASES: dict[str, str] = {
    "꺾은선": "선차트", "꺾은선차트": "선차트", "line chart": "선차트",
    "line": "선차트", "line graph": "선차트",
    "bar chart": "막대차트", "bar graph": "막대차트", "bar": "막대차트",
    "pie chart": "파이차트", "pie": "파이차트",
    "donut chart": "도넛차트", "doughnut": "도넛차트", "donut": "도넛차트",
    "scatter plot": "산점도", "scatter": "산점도", "scatterplot": "산점도",
    "bubble chart": "버블차트", "bubble": "버블차트",
    "area chart": "영역차트", "area": "영역차트",
    "histogram": "히스토그램",
    "heatmap": "히트맵", "heat map": "히트맵",
    "choropleth": "코로플레스맵", "choropleth map": "코로플레스맵",
    "pictogram": "픽토그램",
    "infographic": "인포그래픽",
    "dual axis": "이중축차트", "dual-axis": "이중축차트",
    "mixed chart": "혼합차트", "combo chart": "혼합차트",
    "stacked bar": "누적막대차트", "stacked bar chart": "누적막대차트",
}

# ─────────────────────────────────────────────────────────
# Few-Shot 예시 (에이전트 판단 보조)
# 각 예시: 차트 설명 → 분석 논리 → 결론
# ─────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = """
=== Few-Shot 판단 예시 (12가지) ===

【예시 1 — 축 절단】
차트: 막대그래프, Y축 83에서 시작. GTX980=85, R9 290X=84.
분석: Y축이 83 시작 → truncated_axis. 실제 차이 1/85=1.2%인데 시각 비율은 2/2=100%로 2.5배 과장.
결론: ["truncated_axis"]

【예시 2 — 불규칙 눈금 + 체리피킹】
차트: 선그래프, X축: 1990, 2000, 2005, 2006, 2010. 동일 시각 간격.
분석: 간격 10년/5년/1년/4년이 같은 픽셀 → inconsistent_tick.
     전체 데이터 중 특정 구간만 → cherry_picking.
결론: ["inconsistent_tick", "cherry_picking"]

【예시 3 — 이중 축 + 허위 상관관계】
차트: 이중 Y축. 왼쪽: 아이스크림 판매량(0-1000). 오른쪽: 익사 사망(0-100). 두 선이 완벽 일치.
분석: 스케일 조작으로 두 선 교차 → dual_axis.
     계절적 상관관계를 인과관계처럼 연출 → false_correlation.
결론: ["dual_axis", "false_correlation"]

【예시 4 — 역전 축 (Florida 총기 사례)】
차트: 막대그래프. Y축 위=0, 아래로 갈수록 증가(0→1000). 2005년 이후 막대가 아래로 길어짐.
분석: Y축이 아래로 갈수록 커짐 → inverted_axis. 총기법 시행 후 사망 급증이 감소처럼 보임.
결론: ["inverted_axis"]

【예시 6 — 면적 왜곡 + 3D효과】
차트: 3D 버블차트. A=10, B=20. B 원의 반지름이 A의 2배.
분석: 반지름 비례 시 면적 4배 차이 → area_distortion. 3D 원근으로 추가 왜곡 → 3d_effect.
결론: ["area_distortion", "3d_effect"]

【예시 7 — 로그 스케일 미표시】
차트: 선그래프. Y축: 1, 10, 100, 1000이 균등 간격. 레이블에 'log' 없음.
분석: 균등 간격인데 값이 10배씩 → 로그 스케일. 표시 없음 → log_scale_unlabeled.
결론: ["log_scale_unlabeled"]

【예시 8 — 선택적 강조 + 조작된 주석】
차트: 막대 12개. 11개는 회색, 1개만 빨간색. 빨간 막대에만 수치 레이블. 레이블 값이 실제보다 10% 높음.
분석: 특정 막대만 색상+레이블 강조 → selective_emphasis.
     레이블 수치가 막대 높이와 불일치 → manipulated_annotation.
결론: ["selective_emphasis", "manipulated_annotation"]

【예시 9 — 누적 차트 오용】
차트: 누적 막대. A국(파란10+빨강30), B국(파란40+빨강10). 빨간 부분 비교 원함.
분석: 빨간 부분 시작점이 A=10, B=40으로 달라 직접 높이 비교 불가 → stacked_chart_misuse.
결론: ["stacked_chart_misuse"]

【예시 10 — 불균등 구간 히스토그램】
차트: 히스토그램. 구간: 0-10, 10-20, 20-50, 50-100이 동일 너비로 표시.
분석: 구간 너비 10/10/30/50이 같은 픽셀 → inconsistent_binning. 50-100 구간이 과소 표현.
결론: ["inconsistent_binning"]

【예시 11 — 종횡비 조작 + 불확실성 은폐】
차트: 선그래프, 차트 세로:가로=5:1. 월별 0.3~0.5% 변동이 수직 급등처럼 보임. 오차막대 없음.
분석: 세로 과장 → aspect_ratio_manipulation. 오차막대 없어 추이가 확실한 것처럼 → hidden_uncertainty.
결론: ["aspect_ratio_manipulation", "hidden_uncertainty"]

【예시 12 — 정상 차트】
차트: 월별 매출 막대그래프. Y축 0 시작. 1~12월 순서. 균등 눈금. 단위(억원), 출처, 기준연도 표시.
분석: 모든 원칙 준수. 절단 없음, 순서 정상, 단위 명확.
결론: []
"""

# ─────────────────────────────────────────────────────────
# 내장 MisViz 예시 (벡터DB 다운로드 실패 시 폴백)
# ─────────────────────────────────────────────────────────
MISVIZ_EXAMPLES = [
    {"id": "mv001", "misleaders": ["misrepresentation"],
     "description": "Bar chart where the visual heights do not match the labeled values.",
     "chart_type": "bar chart"},
    {"id": "mv002", "misleaders": ["truncated_axis"],
     "description": "Bar chart with Y-axis starting at 62 instead of 0. Difference of 1% looks like 5x.",
     "chart_type": "bar chart"},
    {"id": "mv003", "misleaders": ["3d_effect"],
     "description": "3D pie chart where front slices appear larger due to perspective.",
     "chart_type": "pie chart"},
    {"id": "mv005", "misleaders": ["dual_axis"],
     "description": "Line chart with two Y-axes scaled to manufacture apparent correlation between unrelated variables.",
     "chart_type": "line chart"},
    {"id": "mv006", "misleaders": ["inconsistent_tick"],
     "description": "Line chart where X-axis shows 1990, 2000, 2005, 2006 at equal visual spacing.",
     "chart_type": "line chart"},
    {"id": "mv007", "misleaders": ["inverted_axis"],
     "description": "Bar chart with Y-axis going from high at bottom to low at top. Florida gun deaths chart.",
     "chart_type": "bar chart"},
    {"id": "mv008", "misleaders": ["inappropriate_order"],
     "description": "Bar chart showing months out of chronological order to hide the declining trend.",
     "chart_type": "bar chart"},
    {"id": "mv009", "misleaders": ["cherry_picking"],
     "description": "Line chart showing global temperature only from 1998-2012 to suggest a pause.",
     "chart_type": "line chart"},
    {"id": "mv010", "misleaders": ["log_scale_unlabeled"],
     "description": "Chart with logarithmic Y-axis (1, 10, 100, 1000 at equal intervals) without any log label.",
     "chart_type": "line chart"},
    {"id": "mv011", "misleaders": ["area_distortion"],
     "description": "Bubble chart where bubble radius (not area) is proportional to data value.",
     "chart_type": "bubble chart"},
    {"id": "mv012", "misleaders": ["missing_label"],
     "description": "Bar chart without axis labels, units, or data source.",
     "chart_type": "bar chart"},
    {"id": "mv013", "misleaders": ["false_correlation"],
     "description": "Dual-axis line chart: Nicolas Cage movies vs swimming pool drownings, both rising together.",
     "chart_type": "line chart"},
    {"id": "mv014", "misleaders": ["truncated_axis", "data_visual_disproportion"],
     "description": "Bar chart Y-axis starts at 83: 1.2% actual difference appears as 2.5x visual ratio.",
     "chart_type": "bar chart"},
    {"id": "mv015", "misleaders": ["misrepresentation"],
     "description": "Pie chart: visual angles don't match the percentages.",
     "chart_type": "pie chart"},
    {"id": "mv016", "misleaders": ["inconsistent_binning"],
     "description": "Histogram with bin widths 10, 10, 30, 50 shown at equal visual width.",
     "chart_type": "histogram"},
    {"id": "mv017", "misleaders": ["misleading_color"],
     "description": "Heatmap using red for small values and green for large values (opposite convention).",
     "chart_type": "heatmap"},
    {"id": "mv018", "misleaders": ["pictogram_distortion"],
     "description": "Pictogram where person icons are both larger AND more numerous for the larger value.",
     "chart_type": "pictogram"},
    {"id": "mv019", "misleaders": ["sample_size_hidden"],
     "description": "Bar chart showing average results without sample size n=5, no error bars.",
     "chart_type": "bar chart"},
    {"id": "mv020", "misleaders": ["inappropriate_range"],
     "description": "Line chart with Y-axis 0-1000 for data varying between 10-12, flattening all variation.",
     "chart_type": "line chart"},
    {"id": "mv021", "misleaders": ["selective_emphasis"],
     "description": "Bar chart where 11 bars are grey, but one specific bar is red with a label annotation.",
     "chart_type": "bar chart"},
    {"id": "mv022", "misleaders": ["hidden_uncertainty"],
     "description": "Bar chart comparing treatment vs control with no error bars, n=8 per group.",
     "chart_type": "bar chart"},
    {"id": "mv023", "misleaders": ["stacked_chart_misuse"],
     "description": "Stacked bar chart comparing only the top segments, which have different baselines.",
     "chart_type": "stacked bar chart"},
    {"id": "mv024", "misleaders": ["aspect_ratio_manipulation"],
     "description": "Line chart showing 0.1% monthly variation in a very tall, narrow chart making it look dramatic.",
     "chart_type": "line chart"},
    {"id": "mv025", "misleaders": ["non_aligned_baseline"],
     "description": "Bar chart where some bars start at y=0 and others float, starting at y=20.",
     "chart_type": "bar chart"},
    {"id": "mv026", "misleaders": ["cropped_visual_context"],
     "description": "Bar chart where bars exceeding 100 are clipped at the top by a Y-axis ending at 80.",
     "chart_type": "bar chart"},
    {"id": "mv027", "misleaders": ["distorted_projection"],
     "description": "3D pie chart tilted at 45 degrees; front slice visually 2x larger than back slice of same value.",
     "chart_type": "pie chart"},
    {"id": "mv028", "misleaders": ["manipulated_annotation"],
     "description": "Bar chart where A=70 and B=80 but the labels above bars show A=80 and B=70.",
     "chart_type": "bar chart"},
    {"id": "mv029", "misleaders": ["inappropriate_line"],
     "description": "Line chart connecting discrete country-level GDP values implying a meaningful trend.",
     "chart_type": "line chart"},
    {"id": "mv030", "misleaders": ["decorative_chartjunk"],
     "description": "Bar chart with building photo background, 20 gridlines, and gradient bars obscuring values.",
     "chart_type": "bar chart"},
]

# 프롬프트용 분류체계 요약
_TAXONOMY_BRIEF = "\n".join(
    f"  {k}: {v['name']} → {v['desc'][:70]}"
    for k, v in MISLEADER_TAXONOMY.items()
)


# ─────────────────────────────────────────────────────────
# 정규화 함수
# ─────────────────────────────────────────────────────────

def normalize_chart_type(raw: str) -> str:
    key = raw.strip().lower()
    return CHART_TYPE_ALIASES.get(key, raw)


def normalize_misleader_label(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    norm = re.sub(r"[\s\-]+", "_", raw.strip().lower())
    norm = re.sub(r"_+", "_", norm).strip("_")
    if norm in MISVIZ_LABEL_TO_KEY:
        return MISVIZ_LABEL_TO_KEY[norm]
    if norm in MISLEADER_KEYS:
        return norm
    return norm
