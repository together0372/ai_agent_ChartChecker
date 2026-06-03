"""
DistortionDetectorAgent — LangGraph 기반 차트 오류 감지 멀티에이전트

그래프:
  [Observer 3단계]
  observer_analyze (1단계: 분석, 도구없음)
  → observer_mandatory_tools (2단계: 차트유형별 필수도구 자동실행)
  → observer_llm ⇄ observer_tools (3단계: 가설기반 ReAct)
  → observer_finalize
  → math (비전 추출 + LLM 루프 + 직접 수학도구)
  → self_critique → [recheck] → reporter → confidence → END

핵심 설계:
  - Observer 3단계: 분석→필수도구→가설검증 ReAct
  - 1단계: 도구없이 차트 메시지·의심요소 파악, "속이는 사람" 관점 가설 도출
  - 2단계: 차트 유형별 필수 도구 LLM 판단 없이 자동 실행
  - 3단계: 가설 기반 추가 검증, 도구 결과마다 해석 텍스트 출력
  - use_rag: RAG 선택적 사용
"""
from __future__ import annotations

import asyncio
import json
import operator
import textwrap
import time
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from .config import (
    LLM_NAME,
    MISLEADER_KEYS,
    MISLEADER_TAXONOMY,
    normalize_chart_type,
)
from .lc_tools import ALL_TOOLS, OBSERVER_TOOLS_BY_NAME
from .llm_utils import (
    call_text_llm,
    call_vision_llm,
    image_to_base64,
    log_conversation,
    parse_json,
)
from .math_tools import MathTools
from .rag_utils import format_rag_hint, query_vector_db


# ─────────────────────────────────────────────────────────
# 타이밍 헬퍼
# ─────────────────────────────────────────────────────────

async def _timed(label: str, coro) -> Any:
    """LLM 호출 코루틴을 래핑해 소요 시간을 출력."""
    t0 = time.perf_counter()
    result = await coro
    elapsed = time.perf_counter() - t0
    print(f"     ⏱️  {label}: {elapsed:.1f}s")
    return result


# ─────────────────────────────────────────────────────────
# 상태 정의
# ─────────────────────────────────────────────────────────
class AgentState(TypedDict):
    image_path:   str
    verbose:      bool
    conversation: list[dict]
    use_rag:      bool

    # Observer ReAct 메시지 (누적 reducer)
    observer_messages: Annotated[list[BaseMessage], operator.add]

    chart_type:           str
    image_description:    str
    initial_observations: str
    suspected_misleaders: list[str]
    tool_evidence:        dict

    observer_tool_call_count: int
    self_critique_removed:    list[str]

    # Observer 3단계 분리용
    observer_analysis:   str        # 1단계: 차트 분석 + 가설 텍스트
    observer_hypotheses: list[str]  # 1단계: 의심 misleader 키 목록
    mandatory_tool_results: dict    # 2단계: 필수 도구 실행 결과

    web_context:       dict
    extracted_numbers: dict
    math_checks:       dict
    similar_examples:  list[dict]
    rag_hint:          str

    final_misleaders: list[str]
    confidence:       dict
    self_confidence:  dict
    explanation:      str
    verdict:          str

    iteration:      int
    max_iterations: int
    needs_recheck:  bool
    recheck_reason: str
    early_exit:     bool


# ─────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────
MAX_OBSERVER_ROUNDS = 3   # Observer ReAct 최대 도구 호출

# 차트 유형별 필수 도구 (2단계에서 자동 실행)
# 값: [(도구명, 인자 추출 함수용 키)]
MANDATORY_TOOLS_BY_TYPE: dict[str, list[str]] = {
    # 영어 키
    "bar":               ["tool_check_axis_truncation", "tool_check_label_value_match", "tool_check_baseline_alignment",
                          "tool_check_cherry_pick_range", "tool_check_unit_switch", "tool_check_cumulative_vs_period"],
    "bidirectional_bar": ["tool_check_axis_truncation", "tool_check_bar_scale_symmetry"],
    "pie":               ["tool_check_multi_pie_sum", "tool_check_pie_angles"],
    "donut":             ["tool_check_multi_pie_sum", "tool_check_selective_annotation"],
    "line":              ["tool_check_axis_truncation", "tool_check_tick_intervals",
                          "tool_check_cherry_pick_range", "tool_check_unit_switch", "tool_check_cumulative_vs_period"],
    "area":              ["tool_check_axis_truncation", "tool_check_tick_intervals",
                          "tool_check_cherry_pick_range", "tool_check_cumulative_vs_period"],
    "histogram":         ["tool_check_axis_truncation", "tool_check_bin_widths"],
    "scatter":           ["tool_check_area_distortion"],
    "bubble":            ["tool_check_area_distortion"],
    # 한국어 키 (normalize_chart_type 변환 결과)
    "막대차트":           ["tool_check_axis_truncation", "tool_check_label_value_match", "tool_check_baseline_alignment",
                          "tool_check_cherry_pick_range", "tool_check_unit_switch", "tool_check_cumulative_vs_period"],
    "양방향막대차트":      ["tool_check_axis_truncation", "tool_check_bar_scale_symmetry"],
    "파이차트":           ["tool_check_multi_pie_sum", "tool_check_pie_angles"],
    "도넛차트":           ["tool_check_multi_pie_sum", "tool_check_selective_annotation"],
    "선차트":             ["tool_check_axis_truncation", "tool_check_tick_intervals",
                          "tool_check_cherry_pick_range", "tool_check_unit_switch", "tool_check_cumulative_vs_period"],
    "영역차트":           ["tool_check_axis_truncation", "tool_check_tick_intervals",
                          "tool_check_cherry_pick_range", "tool_check_cumulative_vs_period"],
    "히스토그램":         ["tool_check_axis_truncation", "tool_check_bin_widths"],
    "산점도":             ["tool_check_area_distortion"],
    "버블차트":           ["tool_check_area_distortion"],
}


# ─────────────────────────────────────────────────────────
# 에이전트 클래스
# ─────────────────────────────────────────────────────────
class DistortionDetectorAgent:

    def __init__(self, use_rag: bool = True) -> None:
        self._use_rag = use_rag
        self._graph   = self._build_graph()

    async def analyze(self, image_path: str, verbose: bool = False) -> AgentState:
        initial: AgentState = {
            "image_path": image_path, "verbose": verbose, "conversation": [],
            "use_rag": self._use_rag,
            "observer_messages": [],
            "chart_type": "", "image_description": "", "initial_observations": "",
            "suspected_misleaders": [], "tool_evidence": {},
            "observer_tool_call_count": 0, "self_critique_removed": [],
            "observer_analysis": "", "observer_hypotheses": [], "mandatory_tool_results": {},
            "web_context": {}, "extracted_numbers": {}, "math_checks": {},
            "similar_examples": [], "rag_hint": "",
            "final_misleaders": [], "confidence": {}, "self_confidence": {},
            "explanation": "", "verdict": "",
            "iteration": 0, "max_iterations": 2,
            "needs_recheck": False, "recheck_reason": "",
            "early_exit": False,
        }
        return await self._graph.ainvoke(initial)

    # ─────────────────────────────────────────────────────
    # 그래프 빌드
    # ─────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(AgentState)

        # Observer 3단계
        g.add_node("observer_analyze",         self._node_observer_analyze)         # 1단계: 분석
        g.add_node("observer_mandatory_tools", self._node_observer_mandatory_tools) # 2단계: 필수도구
        g.add_node("observer_llm",             self._node_observer_llm)             # 3단계: ReAct LLM
        g.add_node("observer_tools",           self._node_observer_tools)           # 3단계: ReAct 도구실행
        g.add_node("observer_finalize",        self._node_observer_finalize)        # 결과 추출

        # Math (extracted_numbers 확보)
        g.add_node("math",          self._node_math_verifier)

        # 하위 파이프라인
        g.add_node("self_critique", self._node_self_critique)
        g.add_node("recheck",       self._node_recheck)
        g.add_node("reporter",      self._node_reporter)
        g.add_node("confidence",    self._node_confidence_scorer)

        # Observer 흐름 (3단계)
        g.add_edge(START,                      "observer_analyze")
        g.add_edge("observer_analyze",         "observer_mandatory_tools")
        g.add_edge("observer_mandatory_tools", "observer_llm")
        g.add_conditional_edges("observer_llm", self._observer_route,
                                {"tools": "observer_tools", "done": "observer_finalize"})
        g.add_edge("observer_tools",    "observer_llm")
        g.add_edge("observer_finalize", "math")

        # Math → Self-critique (Debate 제거)
        g.add_edge("math", "self_critique")

        # Self-critique → (recheck or reporter)
        g.add_conditional_edges("self_critique", self._route_self_critique,
                                {"recheck": "recheck", "reporter": "reporter"})
        g.add_edge("recheck",    "self_critique")
        g.add_edge("reporter",   "confidence")
        g.add_edge("confidence", END)

        return g.compile()

    # ─────────────────────────────────────────────────────
    # 라우팅
    # ─────────────────────────────────────────────────────

    def _observer_route(self, state: AgentState) -> Literal["tools", "done"]:
        msgs = state.get("observer_messages", [])
        if not msgs:
            return "done"
        last = msgs[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            used = sum(1 for m in msgs if isinstance(m, ToolMessage))
            if used >= MAX_OBSERVER_ROUNDS:
                return "done"
            return "tools"
        return "done"

    def _route_self_critique(self, state: AgentState) -> Literal["recheck", "reporter"]:
        if (state.get("needs_recheck")
                and state.get("iteration", 0) < state.get("max_iterations", 2)
                and not state.get("early_exit")):
            return "recheck"
        return "reporter"

    def _get_available_tools(self) -> list:
        if self._use_rag:
            return ALL_TOOLS
        return [t for t in ALL_TOOLS
                if t.name not in ("tool_query_misviz_db", "tool_search_misleader_evidence")]

    # ─────────────────────────────────────────────────────
    # Observer — 1단계: 분석 (도구 없음)
    # ─────────────────────────────────────────────────────

    async def _node_observer_analyze(self, state: AgentState) -> AgentState:
        """
        도구를 전혀 사용하지 않고 차트를 읽어 분석.
        - 차트가 전달하려는 메시지 파악
        - 시각적 이상 요소 관찰
        - "내가 속이는 사람이라면?" 관점으로 가설 도출
        결과를 observer_analysis / observer_hypotheses 에 저장.
        """
        print("\n  🔍 관찰자 에이전트 — [1단계] 차트 분석 중...")

        system = textwrap.dedent(f"""
You are a chart fraud investigator. Your job is to ANALYZE the chart — do NOT call any tools yet.

## Your task:
1. READ the chart carefully: every number, label, axis, color, annotation
2. UNDERSTAND the message: what is this chart trying to make readers believe?
3. THINK like a deceiver: if YOU made this chart to mislead, what tricks would you use?
4. LIST your hypotheses: which specific deception techniques might be present?

## Output ONLY this JSON:
{{
  "chart_type": "bar|line|pie|donut|scatter|bubble|histogram|bidirectional_bar|area|other",
  "chart_message": "one sentence: what conclusion does this chart push readers toward?",
  "all_numbers": "every number visible: axis values, labels, percentages, annotations",
  "visual_anomalies": ["anomaly 1", "anomaly 2"],
  "deception_hypotheses": [
    {{"misleader_key": "truncated_axis", "reason": "Y-axis starts at 60, not 0 — gap looks huge"}},
    {{"misleader_key": "selective_emphasis", "reason": "only one bar is red while others are gray"}}
  ]
}}

## MISLEADER KEYS to choose from:
{', '.join(sorted(MISLEADER_KEYS))}
""").strip()

        raw = await _timed("Observer 1단계 Vision LLM", asyncio.to_thread(
            call_vision_llm,
            "Analyze this chart for potential deception. What is it trying to make you believe? What tricks might be used?",
            state["image_path"],
            system,
            1200,
            0.1,
            False,   # use_thinking=False — JSON 출력 보장
        ))

        parsed = parse_json(raw)
        if parsed:
            chart_type  = normalize_chart_type(parsed.get("chart_type", ""))
            hypotheses  = [
                h["misleader_key"]
                for h in parsed.get("deception_hypotheses", [])
                if h.get("misleader_key") in MISLEADER_KEYS
            ]
            print(f"     차트 유형: {chart_type}")
            print(f"     메시지: {parsed.get('chart_message', '')[:80]}")
            print(f"     이상 요소: {parsed.get('visual_anomalies', [])}")
            print(f"     의심 가설: {hypotheses}")
        else:
            chart_type = ""
            hypotheses = []
            print(f"     ⚠️ 분석 JSON 파싱 실패 → 원문 사용")
            print(f"     [DEBUG 원문 처음 300자]\n{raw[:300]}\n     [/DEBUG]")

        # ── RAG 조기 조회 (가설 기반 쿼리 → 3단계 도구 선택에 활용) ──────
        similar: list[dict] = []
        rag_hint_text = ""
        if state.get("use_rag", True) and (chart_type or hypotheses):
            try:
                # chart_type + 가설 키워드를 합쳐 정확한 쿼리 생성
                rag_query = " ".join(filter(None, [chart_type] + hypotheses))
                similar = await asyncio.to_thread(query_vector_db, rag_query, 3)
                rag_hint_text = await asyncio.to_thread(format_rag_hint, rag_query, 8)
                if similar:
                    print(f"     📚 RAG 유사사례 {len(similar)}건 (가설 기반 조기 로드)")
            except Exception as e:
                print(f"     ⚠️ RAG 건너뜀: {e}")

        return {
            **state,
            "chart_type":          chart_type,
            "observer_analysis":   raw,
            "observer_hypotheses": hypotheses,
            "similar_examples":    similar,
            "rag_hint":            rag_hint_text,
        }

    # ─────────────────────────────────────────────────────
    # Observer — 2단계: 필수 도구 자동 실행
    # ─────────────────────────────────────────────────────

    async def _node_observer_mandatory_tools(self, state: AgentState) -> AgentState:
        """
        차트 유형에 따라 반드시 실행해야 할 도구를 LLM 판단 없이 자동 실행.
        수치는 math_verifier가 추출한 extracted_numbers를 활용하되,
        아직 math 단계 전이므로 이미지에서 미리 숫자를 추출해서 사용.
        결과는 mandatory_tool_results 와 tool_evidence에 저장.
        """
        chart_type = state.get("chart_type", "")
        norm_type  = normalize_chart_type(chart_type)
        tools_to_run = MANDATORY_TOOLS_BY_TYPE.get(norm_type, [])

        print(f"\n  🔧 관찰자 에이전트 — [2단계] 필수 도구 실행 ({norm_type}: {len(tools_to_run)}개)...")

        # 숫자 추출은 chart_type 인식 여부와 무관하게 항상 실행
        # — extracted_numbers를 math 노드가 재사용하므로 도구 실행 없어도 추출은 해야 함
        # math 노드는 이 값을 재사용하여 재추출을 생략함
        extract_sys = textwrap.dedent("""
Extract all numbers from this chart. Output ONLY JSON (null or [] if unreadable):
```json
{
  "y_min": null, "y_max": null, "y_ticks": [], "x_labels": [], "x_ticks": [],
  "data_values": [], "visual_ratios": [], "bar_start_values": [],
  "pie_pcts": [], "pie_angles_deg": [],
  "has_dual_axis": false, "r_min": null, "r_max": null, "r_ticks": [],
  "has_log_scale": false, "log_labeled": false,
  "visual_areas": [], "title": null, "x_label": null, "y_label": null,
  "has_unit": false, "chart_width_px": null, "chart_height_px": null,
  "annotated_indices": [], "highlighted_indices": [],
  "bin_edges": [], "visual_bin_widths_px": [],
  "is_bidirectional_bar": false,
  "left_bar_values": [], "left_bar_px": [],
  "right_bar_values": [], "right_bar_px": [],
  "visual_y_px": []
}
```
Notes:
- visual_ratios = bar heights as 0-1 fraction of Y-axis range.
- visual_areas = relative visual sizes between chart elements (NO pixel measurement needed).
  Estimate by eye: if bubble A looks twice as large as bubble B, use [2.0, 1.0].
  For pie slices: estimate area proportion relative to the first slice.
  Leave [] if the chart has no size-encoded elements (bars, lines, etc.).
- For bidirectional bar: left_bar_px/right_bar_px = estimated bar length 0.0-1.0.
- For donut: annotated_indices = indices of prominently highlighted slice numbers.
- For line charts: visual_y_px = each data point's Y pixel position from top (top=0).
  Estimate based on chart height. Example: chart height 300px, point at 1/3 from top → 100.
""").strip()

        nums: dict = {}
        try:
            raw_nums = await _timed("Observer 2단계 숫자추출 Vision LLM", asyncio.to_thread(
                call_vision_llm,
                f"Extract all numbers. Suspected: {state.get('observer_hypotheses', [])}",
                state["image_path"],
                extract_sys, 1200, 0.0, False,   # use_thinking=False
            ))
            parsed_nums = parse_json(raw_nums)
            if parsed_nums:
                nums = parsed_nums
                print(f"     📐 숫자 추출 완료 → extracted_numbers 저장 (math 노드 재사용)")
        except Exception as e:
            print(f"     ⚠️ 숫자 추출 실패: {e}")

        # 필수 도구가 없으면 숫자 추출 결과만 저장하고 종료
        if not tools_to_run:
            print(f"     차트 유형 '{norm_type}' 필수 도구 없음 → 숫자 추출만 저장")
            return {**state, "extracted_numbers": nums}

        def _get(key, default=None):
            v = nums.get(key)
            return v if v is not None else default

        # 도구별 인자 매핑
        def _build_args(tool_name: str) -> dict | None:
            if tool_name == "tool_check_axis_truncation":
                y_min = _get("y_min", 0)
                y_max = _get("y_max", 100)
                d_min = min(_get("data_values", [y_min or 0]) or [0])
                return {"y_min": y_min, "y_max": y_max, "data_min": d_min}

            if tool_name == "tool_check_label_value_match":
                vals = _get("data_values", [])
                rats = _get("visual_ratios", [])
                if not vals:
                    return None
                return {"label_values": vals, "visual_ratios": rats or [],
                        "y_min": _get("y_min", 0), "y_max": _get("y_max", 100)}

            if tool_name == "tool_check_baseline_alignment":
                starts = _get("bar_start_values", [])
                return {"bar_start_values": starts} if starts else None

            if tool_name == "tool_check_bar_scale_symmetry":
                lv = _get("left_bar_values", [])
                lp = _get("left_bar_px", [])
                rv = _get("right_bar_values", [])
                rp = _get("right_bar_px", [])
                if not lv or not rv:
                    return None
                return {"left_values": lv, "left_px": lp,
                        "right_values": rv, "right_px": rp}

            if tool_name == "tool_check_pie_sum":
                pcts = _get("pie_pcts", [])
                return {"percentages": pcts} if pcts else None

            if tool_name == "tool_check_multi_pie_sum":
                pcts = _get("pie_pcts", [])
                return {"all_pcts": pcts, "chart_count": 0} if pcts else None

            if tool_name == "tool_check_pie_angles":
                pcts   = _get("pie_pcts", [])
                angles = _get("pie_angles_deg", [])
                if not pcts or not angles:
                    return None
                return {"pie_pcts": pcts, "pie_angles_deg": angles}

            if tool_name == "tool_check_selective_annotation":
                idx  = _get("annotated_indices", [])
                vals = _get("pie_pcts") or _get("data_values", [])
                if not vals:
                    return None
                return {"annotated_indices": idx, "all_values": vals}

            if tool_name == "tool_check_tick_intervals":
                ticks = _get("y_ticks", [])
                return {"ticks": ticks} if len(ticks) >= 3 else None

            if tool_name == "tool_check_slope_distortion":
                vals = _get("data_values", [])
                ypx  = _get("visual_y_px", [])
                # visual_y_px가 없으면 결과가 항상 정상으로 나오므로 건너뜀
                # (실제 픽셀 좌표는 LLM이 직접 추출해야 의미있는 검증 가능)
                if len(vals) < 3 or len(ypx) < 3:
                    return None
                return {"values": vals, "visual_y_px": ypx,
                        "x_labels": _get("x_labels", [])}

            if tool_name == "tool_check_area_distortion":
                vals  = _get("data_values", [])
                # visual_areas 우선, 없으면 visual_ratios 로 대체
                # (LLM은 절대 픽셀 면적보다 상대 비율을 훨씬 신뢰성 있게 추정함)
                areas = _get("visual_areas", []) or _get("visual_ratios", [])
                if not vals:
                    return None
                return {"values": vals, "visual_areas": areas or []}

            if tool_name == "tool_check_bin_widths":
                edges = _get("bin_edges", [])
                widths = _get("visual_bin_widths_px", [])
                if not edges:
                    return None
                return {"bin_edges": edges, "visual_widths_px": widths or []}

            if tool_name == "tool_check_cherry_pick_range":
                vals = _get("data_values", [])
                x_labels = _get("x_labels", [])
                if not vals or len(vals) < 3:
                    return None
                return {
                    "shown_values":      vals,
                    "shown_start_label": str(x_labels[0])  if x_labels else "",
                    "shown_end_label":   str(x_labels[-1]) if x_labels else "",
                }

            if tool_name == "tool_check_unit_switch":
                # x_labels를 unit_labels로 재활용 (단위 정보가 레이블에 포함될 수 있음)
                x_labels = _get("x_labels", [])
                vals = _get("data_values", [])
                if not vals:
                    return None
                return {"units": x_labels or [], "values": vals}

            if tool_name == "tool_check_cumulative_vs_period":
                vals = _get("data_values", [])
                x_labels = _get("x_labels", [])
                if not vals or len(vals) < 3:
                    return None
                return {"values": vals, "x_labels": x_labels or [], "declared_type": "period"}

            return None

        # 필수 도구 실행
        mandatory_results: dict = {}
        evidence = dict(state.get("tool_evidence", {}))

        for tool_name in tools_to_run:
            args = _build_args(tool_name)
            if args is None:
                print(f"     ⏭️  {tool_name}: 인자 부족 → 건너뜀")
                continue
            if tool_name not in OBSERVER_TOOLS_BY_NAME:
                continue
            try:
                result = await asyncio.to_thread(
                    OBSERVER_TOOLS_BY_NAME[tool_name].invoke, args
                )
                if not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False)
                mandatory_results[tool_name] = result
                evidence[tool_name]          = result[:400]
                print(f"     ✅ [필수] {tool_name}: {result[:80]}...")
            except Exception as e:
                # 오류 도구는 mandatory_results와 evidence 모두 제외
                # — 오류 문자열을 LLM이 "도구 결과"로 오해하는 것을 방지
                print(f"     ❌ [필수] {tool_name}: {e}")

        return {
            **state,
            "mandatory_tool_results":   mandatory_results,
            "tool_evidence":            evidence,
            "extracted_numbers":        nums,   # math 노드에서 재사용 (재추출 생략)
            "observer_tool_call_count": state.get("observer_tool_call_count", 0) + len(mandatory_results),
        }

    # ─────────────────────────────────────────────────────
    # Observer — LLM 호출
    # ─────────────────────────────────────────────────────

    async def _node_observer_llm(self, state: AgentState) -> AgentState:
        msgs = state.get("observer_messages", [])
        used = sum(1 for m in msgs if isinstance(m, ToolMessage))
        print(f"     [Observer LLM 3단계] 메시지:{len(msgs)} / 추가도구:{used}회")

        # 첫 진입 — 시스템 프롬프트 + 컨텍스트 구성
        if not msgs:
            analysis      = state.get("observer_analysis", "")
            hypotheses    = state.get("observer_hypotheses", [])
            mandatory_res = state.get("mandatory_tool_results", {})
            img_b64       = await asyncio.to_thread(image_to_base64, state["image_path"])

            # 필수 도구 결과 요약 텍스트
            if mandatory_res:
                mand_lines = "\n".join(
                    f"  - {k}: {v[:120]}" for k, v in mandatory_res.items()
                )
                mandatory_section = f"## Mandatory tool results (already run):\n{mand_lines}"
            else:
                mandatory_section = "## Mandatory tools: none ran (chart type unknown)"

            # RAG 힌트
            rag_hint  = state.get("rag_hint", "")
            rag_section = (
                f"\n## Similar deceptive charts from database:\n{rag_hint[:800]}\n"
                if rag_hint else ""
            )

            system = textwrap.dedent(f"""
You are a chart fraud investigator. Phase 1 analysis and mandatory tool checks are already done.
Your job now: verify the hypotheses with targeted tools, then summarize findings.

{mandatory_section}
{rag_section}
## Hypotheses from Phase 1 analysis (need verification):
{json.dumps(hypotheses, ensure_ascii=False)}

## Your workflow — follow this EXACTLY:
For each remaining hypothesis (not yet confirmed or ruled out by mandatory tools):
  1. Write: "[Checking] <hypothesis> — because <specific visual evidence>"
  2. Call ONE relevant tool with precise values read from the chart
  3. After tool result: Write "[Found] <what the result means>" OR "[Clear] no issue found"
  4. If tool result reveals a NEW suspicion → add it and check it too

After all hypotheses are checked, output ONLY this JSON and nothing else:
{{"done": true}}

## MISLEADER KEYS: {', '.join(sorted(MISLEADER_KEYS))}

RULES:
- Only call tools for hypotheses NOT already covered by mandatory tools above.
- Write interpretation text after EVERY tool result.
- When all hypotheses are verified (or no more tools are needed), output {{"done": true}}.
- Do NOT write a long summary or analysis after finishing — just {{"done": true}}.
""").strip()

            human_content: list[Any] = [
                {"type": "text", "text": (
                    f"Phase 1 analysis:\n{analysis[:600]}\n\n"
                    "Now verify remaining hypotheses with targeted tool calls. "
                    "For each tool call, explain WHY you are calling it and interpret the result."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]
            msgs = [SystemMessage(content=system), HumanMessage(content=human_content)]

        # 루프 중 — 독려 메시지 갱신 + 이미지 제거 (이전 독려 메시지 제거 후 새 것으로 교체)
        elif used > 0:
            already_called = list(state.get("tool_evidence", {}).keys())
            already_str = ", ".join(already_called) if already_called else "none"
            msgs = list(msgs)

            # ── 이미지 토큰 제거: 루프 2+는 도구 결과만 보면 됨 ──
            # HumanMessage 중 image_url 블록이 있는 경우 text 부분만 남김
            stripped: list[BaseMessage] = []
            for m in msgs:
                if (isinstance(m, HumanMessage)
                        and isinstance(m.content, list)
                        and any(b.get("type") == "image_url" for b in m.content if isinstance(b, dict))):
                    text_parts = " ".join(
                        b.get("text", "") for b in m.content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    stripped.append(HumanMessage(content=text_parts or "[chart image — already analyzed]"))
                else:
                    stripped.append(m)
            msgs = stripped

            # 직전에 추가된 독려 메시지가 있으면 교체 (중복 누적 방지)
            if (msgs and isinstance(msgs[-1], HumanMessage)
                    and isinstance(msgs[-1].content, str)
                    and msgs[-1].content.startswith("Good.")):
                msgs = msgs[:-1]
            msgs = msgs + [HumanMessage(content=(
                f"Good. {used} tool(s) called so far.\n"
                f"Already called (do NOT call these again): {already_str}\n"
                'If all hypotheses are verified, output {"done": true} to finish. '
                "If there are remaining unchecked hypotheses, call ONE new tool (not from the list above). "
                "Remember: write [Checking] before calling a tool, [Found]/[Clear] after result."
            ))]

        # ── 루프 단계에 따라 num_predict 조정 ──
        # 루프#1(첫 진입): 이미지 분석 + 가설 텍스트 생성 → 넉넉하게
        # 루프#2+: 도구 호출 JSON or {"done":true} 신호만 생성 → 크게 줄임
        num_predict = 4000 if used == 0 else 800

        # ── 이미 호출한 도구는 스키마에서 제외 (컨텍스트 절약) ──
        already_called_set = set(state.get("tool_evidence", {}).keys())
        available = [t for t in self._get_available_tools() if t.name not in already_called_set]
        if not available:
            available = self._get_available_tools()  # 모두 호출됐으면 전체 유지

        llm = ChatOllama(model=LLM_NAME, num_predict=num_predict, temperature=0.1, reasoning=False)
        llm_with_tools = llm.bind_tools(available)

        _loop_label = f"Observer 3단계 ReAct LLM (루프#{used+1})"
        try:
            resp = await _timed(_loop_label, asyncio.to_thread(llm_with_tools.invoke, msgs))
        except Exception as e:
            print(f"     ❌ LLM 실패: {e}")
            resp = AIMessage(content="[LLM 호출 실패]")

        if getattr(resp, "tool_calls", None):
            print(f"     → 추가 도구: {[tc['name'] for tc in resp.tool_calls]}")
        else:
            preview = (resp.content or "")[:120].replace("\n", " ")
            print(f"     → 최종 답변: {preview}...")

        # 첫 진입이면 시스템+휴먼 메시지도 함께 반환 (reducer가 누적)
        if not state.get("observer_messages"):
            return {"observer_messages": msgs + [resp]}
        return {"observer_messages": [resp]}

    # ─────────────────────────────────────────────────────
    # Observer — 도구 실행
    # ─────────────────────────────────────────────────────

    async def _node_observer_tools(self, state: AgentState) -> AgentState:
        msgs = state.get("observer_messages", [])
        if not msgs or not isinstance(msgs[-1], AIMessage):
            return state

        last_ai   = msgs[-1]
        tool_msgs: list[ToolMessage] = []
        evidence  = dict(state.get("tool_evidence", {}))

        for tc in (last_ai.tool_calls or []):
            name, args, tid = tc["name"], tc["args"], tc["id"]
            if name not in OBSERVER_TOOLS_BY_NAME:
                result = f"[UNKNOWN TOOL] '{name}' does not exist. Do NOT call this tool again."
                # 존재하지 않는 도구도 evidence에 등록 → 다음 루프에서 already_called_set에 포함되어 재호출 차단
                if name not in evidence:
                    evidence[name] = result[:200]
                    print(f"       ⚠️  {name}: 존재하지 않는 도구 (재호출 차단 등록)")
            elif name in evidence:
                # 이미 실행된 도구 — 재실행 차단, 기존 결과 반환
                result = f"[DUPLICATE] {name} was already called. Previous result: {evidence[name][:200]}. Do NOT call this tool again."
                print(f"       ⛔ {name}: 중복 호출 차단 (기존 결과 재사용)")
            else:
                try:
                    result = await asyncio.to_thread(OBSERVER_TOOLS_BY_NAME[name].invoke, args)
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)
                    evidence[name] = result[:400]
                    print(f"       ✅ {name}: {result[:100]}...")
                except Exception as te:
                    result = f"오류: {te}"
                    print(f"       ❌ {name}: {te}")
            tool_msgs.append(ToolMessage(tool_call_id=tid, name=name, content=result))

        new_count = state.get("observer_tool_call_count", 0) + len(tool_msgs)
        return {
            "observer_messages":        tool_msgs,
            "tool_evidence":            evidence,
            "observer_tool_call_count": new_count,
        }

    # ─────────────────────────────────────────────────────
    # Observer — 결과 추출
    # ─────────────────────────────────────────────────────

    async def _node_observer_finalize(self, state: AgentState) -> AgentState:
        msgs          = state.get("observer_messages", [])
        tool_evidence = state.get("tool_evidence", {})

        final_content = ""
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.content:
                final_content = m.content
                break

        if state.get("verbose"):
            print(f"     [Observer 최종]\n{final_content[:500]}")

        # ── chart_type / image_desc: stage1 결과 우선 사용 ──────────────────
        # ReAct 최종 답변은 이제 {"done": true} 신호뿐 → 파싱 불필요
        fallback_type = state.get("chart_type", "") or "알 수 없음"
        chart_type    = fallback_type

        # ── image_desc: stage1 JSON에서 의미있는 텍스트 추출 ──────────────
        # ReAct 최종 답변은 {"done": true} 신호뿐 → stage1 분석에서 가져옴
        # 원시 JSON 전체 대신 핵심 필드만 조합 → reporter/RAG에 깔끔한 컨텍스트 제공
        _obs_raw  = state.get("observer_analysis", "")
        _obs_data = parse_json(_obs_raw)
        if _obs_data:
            _parts = []
            if _obs_data.get("chart_message"):
                _parts.append(_obs_data["chart_message"])
            if _obs_data.get("all_numbers"):
                _parts.append(f"수치: {str(_obs_data['all_numbers'])[:150]}")
            if _obs_data.get("visual_anomalies"):
                _anomalies = "; ".join(str(a) for a in _obs_data["visual_anomalies"][:3])
                _parts.append(f"이상: {_anomalies}")
            image_desc = " | ".join(_parts) if _parts else _obs_raw[:400]
        else:
            image_desc = _obs_raw[:400] or final_content[:400]

        # ── stage1 시각 가설 보존 ─────────────────────────────────────────
        stage1_hypotheses = [m for m in state.get("observer_hypotheses", []) if m in MISLEADER_KEYS]

        # ── 합성 LLM: stage1 가설 + 도구 결과 → suspected 독립 판단 ─────
        # ReAct LLM 대화 노이즈 없이 정제된 증거만으로 결론 도출
        synthesized = await self._observer_synthesize(
            stage1_hypotheses, tool_evidence, image_desc
        )
        # 합집합: stage1(시각) + 합성LLM 판단. _extract_from_evidence는 도구확증 추가 보험
        suspected = list(dict.fromkeys(stage1_hypotheses + synthesized))
        suspected = self._extract_from_evidence(suspected, tool_evidence)

        tool_count = sum(1 for m in msgs if isinstance(m, ToolMessage))
        print(f"     → 도구 {tool_count}회 / 증거: {list(tool_evidence.keys())}")
        print(f"     → 1차 의심: {suspected}")

        # RAG는 1단계(observer_analyze)에서 이미 조회 완료 → 결과 재사용
        similar       = state.get("similar_examples", [])
        rag_hint_text = state.get("rag_hint", "")
        if state.get("use_rag", True):
            try:
                if not rag_hint_text:
                    # fallback: 1단계 RAG 실패 시 여기서 재시도
                    rag_query     = image_desc or chart_type
                    similar       = await asyncio.to_thread(query_vector_db, rag_query, 3)
                    rag_hint_text = await asyncio.to_thread(format_rag_hint, rag_query, 8)
                    if similar:
                        print(f"     📚 RAG 유사사례 {len(similar)}건 로드 (fallback)")
                else:
                    print(f"     📚 RAG 유사사례 {len(similar)}건 (1단계 결과 재사용)")
                # _rag_refine: stage1 시각 가설은 도구 모순만으로 제거 금지
                suspected = await self._rag_refine(suspected, chart_type, image_desc,
                                                   tool_evidence, rag_hint_text,
                                                   protected=stage1_hypotheses)
            except Exception as e:
                print(f"     ⚠️ RAG 건너뜀: {e}")

        conv = log_conversation(state.get("conversation", []), "🔍 관찰자",
            f"차트:{chart_type} | 도구:{tool_count}회 | 의심:{suspected}")

        return {
            **state,
            "chart_type": chart_type, "image_description": image_desc,
            "initial_observations": _obs_raw or final_content,  # stage1 분석 원문 저장
            "suspected_misleaders": suspected,
            "similar_examples": similar, "rag_hint": rag_hint_text, "conversation": conv,
            "iteration": 0, "early_exit": False,
            "math_checks": {
                **state.get("math_checks", {}),
                # unknown/duplicate 메시지는 JSON이 아니므로 제외 — 실제 도구 결과만 저장
                **{
                    f"obs_tool_{k}": {"note": v[:200]}
                    for k, v in tool_evidence.items()
                    if k in OBSERVER_TOOLS_BY_NAME  # 실존하는 도구 결과만
                },
            },
        }

    # ─────────────────────────────────────────────────────
    # Math Verifier (토론 전 실행)
    # ─────────────────────────────────────────────────────

    async def _node_math_verifier(self, state: AgentState) -> AgentState:
        print("\n  🧮 수학 검증 에이전트...")

        # 2단계(observer_mandatory_tools)에서 이미 추출한 경우 재사용, 없으면 여기서 추출
        nums: dict = state.get("extracted_numbers") or {}
        if nums:
            print(f"     ♻️  extracted_numbers 재사용 (Vision LLM 재추출 생략)")
        else:
            print(f"     📐 extracted_numbers 없음 → Vision LLM으로 추출")
            extract_sys = textwrap.dedent("""
Extract all numbers from this chart. Output ONLY JSON (null or [] if unreadable):
```json
{
  "y_min": null, "y_max": null, "y_ticks": [], "x_labels": [], "x_ticks": [],
  "data_values": [], "visual_ratios": [], "bar_start_values": [],
  "pie_pcts": [], "pie_angles_deg": [],
  "has_dual_axis": false, "r_min": null, "r_max": null, "r_ticks": [],
  "has_log_scale": false, "log_labeled": false,
  "visual_areas": [], "title": null, "x_label": null, "y_label": null,
  "has_unit": false, "chart_width_px": null, "chart_height_px": null,
  "annotated_indices": [], "highlighted_indices": [],
  "bin_edges": [], "visual_bin_widths_px": [],
  "is_bidirectional_bar": false,
  "left_bar_values": [], "left_bar_px": [],
  "right_bar_values": [], "right_bar_px": [],
  "visual_y_px": []
}
```
Notes:
- visual_ratios = bar heights as 0-1 fraction of Y-axis range.
- visual_areas = relative visual sizes between chart elements (NO pixel measurement needed).
  Estimate by eye: if bubble A looks twice as large as bubble B, use [2.0, 1.0].
  For pie slices: estimate area proportion relative to the first slice.
  Leave [] if the chart has no size-encoded elements (bars, lines, etc.).
- For bidirectional bar: left_bar_px/right_bar_px = estimated bar length 0.0-1.0.
- For donut: annotated_indices = indices of prominently highlighted slice numbers.
- For line charts: visual_y_px = each data point's Y pixel position from top (top=0).
  Estimate based on chart height. Example: chart height 300px, point at 1/3 from top → 100.
""")
            raw = await _timed("Math 숫자추출 Vision LLM (fallback)", asyncio.to_thread(
                call_vision_llm,
                f"Extract all numbers. Suspected: {state['suspected_misleaders']}",
                state["image_path"], extract_sys, 1200, 0.0, False,   # use_thinking=False
            ))
            parsed = parse_json(raw)
            if isinstance(parsed, dict):
                nums = parsed

        # ── LLM 수치 분석 (도구 없이 JSON 판단만) ──────────
        math_sys = textwrap.dedent(f"""
You are a mathematical fraud verifier. Analyze the extracted numbers and identify confirmed manipulations.

Based ONLY on the numbers provided, determine which of these are mathematically confirmed:
- y_min > 0 AND data values significantly higher → truncated_axis
- tick intervals inconsistent → inconsistent_tick
- visual ratios don't match label values → data_visual_disproportion
- bar starts != 0 → non_aligned_baseline
- pie angles don't match percentages → misrepresentation

Output ONLY this JSON:
{{"confirmed_misleaders": ["key1", "key2"], "math_summary": "brief summary"}}

Available keys: {', '.join(sorted(MISLEADER_KEYS))}
""").strip()

        math_msgs: list[Any] = [
            SystemMessage(content=math_sys),
            HumanMessage(content=(
                f"Numbers:\n{json.dumps(nums, ensure_ascii=False, indent=2)}\n\n"
                f"Suspected: {state['suspected_misleaders']}\n\n"
                "Analyze the numbers above and output ONLY the JSON."
            )),
        ]

        math_confirmed: list[str] = []
        # bind_tools 대신 JSON 출력만 요청 (XML 직렬화 오류 방지)
        llm = ChatOllama(model=LLM_NAME, num_predict=1000, temperature=0.0, reasoning=False)
        try:
            resp = await _timed("Math 수치분석 LLM", asyncio.to_thread(llm.invoke, math_msgs))
            parsed_math = parse_json(resp.content or "") or {}
            math_confirmed = [
                m for m in parsed_math.get("confirmed_misleaders", [])
                if m in MISLEADER_KEYS
            ]
            if math_confirmed:
                print(f"     → 수학 LLM 확정: {math_confirmed}")
        except Exception as e:
            print(f"     ⚠️ 수학 LLM 실패: {e}")

        # 직접 수학 도구 실행
        mt     = MathTools()
        checks = self._run_math_tools(mt, nums)

        suspected = list(state.get("suspected_misleaders", []))
        suspected = self._update_suspected_from_math(suspected, checks)
        for mk in math_confirmed:
            if mk not in suspected:
                suspected.append(mk)

        clear_errors = sum([
            checks.get("axis_truncation", {}).get("truncated", False),
            checks.get("pie", {}).get("verdict") in ("부적절", "오류"),
            checks.get("pie_angles", {}).get("distortion_detected", False),
            checks.get("bar_scale_symmetry", {}).get("scale_manipulation", False),
            checks.get("dual_axis", {}).get("misleading", False),
            checks.get("log_scale", {}).get("unlabeled", False),
            not checks.get("order", {}).get("correct", True),
            checks.get("label_match", {}).get("distortion_detected", False),
            checks.get("baseline", {}).get("misaligned", False),
            checks.get("data_gap", {}).get("gaps_detected", False),
            len(math_confirmed) > 0,
        ])
        early_exit = clear_errors >= 2

        conv = log_conversation(state.get("conversation", []), "🧮 수학검증",
            f"LLM확정:{math_confirmed} | 의심:{suspected}")

        return {
            **state,
            "extracted_numbers":    nums,
            "math_checks":          {**state.get("math_checks", {}), **checks},
            "suspected_misleaders": suspected,
            "early_exit":           early_exit,
            "conversation":         conv,
        }

    # ─────────────────────────────────────────────────────
    # Self-Critique — 보수적 기준으로 증거 확인
    # ─────────────────────────────────────────────────────

    async def _node_self_critique(self, state: AgentState) -> AgentState:
        suspected   = list(state.get("suspected_misleaders", []))
        math_checks = state.get("math_checks", {})
        tool_ev     = state.get("tool_evidence", {})

        print(f"\n  🤔 최종 판정 ({len(suspected)}개)...")

        # ── 1단계: 최종 misleaders 결정 (최대 5개, 우선순위 정렬) ─
        # 강도 순 정렬: math확인(0) > 도구증거(1) > RAG유사사례(2) > 의심만(3)
        rag_keys: set[str] = set()
        for ex in state.get("similar_examples", []):
            for m in ex.get("misleaders", []):
                if m in MISLEADER_KEYS:
                    rag_keys.add(m)
        if rag_keys:
            print(f"     📚 RAG 우선순위 보너스 대상: {sorted(rag_keys)}")

        def _priority(k: str) -> int:
            if self._math_confirms(k, math_checks):
                return 0
            if any(self._tool_evidence_supports(k, tn, tv) for tn, tv in tool_ev.items()):
                return 1
            if k in rag_keys:
                return 2   # RAG 유사사례에서 나온 항목 — 순수 시각 의심보다 우선
            return 3

        final = sorted(suspected, key=_priority)[:5]

        # ── 2단계: 규칙 기반 verdict ──────────────────────
        # 오류: 수학 또는 도구증거로 확증된 항목이 있을 때
        # 경고: 의심 항목이 있지만 확증 없을 때
        # 정상: 최종 목록이 비어 있을 때
        math_confirmed_set = {k for k in final if self._math_confirms(k, math_checks)}
        tool_confirmed_set = {k for k in final
                              if any(self._tool_evidence_supports(k, tn, tv)
                                     for tn, tv in tool_ev.items())}

        if math_confirmed_set or tool_confirmed_set:
            verdict = "오류"
        elif final:
            verdict = "경고"
        else:
            verdict = "정상"

        # ── 3단계: recheck 조건 (관찰자가 도구 미호출 시만) ─
        obs_tools      = state.get("observer_tool_call_count", 0)
        iteration      = state.get("iteration", 0)
        needs_recheck  = False
        recheck_reason = ""
        if obs_tools == 0 and iteration < 1 and not state.get("early_exit"):
            needs_recheck  = True
            recheck_reason = "FULL_RESTART: 관찰자 도구 미실행"

        # ── 4단계: 신뢰도 구조 (confidence scorer용) ────────
        # 설명(explanation)은 reporter가 직접 생성
        # math(0.9) > 도구증거(0.75) > 의심만(0.5)
        confidence = {
            k: {
                "score":            0.9  if self._math_confirms(k, math_checks) else
                                    0.75 if any(self._tool_evidence_supports(k, tn, tv)
                                                for tn, tv in tool_ev.items()) else 0.5,
                "math_confirmed":   self._math_confirms(k, math_checks),
                "debate_consensus": False,   # debate 제거 — 항상 False
                "reason":           "auto-scored by self_critique",
            }
            for k in final
        }

        print(f"     → 최종: {final} | {verdict} | 재검토:{needs_recheck}")
        conv = log_conversation(state.get("conversation", []), "🤔 최종판정",
            f"판결:{verdict} | 확정:{final}")

        return {
            **state,
            "suspected_misleaders":  suspected,
            "final_misleaders":      final,
            "self_critique_removed": [],
            "verdict":               verdict,
            "confidence":            confidence,
            "needs_recheck":         needs_recheck,
            "recheck_reason":        recheck_reason,
            "iteration":             iteration + 1,
            "conversation":          conv,
        }

    # ─────────────────────────────────────────────────────
    # Recheck
    # ─────────────────────────────────────────────────────

    async def _node_recheck(self, state: AgentState) -> AgentState:
        reason          = state.get("recheck_reason", "")
        is_full_restart = reason.startswith("FULL_RESTART:")
        print(f"\n  🔄 {'완전 재분석' if is_full_restart else '재검토'} ({reason[:60]})")

        if is_full_restart:
            system = textwrap.dedent(f"""

Re-analyze this chart from scratch as a fraud investigator.
Reason: {reason.replace('FULL_RESTART: ', '')}

1. Identify chart type and ALL numbers
2. Check each deception technique systematically
3. Output ONLY:
```json
{{"confirmed":[],"dismissed":[],"new":[],"suspected":[],"summary":""}}
```
Available keys: {', '.join(sorted(MISLEADER_KEYS))}
""")
            raw = await _timed("Recheck 완전재분석 Vision LLM", asyncio.to_thread(
                call_vision_llm,
                "Re-analyze this chart for deliberate manipulation. Find ALL deceptions.",
                state["image_path"], system, 800, 0.1, False,   # use_thinking=False
            ))
            p = parse_json(raw) or {}
            if not p:
                print(f"     ⚠️ 완전재분석 JSON 파싱 실패 → 기존 suspected 유지")
                updated = list(state.get("suspected_misleaders", []))
            else:
                new_suspected = [m for m in p.get("suspected", []) if m in MISLEADER_KEYS]
                confirmed     = [m for m in p.get("confirmed", []) if m in MISLEADER_KEYS]
                new_found     = [m for m in p.get("new", [])       if m in MISLEADER_KEYS]
                updated = list(dict.fromkeys(confirmed + new_found + new_suspected))
            print(f"     → 완전재분석: {updated}")
        else:
            system = textwrap.dedent(f"""

Recheck: {reason} | Targets: {state.get('suspected_misleaders')}
```json
{{"confirmed":[],"dismissed":[],"new":[],"summary":""}}
```
""")
            raw = await _timed("Recheck 부분재검토 Vision LLM", asyncio.to_thread(
                call_vision_llm,
                f"Recheck: {reason}\nVerify: {state['suspected_misleaders']}",
                state["image_path"], system, 600, 0.1, False,   # use_thinking=False
            ))
            p = parse_json(raw) or {}
            if not p:
                print(f"     ⚠️ 재검토 JSON 파싱 실패 → 기존 suspected 유지")
                updated = list(state.get("suspected_misleaders", []))
            else:
                confirmed = [m for m in p.get("confirmed", []) if m in MISLEADER_KEYS]
                dismissed = [m for m in p.get("dismissed", []) if m in MISLEADER_KEYS]
                new_found = [m for m in p.get("new", [])       if m in MISLEADER_KEYS]
                updated   = list(dict.fromkeys(
                    [m for m in state["suspected_misleaders"] if m not in dismissed] + new_found
                ))

        conv = log_conversation(state.get("conversation", []), "🔄 재검토", f"확인:{updated}")
        return {**state, "suspected_misleaders": updated, "needs_recheck": False, "conversation": conv}

    # ─────────────────────────────────────────────────────
    # Reporter
    # ─────────────────────────────────────────────────────

    async def _node_reporter(self, state: AgentState) -> AgentState:
        print("\n  📋 보고서 에이전트...")
        final      = state.get("final_misleaders", [])
        verdict    = state.get("verdict", "확인불가")
        chart_type = state.get("chart_type", "알 수 없음")

        # ── 오류 목록 (확증 수준 포함) ──────────────────────
        conf_data = state.get("confidence", {})
        error_lines = []
        for key in final:
            info      = MISLEADER_TAXONOMY.get(key, {})
            cd        = conf_data.get(key, {})
            math_ok   = cd.get("math_confirmed", False)
            score     = cd.get("score", 0.5)
            level     = "수학적 확증" if math_ok else ("도구 확증" if score >= 0.70 else "시각 의심")
            error_lines.append(
                f"- {info.get('name', key)} [{level}, 확신도:{score:.0%}]: {info.get('desc', '')[:100]}"
            )
        errors_str = "\n".join(error_lines) if error_lines else "발견된 조작 없음"

        system = """
보고서 에이전트. 차트 조작 수사 결과를 전문적인 한국어로 작성.

구조:
1. 📊 분석 요약 (차트 유형, 판결, 수사 과정)
2. 🚨 발견된 조작 (각 조작: 이름 + 확증 수준 + 구체적 수치 증거 + 독자 오도 방법)
3. 🕵️ 교묘한 오류 분석 (어떻게 숨겨져 있었는가, 수치 인용 필수)
4. ✅ 권고사항

정상 판정 시: 도구 검증 결과를 간결하게 요약.
중요 규칙:
- 오류마다 [수학적 확증] / [도구 확증] / [시각 의심] 수준을 명시
- 구체적인 수치를 반드시 인용 (예: "Y축이 60에서 시작해 실제 차이를 과장")
- 기술 용어 나열 금지, 독자가 이해하기 쉬운 한국어로 작성
"""
        # ── 수학 검증 결과 (obs_tool_* 제외, 핵심만 전달) ──
        math_checks = state.get("math_checks", {})
        relevant_checks = {
            k: v for k, v in math_checks.items()
            if not k.startswith("obs_tool_") and not k.startswith("debate_")
        }
        math_str = json.dumps(relevant_checks, ensure_ascii=False)[:600]

        # ── 유사 RAG 사례 요약 (최대 3건, 보고서 근거 강화) ─
        similar = state.get("similar_examples", [])
        similar_lines = []
        for i, ex in enumerate(similar[:3], 1):
            names = ", ".join(
                MISLEADER_TAXONOMY.get(m, {}).get("name", m)
                for m in ex.get("misleaders", [])
            ) or "정상(오류 없음)"
            similar_lines.append(f"  {i}. [{ex.get('chart_type','?')}] {names}")
        similar_str = "\n".join(similar_lines) if similar_lines else "없음"

        user = (
            f"차트 유형: {chart_type}\n판정: {verdict}\n"
            f"설명: {state.get('image_description','')[:300]}\n"
            f"오류 (확증 수준 포함):\n{errors_str}\n"
            f"수학 검증 결과:\n{math_str}\n"
            f"유사 MisViz 사례:\n{similar_str}\n"
        )
        report = await _timed("Reporter 보고서 LLM", asyncio.to_thread(
            call_text_llm, [{"role": "user", "content": user}], system, 1200, 0.2, False
        ))
        conv = log_conversation(state.get("conversation", []), "📋 보고서", report[:200])
        return {**state, "explanation": report, "conversation": conv}

    # ─────────────────────────────────────────────────────
    # Confidence Scorer
    # ─────────────────────────────────────────────────────

    async def _node_confidence_scorer(self, state: AgentState) -> AgentState:
        print("\n  📊 자기 신뢰도 스코어링...")

        final          = state.get("final_misleaders", [])
        conf_data      = state.get("confidence", {})
        math_checks    = state.get("math_checks", {})
        tool_evidence  = state.get("tool_evidence", {})
        obs_tools = state.get("observer_tool_call_count", 0)

        per_misleader: dict[str, dict] = {}
        for m in final:
            score      = 0.0
            breakdown: dict[str, Any] = {}

            js = conf_data.get(m, {}).get("score", 0.5)
            score += float(js) * 0.35 if isinstance(js, (int, float)) else 0.175
            breakdown["conf_score"] = round(float(js), 2) if isinstance(js, (int, float)) else 0.5

            math_ok = self._math_confirms(m, math_checks)
            if math_ok:
                score += 0.30
            breakdown["math_confirmed"] = math_ok

            breakdown["debate_consensus"] = False

            # 도구 증거 지지 개수에 따라 보너스 (최대 +0.20)
            tool_count = sum(
                1 for k in tool_evidence
                if self._tool_evidence_supports(m, k, tool_evidence[k])
            )
            score += min(tool_count * 0.05, 0.20)
            breakdown["supporting_tools"] = tool_count

            if state.get("early_exit") and math_ok:
                score += 0.03

            per_misleader[m] = {
                "score":     round(min(max(score, 0.0), 1.0), 3),
                "grade":     self._score_grade(round(min(max(score, 0.0), 1.0), 3)),
                "breakdown": breakdown,
                "reason":    conf_data.get(m, {}).get("reason", ""),
            }

        overall = self._compute_overall_confidence(state, per_misleader)

        self_confidence = {
            "per_misleader":    per_misleader,
            "overall":          overall,
            "tool_call_count":  obs_tools,
            "math_tool_count":  len([k for k in math_checks
                                     if not k.startswith("obs_tool_") and not k.startswith("debate_")]),
            "agent_iterations": state.get("iteration", 1),
            "early_exit":       state.get("early_exit", False),
        }

        print(f"     → 전체 신뢰도: {overall['score']:.0%} ({overall['grade']})")
        for m, d in per_misleader.items():
            name = MISLEADER_TAXONOMY.get(m, {}).get("name", m)
            math = "🔢수학" if d["breakdown"].get("math_confirmed") else ""
            print(f"       • {name}: {d['score']:.0%} ({d['grade']}) {math}")

        return {**state, "self_confidence": self_confidence}

    # ─────────────────────────────────────────────────────
    # Math 도구 직접 실행
    # ─────────────────────────────────────────────────────

    def _run_math_tools(self, mt: MathTools, nums: dict) -> dict:
        checks:   dict = {}
        y_min     = nums.get("y_min")
        y_max     = nums.get("y_max")
        data_vals = [v for v in (nums.get("data_values") or []) if isinstance(v, (int, float))]

        if y_min is not None and y_max is not None:
            dmin = min(data_vals) if data_vals else float(y_min)
            checks["axis_truncation"] = mt.check_axis_truncation(float(y_min), float(y_max), dmin)

        y_ticks = [v for v in (nums.get("y_ticks") or []) if isinstance(v, (int, float))]
        if len(y_ticks) >= 3:
            checks["y_tick"] = mt.check_tick_intervals(y_ticks)
        x_ticks = [v for v in (nums.get("x_ticks") or []) if isinstance(v, (int, float))]
        if len(x_ticks) >= 3:
            checks["x_tick"] = mt.check_tick_intervals(x_ticks)

        pie = [v for v in (nums.get("pie_pcts") or []) if isinstance(v, (int, float))]
        if pie:
            checks["pie"] = mt.check_multi_pie_sum(pie)  # 다중 파이차트 오탐 방지

        vr = [v for v in (nums.get("visual_ratios") or []) if isinstance(v, (int, float))]
        if vr and data_vals and len(vr) == len(data_vals):
            checks["proportion"] = mt.check_proportion_accuracy(vr, data_vals)

        if nums.get("has_dual_axis") and y_min is not None and y_max is not None:
            r_min, r_max = nums.get("r_min"), nums.get("r_max")
            if r_min is not None and r_max is not None:
                checks["dual_axis"] = mt.check_dual_axis(float(y_min), float(y_max),
                                                         float(r_min), float(r_max))

        if len(y_ticks) >= 3:
            lc = mt.check_log_scale(y_ticks)
            if lc.get("is_log") and not nums.get("log_labeled"):
                checks["log_scale"] = {**lc, "unlabeled": True}

        # visual_areas 우선, 없으면 visual_ratios 로 대체 (LLM이 상대 비율을 더 신뢰성 있게 추정)
        va = [v for v in (nums.get("visual_areas") or []) if isinstance(v, (int, float))]
        if not va:
            va = [v for v in (nums.get("visual_ratios") or []) if isinstance(v, (int, float))]
        if va and data_vals and len(va) == len(data_vals):
            checks["area"] = mt.check_area_distortion(data_vals, va)

        x_labels = nums.get("x_labels") or []
        if x_labels:
            checks["order"]    = mt.check_item_order(x_labels)
            checks["data_gap"] = mt.check_data_gap_detection(x_labels)

        checks["labels"] = {
            "has_title":         bool(nums.get("title")),
            "has_x_label":       bool(nums.get("x_label")),
            "has_y_label":       bool(nums.get("y_label")),
            "has_unit":          bool(nums.get("has_unit")),
            "missing_something": not all([nums.get("y_label"), nums.get("has_unit")]),
        }

        pie_angles = [v for v in (nums.get("pie_angles_deg") or []) if isinstance(v, (int, float))]
        if pie and pie_angles and len(pie) == len(pie_angles):
            checks["pie_angles"] = mt.check_pie_angles(pie, pie_angles)

        if nums.get("is_bidirectional_bar"):
            lv = [v for v in (nums.get("left_bar_values")  or []) if isinstance(v, (int, float))]
            lp = [v for v in (nums.get("left_bar_px")      or []) if isinstance(v, (int, float))]
            rv = [v for v in (nums.get("right_bar_values") or []) if isinstance(v, (int, float))]
            rp = [v for v in (nums.get("right_bar_px")     or []) if isinstance(v, (int, float))]
            if len(lv) >= 2 and len(lv) == len(lp) == len(rv) == len(rp):
                checks["bar_scale_symmetry"] = mt.check_bar_scale_symmetry(lv, lp, rv, rp)

        if vr and data_vals and y_min is not None and y_max is not None and len(vr) == len(data_vals):
            checks["label_match"] = mt.check_label_value_match(data_vals, vr,
                                                                float(y_min), float(y_max))

        bar_starts = [v for v in (nums.get("bar_start_values") or []) if isinstance(v, (int, float))]
        if bar_starts:
            checks["baseline"] = mt.check_baseline_alignment(bar_starts)

        ann_idx = [v for v in (nums.get("annotated_indices") or []) if isinstance(v, int)]
        if ann_idx and data_vals:
            checks["annotation_bias"] = mt.check_selective_annotation(ann_idx, data_vals)

        hl_idx = [v for v in (nums.get("highlighted_indices") or []) if isinstance(v, int)]
        if hl_idx and data_vals:
            checks["color_emphasis"] = mt.check_color_emphasis_bias(hl_idx, data_vals)

        bin_edges = [v for v in (nums.get("bin_edges") or []) if isinstance(v, (int, float))]
        vis_bins  = [v for v in (nums.get("visual_bin_widths_px") or []) if isinstance(v, (int, float))]
        if len(bin_edges) >= 3:
            checks["bin_widths"] = mt.check_bin_widths(bin_edges, vis_bins)

        cw = nums.get("chart_width_px")
        ch = nums.get("chart_height_px")
        if cw and ch and x_ticks and data_vals:
            x_range = max(x_ticks) - min(x_ticks) if len(x_ticks) > 1 else 1
            y_range = ((y_max - y_min)
                       if (y_max is not None and y_min is not None)
                       else max(data_vals, default=1))
            if x_range > 0 and y_range > 0:
                checks["aspect_ratio"] = mt.check_aspect_ratio(int(cw), int(ch), x_range, y_range)

        # slope_distortion: 실제 visual_y_px가 있을 때만 검증 (역산 픽셀은 의미 없음)
        vis_ypx = [v for v in (nums.get("visual_y_px") or []) if isinstance(v, (int, float))]
        if len(data_vals) >= 3 and len(vis_ypx) == len(data_vals):
            x_labels = nums.get("x_labels") or []
            checks["slope"] = mt.check_slope_distortion(data_vals, vis_ypx, x_labels)

        return checks

    # ─────────────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────────────

    # _key_to_tool 삭제 — _infer_tool_args 제거 이후 호출처 없음
    # _infer_tool_args 삭제 — 2단계 _build_args 가 동일 역할을 담당하며 호출 참조 없음

    def _update_suspected_from_math(self, suspected: list[str], math_checks: dict) -> list[str]:
        result = list(suspected)
        for key in MISLEADER_KEYS:
            if key not in result and self._math_confirms(key, math_checks):
                result.append(key)
                print(f"       📌 math → {key}")
        return result

    def _extract_from_evidence(self, suspected: list[str], tool_evidence: dict) -> list[str]:
        result = list(suspected)
        for tool_name, evidence in tool_evidence.items():
            try:
                ev = json.loads(evidence)
            except Exception:
                continue

            def _add(key: str, label: str = ""):
                if key not in result and key in MISLEADER_KEYS:
                    result.append(key)
                    print(f"       📌 {tool_name} → {key}{' ('+label+')' if label else ''}")

            if tool_name == "tool_check_axis_truncation"     and ev.get("truncated"):              _add("truncated_axis")
            if tool_name == "tool_check_pie_angles"          and ev.get("distortion_detected"):    _add("misrepresentation", "각도왜곡")
            if tool_name == "tool_check_tick_intervals"      and not ev.get("consistent", True):   _add("inconsistent_tick")
            if tool_name == "tool_check_item_order"          and not ev.get("correct", True):      _add("inappropriate_order")
            if tool_name == "tool_check_dual_axis"           and ev.get("manipulation_likely"):    _add("dual_axis")
            if tool_name == "tool_check_log_scale"           and ev.get("is_log"):                 _add("log_scale_unlabeled")
            if tool_name == "tool_check_area_distortion"     and ev.get("distortion_detected"):    _add("area_distortion")
            if tool_name == "tool_check_bar_scale_symmetry"  and ev.get("scale_manipulation"):     _add("misrepresentation", "좌우스케일")
            if tool_name == "tool_check_label_value_match"   and ev.get("distortion_detected"):
                _add("data_visual_disproportion"); _add("manipulated_annotation")
            if tool_name == "tool_check_selective_annotation" and ev.get("biased"):               _add("selective_emphasis")
            if tool_name == "tool_check_aspect_ratio"        and ev.get("aspect_manipulated"):    _add("aspect_ratio_manipulation")
            if tool_name == "tool_check_bin_widths"          and ev.get("inconsistent"):          _add("inconsistent_binning")
            if tool_name == "tool_check_baseline_alignment"  and ev.get("misaligned"):            _add("non_aligned_baseline")
            if tool_name == "tool_check_data_gap"            and ev.get("gaps_detected"):         _add("cherry_picking")
            if tool_name == "tool_check_color_emphasis_bias" and ev.get("emphasis_biased"):       _add("selective_emphasis")
            if tool_name == "tool_check_slope_distortion"    and ev.get("distorted"):              _add("slope_distortion")
            if tool_name == "tool_check_misleading_annotation" and ev.get("misleading"):           _add("misleading_annotation")
            if tool_name == "tool_check_cherry_pick_range"   and ev.get("cherry_pick_detected"):  _add("cherry_picking")
            if tool_name == "tool_check_unit_switch"         and ev.get("unit_switch_detected"):  _add("unit_switch")
            if tool_name == "tool_check_cumulative_vs_period" and ev.get("misleading"):            _add("cumulative_misrepresentation")
        return result

    async def _observer_synthesize(
        self,
        stage1_hypotheses: list[str],
        tool_evidence: dict,
        observer_analysis: str,
    ) -> list[str]:
        """
        ReAct 루프 종료 후 전담 경량 LLM으로 최종 suspected_misleaders 결정.

        - stage1 시각 가설 + 도구 결과를 정제된 형태로 전달
        - 이미지·도구 스키마 없음 → 입력 토큰 최소
        - 도구 충돌 시 시각 관찰을 우선 고려하도록 명시
        """
        real_evidence = {
            k: v[:150] for k, v in tool_evidence.items()
            if k in OBSERVER_TOOLS_BY_NAME
        }

        system = textwrap.dedent(f"""
You are finalizing a chart fraud investigation.
You have two sources of evidence:
  1. Visual observations from Phase 1 (human-level perception)
  2. Automated tool results (may contain errors)

IMPORTANT: Tools can be wrong — if visual observation strongly indicates an issue
but a tool says otherwise, trust the visual observation.

Output ONLY JSON (no explanation outside the JSON):
```json
{{"suspected_misleaders": ["key1", "key2"], "reasoning": "one sentence"}}
```

MISLEADER KEYS: {', '.join(sorted(MISLEADER_KEYS))}
""").strip()

        user = (
            f"Phase 1 visual hypotheses: {json.dumps(stage1_hypotheses, ensure_ascii=False)}\n"
            f"Phase 1 analysis summary: {observer_analysis[:300]}\n"
            f"Tool results:\n{json.dumps(real_evidence, ensure_ascii=False)[:600]}\n\n"
            "Which misleading techniques are actually present? "
            "List only confirmed or strongly suspected ones based on ALL evidence."
        )

        try:
            raw = await _timed("Observer 합성 LLM", asyncio.to_thread(
                call_text_llm,
                [{"role": "user", "content": user}],
                system,
                400,   # 합성 결과만 생성 → 짧아도 충분
                0.1,
            ))
            rp = parse_json(raw)
            if rp:
                suspected = [m for m in rp.get("suspected_misleaders", []) if m in MISLEADER_KEYS]
                reasoning = rp.get("reasoning", "")
                if reasoning:
                    print(f"     🔬 합성 근거: {reasoning[:120]}")
                return suspected
        except Exception as e:
            print(f"     ⚠️  합성 LLM 실패: {e}")
        return []

    async def _rag_refine(self, suspected, chart_type, image_desc, tool_evidence, rag_hint,
                          protected: list[str] | None = None) -> list[str]:
        try:
            system = textwrap.dedent(f"""
Verify chart manipulation findings using MisViz evidence.
{rag_hint}
Output JSON only:
```json
{{"confirmed":[],"added":[],"removed":[],"reason":""}}
```
""").strip()
            # 실제 도구 결과만 전달 (unknown/duplicate 메시지 제외)
            real_evidence = {
                k: v[:80] for k, v in tool_evidence.items()
                if k in OBSERVER_TOOLS_BY_NAME
            }
            user = (
                f"Chart: {chart_type}\nDesc: {image_desc[:400]}\n"
                f"Tools: {json.dumps(real_evidence, ensure_ascii=False)[:400]}\n"
                f"Suspected: {suspected}\nRefine the list."
            )
            raw = await _timed("RAG Refine LLM", asyncio.to_thread(
                call_text_llm, [{"role": "user", "content": user}], system, 200, 0.1
            ))
            rp = parse_json(raw)
            if rp:
                confirmed = [m for m in rp.get("confirmed", []) if m in MISLEADER_KEYS]
                added     = [m for m in rp.get("added", [])     if m in MISLEADER_KEYS]
                # stage1 시각 관찰 항목은 도구 한 개의 판단만으로 제거하지 않음
                # (도구가 틀릴 수 있으므로, self_critique 단계에서 최종 필터링)
                protected_set = set(protected or [])
                removed   = set(rp.get("removed", [])) - protected_set
                if protected_set & set(rp.get("removed", [])):
                    blocked = protected_set & set(rp.get("removed", []))
                    print(f"       🛡️  RAG remove 차단 (stage1 관찰): {blocked}")
                merged    = list(dict.fromkeys(
                    m for m in (suspected + confirmed + added) if m not in removed
                ))
                new = [m for m in merged if m not in suspected]
                if new:
                    print(f"       📌 RAG 보강: {new}")
                return merged
        except Exception as e:
            print(f"     ⚠️ RAG 건너뜀: {e}")
        return suspected

    def _math_confirms(self, key: str, math_checks: dict) -> bool:
        mapping = {
            "truncated_axis":            lambda c: c.get("axis_truncation", {}).get("truncated", False),
            "misrepresentation":         lambda c: (
                c.get("pie_angles", {}).get("distortion_detected", False) or
                c.get("bar_scale_symmetry", {}).get("scale_manipulation", False) or
                c.get("label_match", {}).get("distortion_detected", False)
            ),
            "data_visual_disproportion": lambda c: c.get("label_match", {}).get("distortion_detected", False),
            "manipulated_annotation":    lambda c: c.get("label_match", {}).get("distortion_detected", False),
            "inconsistent_tick":         lambda c: (
                not c.get("y_tick", {}).get("consistent", True) or
                not c.get("x_tick", {}).get("consistent", True)
            ),
            "dual_axis":                 lambda c: c.get("dual_axis", {}).get("misleading", False),
            "log_scale_unlabeled":       lambda c: c.get("log_scale", {}).get("unlabeled", False),
            "area_distortion":           lambda c: c.get("area", {}).get("distortion_detected", False),
            "inappropriate_order":       lambda c: not c.get("order", {}).get("correct", True),
            "selective_emphasis":        lambda c: (
                c.get("annotation_bias", {}).get("biased", False) or
                c.get("color_emphasis", {}).get("emphasis_biased", False)
            ),
            "non_aligned_baseline":      lambda c: c.get("baseline", {}).get("misaligned", False),
            "cherry_picking":            lambda c: c.get("data_gap", {}).get("gaps_detected", False),
            "inconsistent_binning":          lambda c: c.get("bin_widths", {}).get("inconsistent", False),
            "aspect_ratio_manipulation":     lambda c: c.get("aspect_ratio", {}).get("aspect_manipulated", False),
            "unit_switch":                   lambda c: c.get("unit_switch", {}).get("unit_switch_detected", False),
            "cumulative_misrepresentation":  lambda c: c.get("cumulative", {}).get("misleading", False),
        }
        checker = mapping.get(key)
        return checker(math_checks) if checker else False

    def _tool_evidence_supports(self, key: str, tool_name: str, evidence: str) -> bool:
        try:
            ev = json.loads(evidence)
            # 키당 복수 도구 지원: list of (tool_name, field, expect_true)
            pairs: dict[str, list[tuple]] = {
                "truncated_axis": [
                    ("tool_check_axis_truncation",      "truncated",           True),
                ],
                "inconsistent_tick": [
                    ("tool_check_tick_intervals",       "consistent",          False),
                ],
                "inappropriate_order": [
                    ("tool_check_item_order",           "correct",             False),
                ],
                "dual_axis": [
                    ("tool_check_dual_axis",            "manipulation_likely", True),
                ],
                "log_scale_unlabeled": [
                    ("tool_check_log_scale",            "is_log",              True),
                ],
                "area_distortion": [
                    ("tool_check_area_distortion",      "distortion_detected", True),
                ],
                "data_visual_disproportion": [
                    ("tool_check_label_value_match",    "distortion_detected", True),
                ],
                "manipulated_annotation": [
                    ("tool_check_label_value_match",    "distortion_detected", True),
                ],
                # misrepresentation: 파이각도·양방향막대·레이블불일치 세 도구 모두 해당
                "misrepresentation": [
                    ("tool_check_pie_angles",           "distortion_detected", True),
                    ("tool_check_bar_scale_symmetry",   "scale_manipulation",  True),
                    ("tool_check_label_value_match",    "distortion_detected", True),
                ],
                # selective_emphasis: 색상 강조 + 선택적 주석 두 도구 모두 해당
                "selective_emphasis": [
                    ("tool_check_color_emphasis_bias",  "emphasis_biased",     True),
                    ("tool_check_selective_annotation", "biased",              True),
                ],
                "aspect_ratio_manipulation": [
                    ("tool_check_aspect_ratio",         "aspect_manipulated",  True),
                ],
                "inconsistent_binning": [
                    ("tool_check_bin_widths",           "inconsistent",        True),
                ],
                "non_aligned_baseline": [
                    ("tool_check_baseline_alignment",   "misaligned",          True),
                ],
                "cherry_picking": [
                    ("tool_check_data_gap",             "gaps_detected",       True),
                    ("tool_check_cherry_pick_range",    "cherry_pick_detected",True),
                ],
                "unit_switch": [
                    ("tool_check_unit_switch",          "unit_switch_detected",True),
                ],
                "cumulative_misrepresentation": [
                    ("tool_check_cumulative_vs_period", "misleading",          True),
                ],
                "slope_distortion": [
                    ("tool_check_slope_distortion",     "distorted",           True),
                ],
                "misleading_annotation": [
                    ("tool_check_misleading_annotation","misleading",          True),
                ],
            }
            if key not in pairs:
                return False
            for t_name, field, expect_true in pairs[key]:
                if tool_name == t_name:
                    return bool(ev.get(field)) == expect_true
            return False
        except Exception:
            return False

    @staticmethod
    def _score_grade(score: float) -> str:
        if score >= 0.85: return "매우 높음"
        if score >= 0.70: return "높음"
        if score >= 0.55: return "중간"
        if score >= 0.40: return "낮음"
        return "매우 낮음"

    def _compute_overall_confidence(
        self, state: AgentState, per_misleader: dict
    ) -> dict:
        obs_tools  = state.get("observer_tool_call_count", 0)
        math_tools = len([k for k in state.get("math_checks", {})
                          if not k.startswith("obs_tool_") and not k.startswith("debate_")])
        final      = state.get("final_misleaders", [])

        if final:
            scores = [d["score"] for d in per_misleader.values()]
            base   = sum(scores) / len(scores) if scores else 0.5
            bonus  = min(obs_tools  * 0.008, 0.06)
            bonus += min(math_tools * 0.025, 0.10)
            bonus += 0.02 if state.get("early_exit") else 0.0
        else:
            base  = 0.60
            bonus = min(obs_tools * 0.04, 0.20) + min(math_tools * 0.02, 0.10)

        overall = round(min(base + bonus, 1.0), 3)
        return {
            "score": overall,
            "grade": DistortionDetectorAgent._score_grade(overall),
            "basis": (
                f"Observer도구:{obs_tools}회 | 수학도구:{math_tools}개 | "
                f"판단반복:{state.get('iteration',1)}회"
            ),
        }
