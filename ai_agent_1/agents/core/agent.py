"""
DistortionDetectorAgent — LangGraph 기반 차트 오류 감지 멀티에이전트

그래프:
  [Observer 3단계]
  observer_analyze (1단계: 분석, 도구없음)
  → observer_mandatory_tools (2단계: 차트유형별 필수도구 자동실행)
  → observer_llm ⇄ observer_tools (3단계: 가설기반 ReAct)
  → observer_finalize
  → math (비전 추출 + LLM 루프 + 직접 수학도구)
  → debate_adv → debate_obs_llm ⇄ debate_obs_tools → debate_synthesize [1라운드]
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
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from .config import (
    IMAGE_TOKENS,
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

    # 토론 상태
    debate_round:          int
    debate_history:        list[dict]
    debate_obs_messages:   list[BaseMessage]   # 매 라운드 리셋
    debate_adv_claims:     list[dict]
    debate_final_suspects: list[str]
    debate_no_progress:    bool

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
MAX_OBSERVER_ROUNDS  = 4   # Observer ReAct 최대 도구 호출 (가설 기반이라 줄임)
MAX_DEBATE_ROUNDS    = 2   # 토론 최대 라운드
MAX_DEBATE_OBS_TOOLS = 6   # 토론 내 Observer 도구 호출 최대
MAX_MATH_LLM_LOOPS   = 4   # 수학 LLM 도구 루프 최대

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
            "debate_round": 0, "debate_history": [],
            "debate_obs_messages": [], "debate_adv_claims": [],
            "debate_final_suspects": [], "debate_no_progress": False,
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

        # Math (토론 전 실행 — extracted_numbers 확보)
        g.add_node("math",              self._node_math_verifier)

        # Debate 루프 (전체 ReAct)
        g.add_node("debate_adv",        self._node_debate_adversarial)
        g.add_node("debate_obs_llm",    self._node_debate_observer_llm)
        g.add_node("debate_obs_tools",  self._node_debate_observer_tools)
        g.add_node("debate_synthesize", self._node_debate_synthesize)

        # 하위 파이프라인
        g.add_node("self_critique",     self._node_self_critique)
        g.add_node("recheck",           self._node_recheck)
        g.add_node("reporter",          self._node_reporter)
        g.add_node("confidence",        self._node_confidence_scorer)

        # Observer 흐름 (3단계)
        g.add_edge(START,                      "observer_analyze")
        g.add_edge("observer_analyze",         "observer_mandatory_tools")
        g.add_edge("observer_mandatory_tools", "observer_llm")
        g.add_conditional_edges("observer_llm", self._observer_route,
                                {"tools": "observer_tools", "done": "observer_finalize"})
        g.add_edge("observer_tools",    "observer_llm")
        g.add_edge("observer_finalize", "math")

        # Math → Debate
        g.add_edge("math", "debate_adv")

        # Debate 흐름 (전체 ReAct 루프)
        g.add_edge("debate_adv", "debate_obs_llm")
        g.add_conditional_edges("debate_obs_llm", self._debate_obs_route,
                                {"tools": "debate_obs_tools", "done": "debate_synthesize"})
        g.add_edge("debate_obs_tools", "debate_obs_llm")
        g.add_conditional_edges("debate_synthesize", self._route_debate,
                                {"continue": "debate_adv", "done": "self_critique"})

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

    def _debate_obs_route(self, state: AgentState) -> Literal["tools", "done"]:
        msgs = state.get("debate_obs_messages", [])
        if not msgs:
            return "done"
        last = msgs[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            used = sum(1 for m in msgs if isinstance(m, ToolMessage))
            if used >= MAX_DEBATE_OBS_TOOLS:
                return "done"
            return "tools"
        return "done"

    def _route_debate(self, state: AgentState) -> Literal["continue", "done"]:
        if state.get("debate_no_progress", False):
            print(f"     ⏩ 토론 조기 종료 (R{state.get('debate_round',0)}: 새 발견 없음)")
            return "done"
        if state.get("debate_round", 0) >= MAX_DEBATE_ROUNDS:
            return "done"
        return "continue"

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

        img_b64 = await asyncio.to_thread(image_to_base64, state["image_path"])

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

        raw = await asyncio.to_thread(
            call_vision_llm,
            "Analyze this chart for potential deception. What is it trying to make you believe? What tricks might be used?",
            state["image_path"],
            system,
            1200,
            0.1,
            False,   # use_thinking=False — JSON 출력 보장
        )

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
            raw_nums = await asyncio.to_thread(
                call_vision_llm,
                f"Extract all numbers. Suspected: {state.get('observer_hypotheses', [])}",
                state["image_path"],
                extract_sys, 1200, 0.0, False,   # use_thinking=False
            )
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
            use_rag       = state.get("use_rag", True)
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
                f"\n## Similar deceptive charts from database:\n{rag_hint[:400]}\n"
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

After all hypotheses are checked, output ONLY this JSON:
{{
  "chart_type": "bar|line|pie|donut|scatter|bubble|histogram|bidirectional_bar|area",
  "image_description": "all numbers, labels, colors, annotations seen in the chart",
  "suspected_misleaders": ["key1", "key2"],
  "tool_evidence": {{"tool_name": "what was found"}}
}}

## MISLEADER KEYS: {', '.join(sorted(MISLEADER_KEYS))}

RULES:
- Only call tools for hypotheses NOT already covered by mandatory tools above.
- If mandatory tools already confirmed an error → include it in final JSON.
- Write interpretation text after EVERY tool result.
- Output JSON only when all hypotheses are verified.
- Take your time and think step by step.
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

        # 루프 중 — 독려 메시지 갱신 (이전 독려 메시지 제거 후 새 것으로 교체)
        elif used > 0:
            already_called = list(state.get("tool_evidence", {}).keys())
            already_str = ", ".join(already_called) if already_called else "none"
            msgs = list(msgs)
            # 직전에 추가된 독려 메시지가 있으면 교체 (중복 누적 방지)
            if (msgs and isinstance(msgs[-1], HumanMessage)
                    and isinstance(msgs[-1].content, str)
                    and msgs[-1].content.startswith("Good.")):
                msgs = msgs[:-1]
            msgs = msgs + [HumanMessage(content=(
                f"Good. {used} tool(s) called so far.\n"
                f"Already called (do NOT call these again): {already_str}\n"
                "If all hypotheses are verified, output the final JSON now. "
                "If there are remaining unchecked hypotheses, call ONE new tool (not from the list above). "
                "Remember: write [Checking] before calling a tool, [Found]/[Clear] after result."
            ))]

        available = self._get_available_tools()
        llm = ChatOllama(model=LLM_NAME, num_predict=4000, temperature=0.1, reasoning=False)
        llm_with_tools = llm.bind_tools(available)

        try:
            resp = await asyncio.to_thread(llm_with_tools.invoke, msgs)
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
                result = f"알 수 없는 도구: {name}"
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

        parsed = parse_json(final_content)
        if not parsed:
            suspected  = [k for k in MISLEADER_KEYS if k in final_content]
            chart_type = "알 수 없음"
            image_desc = final_content[:400]
        else:
            chart_type = normalize_chart_type(parsed.get("chart_type", "알 수 없음"))
            image_desc = parsed.get("image_description", final_content[:400])
            suspected  = [m for m in parsed.get("suspected_misleaders", []) if m in MISLEADER_KEYS]

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
                # _rag_refine: 최종 tool_evidence를 반영해 suspected 목록 정제
                suspected = await self._rag_refine(suspected, chart_type, image_desc,
                                                   tool_evidence, rag_hint_text)
            except Exception as e:
                print(f"     ⚠️ RAG 건너뜀: {e}")

        conv = log_conversation(state.get("conversation", []), "🔍 관찰자",
            f"차트:{chart_type} | 도구:{tool_count}회 | 의심:{suspected}")

        return {
            **state,
            "chart_type": chart_type, "image_description": image_desc,
            "initial_observations": final_content,
            "suspected_misleaders": suspected,
            "similar_examples": similar, "rag_hint": rag_hint_text, "conversation": conv,
            "iteration": 0, "early_exit": False,
            "math_checks": {
                **state.get("math_checks", {}),
                **{f"obs_tool_{k}": {"note": v[:200]} for k, v in tool_evidence.items()},
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
            raw = await asyncio.to_thread(
                call_vision_llm,
                f"Extract all numbers. Suspected: {state['suspected_misleaders']}",
                state["image_path"], extract_sys, 1200, 0.0, False,   # use_thinking=False
            )
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
            resp = await asyncio.to_thread(llm.invoke, math_msgs)
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
            "math_checks":          {**checks, **state.get("math_checks", {})},
            "suspected_misleaders": suspected,
            "early_exit":           early_exit,
            "conversation":         conv,
        }

    # ─────────────────────────────────────────────────────
    # Debate — 반박자 (Devil's Advocate)
    # ─────────────────────────────────────────────────────

    async def _node_debate_adversarial(self, state: AgentState) -> AgentState:
        rnd        = state.get("debate_round", 0) + 1
        suspected  = state.get("suspected_misleaders", [])
        history    = state.get("debate_history", [])
        chart_type = state.get("chart_type", "")
        image_desc = state.get("image_description", "")
        tool_ev    = state.get("tool_evidence", {})

        print(f"\n  😈 반박자 에이전트 [라운드 {rnd}/{MAX_DEBATE_ROUNDS}]...")

        # 이전 라운드 요약
        prev_str = ""
        for h in history:
            prev_str += (
                f"\n[라운드 {h['round']}]\n"
                f"  반박자 주장: {h.get('adv_claims_summary','')}\n"
                f"  관찰자 응답: {h.get('obs_response_summary','')}\n"
                f"  합의된 오류: {h.get('agreed_errors',[])}\n"
                f"  기각된 항목: {h.get('dismissed',[])}\n"
            )

        # 도구 증거 (300자로 확장)
        tool_ev_str = json.dumps(
            {k: v[:300] for k, v in tool_ev.items()},
            ensure_ascii=False, indent=2
        )

        rag_hint    = state.get("rag_hint", "")
        rag_section = f"\n## Similar deceptive chart cases:\n{rag_hint[:800]}\n" if rag_hint else ""

        all_keys = ", ".join(sorted(MISLEADER_KEYS))

        system = textwrap.dedent(f"""
You are an aggressive devil's advocate auditing a chart fraud investigation.
Your mission: ATTACK the observer's findings ruthlessly and find errors they MISSED.

## Chart you are analyzing:
Type: {chart_type}
Description: {image_desc[:1000]}

## Observer's current findings:
Suspected errors: {suspected}

## All tool evidence collected so far:
{tool_ev_str[:1200]}
{rag_section}
## Previous debate rounds:
{prev_str.strip() if prev_str else 'none'}

## Your attack strategy (MUST follow all 3):

### 1. CHALLENGE existing findings (be skeptical)
For EACH suspected error the observer found, ask:
- Is the evidence actually strong enough? Could this be a false alarm?
- Did the tool measure the right thing? Are the numbers reliable?
- Is there an alternative innocent explanation?

### 2. FIND missed errors (look at the chart yourself)
Look at the image carefully. The observer may have missed:
- Subtle axis manipulations (does Y-axis start at 0?)
- Color/emphasis tricks (are certain bars highlighted?)
- Missing context (is this cherry-picked data?)
- Scale distortions (do visual sizes match actual values?)
- Annotation tricks (are labels selective or misleading?)

### 3. ESCALATE any confirmed errors
If the observer found a real error, push harder: is it worse than reported?
Are there related deceptions they didn't notice?

## ALL possible misleader keys:
{all_keys}

Take your time and think step by step before forming your challenges and hypotheses.

## Output ONLY valid JSON:
{{
  "challenges": [
    {{"target_error": "<key from suspected list>", "challenge_strength": "weak|moderate|strong", "reason": "<specific visual or numeric evidence for why it may be wrong or overstated>"}}
  ],
  "new_hypotheses": [
    {{"misleader_key": "<key>", "hypothesis": "<specific visual claim with numbers>", "suggested_tool": "<tool_name>", "visual_evidence": "<what you see in the chart>"}}
  ],
  "adversarial_summary": "<overall verdict: what was missed, overstated, or needs deeper investigation>"
}}
""").strip()

        prompt = (
            f"Round {rnd}/{MAX_DEBATE_ROUNDS}. "
            "You have access to the chart image. "
            "Inspect it carefully and challenge the observer's findings aggressively. "
            "Find at least 2 missed errors and challenge at least 1 existing finding with specific visual evidence."
        )

        try:
            raw = await asyncio.to_thread(
                call_vision_llm,
                prompt,
                state["image_path"],
                system,
                2000,   # max_tokens
                0.3,    # temperature (약간 창의적)
                False,  # use_thinking=False
            )
        except Exception as e:
            print(f"     ⚠️ Vision LLM 실패, 텍스트 LLM fallback: {e}")
            raw = await asyncio.to_thread(
                call_text_llm,
                [{"role": "user", "content": prompt}],
                system, 2000, 0.3
            )

        p      = parse_json(raw)
        claims = []
        if p:
            for h in p.get("new_hypotheses", []):
                key = h.get("misleader_key", "")
                if key:
                    claims.append({
                        "type":           key,
                        "hypothesis":     h.get("hypothesis", ""),
                        "tool":           h.get("suggested_tool", h.get("tool", "")),
                        "visual_evidence": h.get("visual_evidence", ""),
                    })
            for c in p.get("challenges", []):
                target = c.get("target_error", "")
                if target:
                    claims.append({
                        "type":             "challenge",
                        "target_error":     target,
                        "challenge_strength": c.get("challenge_strength", "moderate"),
                        "reason":           c.get("reason", ""),
                    })
            print(f"     → 주장 {len(claims)}개: {[c.get('type','?') for c in claims]}")
        else:
            print(f"     ⚠️ 반박자 JSON 파싱 실패 → 원문 사용")

        adv_summary = p.get("adversarial_summary", raw[:200]) if p else raw[:200]
        conv = log_conversation(state.get("conversation", []),
            f"😈 반박자[R{rnd}]",
            f"주장:{len(claims)}개 | {adv_summary[:100]}")

        return {
            **state,
            "debate_round":        rnd,
            "debate_adv_claims":   claims,
            "conversation":        conv,
            "debate_obs_messages": [],   # 새 라운드마다 리셋
            "debate_no_progress":  False,
        }

    # ─────────────────────────────────────────────────────
    # Debate — Observer LLM (전체 ReAct 루프)
    # ─────────────────────────────────────────────────────

    async def _node_debate_observer_llm(self, state: AgentState) -> AgentState:
        rnd        = state.get("debate_round", 1)
        claims     = state.get("debate_adv_claims", [])
        suspected  = state.get("suspected_misleaders", [])
        chart_type = state.get("chart_type", "")
        image_desc = state.get("image_description", "")
        nums       = state.get("extracted_numbers", {})
        msgs       = state.get("debate_obs_messages", [])

        # 첫 메시지 — 초기화 (이미지 재첨부 없음 — context 절약)
        if not msgs:
            print(f"\n  🔍 토론 Observer [라운드 {rnd}] 초기화...")

            # nums는 핵심 수치만 (context 절약)
            key_nums = {k: v for k, v in nums.items()
                        if k in ("data_values", "x_labels", "y_min", "y_max",
                                 "y_ticks", "pie_pcts", "visual_ratios")}
            nums_str = json.dumps(key_nums, ensure_ascii=False)[:600]

            # claims는 핵심 필드만
            claims_slim = [
                {k: c[k] for k in ("type", "target_error", "reason", "hypothesis",
                                    "suggested_tool", "visual_evidence",
                                    "challenge_strength") if k in c}
                for c in claims
            ]
            claims_str = json.dumps(claims_slim, ensure_ascii=False, indent=2)[:800]

            rag_hint    = state.get("rag_hint", "")
            rag_section = f"\n## MisViz RAG:\n{rag_hint[:300]}\n" if rag_hint else ""

            # 도전 항목과 새 가설 분리
            challenges    = [c for c in claims if c.get("type") == "challenge"]
            new_hyps      = [c for c in claims if c.get("type") != "challenge"]
            challenge_txt = "\n".join(
                f"  - [{c.get('challenge_strength','?').upper()}] '{c['target_error']}': {c.get('reason','')}"
                for c in challenges
            ) or "  없음"
            new_hyp_txt = "\n".join(
                f"  - '{h['type']}': {h.get('hypothesis','')} (시각증거: {h.get('visual_evidence','')})"
                for h in new_hyps
            ) or "  없음"

            system = textwrap.dedent(f"""
You are the chart fraud investigator defending your findings in Debate Round {rnd}/{MAX_DEBATE_ROUNDS}.
{rag_section}
## Chart: {chart_type} | {image_desc[:200]}

## Key numbers:
{nums_str}

## Response strategy:

### 1. DEFEND or CONCEDE each challenge
Call a tool → write "[DEFENDED] <key>" or "[CONCEDED] <key>: why"

### 2. INVESTIGATE new hypotheses
Call the suggested tool → write "[CONFIRMED NEW] <key>" or "[RULED OUT] <key>"

### 3. Final JSON:
```json
{{
  "newly_confirmed": [],
  "conceded": [],
  "still_suspected": [],
  "observer_summary": ""
}}
```
Rules: call tools first · use actual numbers · no guessing
""").strip()

            human_txt = (
                f"Suspected errors: {suspected}\n\n"
                f"## Challenges:\n{challenge_txt}\n\n"
                f"## New hypotheses:\n{new_hyp_txt}\n\n"
                "Call tools to defend/investigate. Start with the first tool call."
            )
            msgs = [SystemMessage(content=system), HumanMessage(content=human_txt)]

        used = sum(1 for m in msgs if isinstance(m, ToolMessage))
        print(f"     [토론 Observer LLM R{rnd}] 메시지:{len(msgs)} / 도구결과:{used}")

        # 루프 중 — 독려 메시지 갱신 (이전 독려 메시지 제거 후 새 것으로 교체)
        if used > 0:
            already_called = list(state.get("tool_evidence", {}).keys())
            already_str = ", ".join(already_called) if already_called else "none"
            msgs = list(msgs)
            # 직전에 추가된 독려 메시지가 있으면 교체 (중복 누적 방지)
            if (msgs and isinstance(msgs[-1], HumanMessage)
                    and isinstance(msgs[-1].content, str)
                    and msgs[-1].content.startswith("Good.")):
                msgs = msgs[:-1]
            msgs = msgs + [HumanMessage(content=(
                f"Good. {used} tool(s) called so far.\n"
                f"Already called (do NOT call these again): {already_str}\n"
                "If all challenges and new hypotheses are addressed, output the final JSON now. "
                "If there are remaining unchecked items, call ONE new tool (not from the list above). "
                "Remember: write [DEFENDED]/[CONCEDED]/[CONFIRMED NEW]/[RULED OUT] after each result."
            ))]

        llm = ChatOllama(model=LLM_NAME, num_predict=4000, temperature=0.1, reasoning=False)
        llm_with_tools = llm.bind_tools(self._get_available_tools())

        try:
            resp = await asyncio.to_thread(llm_with_tools.invoke, msgs)
        except Exception as e:
            print(f"     ❌ LLM 실패: {e}")
            resp = AIMessage(content="[LLM 호출 실패]")

        if getattr(resp, "tool_calls", None):
            print(f"     → 도구 호출: {[tc['name'] for tc in resp.tool_calls]}")
        else:
            print(f"     → 최종 반론 생성 (R{rnd})")

        return {"debate_obs_messages": msgs + [resp]}

    # ─────────────────────────────────────────────────────
    # Debate — Observer 도구 실행
    # ─────────────────────────────────────────────────────

    async def _node_debate_observer_tools(self, state: AgentState) -> AgentState:
        msgs = state.get("debate_obs_messages", [])
        if not msgs or not isinstance(msgs[-1], AIMessage):
            return state

        last_ai   = msgs[-1]
        tool_msgs: list[ToolMessage] = []
        evidence  = dict(state.get("tool_evidence", {}))

        for tc in (last_ai.tool_calls or []):
            name, args, tid = tc["name"], tc["args"], tc["id"]
            if name not in OBSERVER_TOOLS_BY_NAME:
                result = f"알 수 없는 도구: {name}"
            elif name in evidence:
                # 이미 실행된 도구 — 재실행 차단, 기존 결과 반환
                result = f"[DUPLICATE] {name} was already called. Previous result: {evidence[name][:200]}. Do NOT call this tool again."
                print(f"       ⛔ [토론] {name}: 중복 호출 차단 (기존 결과 재사용)")
            else:
                try:
                    result = await asyncio.to_thread(OBSERVER_TOOLS_BY_NAME[name].invoke, args)
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)
                    evidence[name] = result[:400]
                    print(f"       ✅ [토론] {name}: {result[:100]}...")
                except Exception as te:
                    result = f"오류: {te}"
                    print(f"       ❌ [토론] {name}: {te}")
            tool_msgs.append(ToolMessage(tool_call_id=tid, name=name, content=result))

        return {
            "debate_obs_messages": msgs + tool_msgs,
            "tool_evidence":       evidence,
        }

    # ─────────────────────────────────────────────────────
    # Debate — 라운드 합의 추출
    # ─────────────────────────────────────────────────────

    async def _node_debate_synthesize(self, state: AgentState) -> AgentState:
        rnd       = state.get("debate_round", 1)
        msgs      = state.get("debate_obs_messages", [])
        claims    = state.get("debate_adv_claims", [])
        suspected = list(state.get("suspected_misleaders", []))
        history   = list(state.get("debate_history", []))
        tool_ev   = state.get("tool_evidence", {})

        final_content = ""
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.content:
                final_content = m.content
                break

        if state.get("verbose"):
            print(f"     [토론 합의 R{rnd}]\n{final_content[:400]}")

        p               = parse_json(final_content)
        newly_confirmed: list[str] = []
        conceded:        list[str] = []

        if p:
            newly_confirmed = [m for m in p.get("newly_confirmed", []) if m in MISLEADER_KEYS]
            conceded        = [m for m in p.get("conceded", [])        if m in MISLEADER_KEYS]
            for hv in p.get("hypothesis_verifications", []):
                if hv.get("result") == "confirmed":
                    key = hv.get("hypothesis_type", "")
                    if key in MISLEADER_KEYS and key not in newly_confirmed:
                        newly_confirmed.append(key)

        # no_progress 판단: 이번 라운드 토론에서 새로 확정/기각된 것만 기준
        no_progress = (len(newly_confirmed) == 0 and len(conceded) == 0)

        # agreed_errors = 실제 토론 합의 항목만 기록 (safety net 제외)
        # → self_critique에서 debate_consensus 점수(0.8)를 정확하게 부여하기 위함
        debate_agreed_this_round = list(newly_confirmed)

        # 도구 증거에서 자동 추출 (safety net) — suspected 보강용으로만 사용
        newly_from_tools = self._extract_from_evidence([], tool_ev)
        for k in newly_from_tools:
            if k not in suspected:
                suspected.append(k)
                print(f"     📌 [토론R{rnd}] {k} 추가 (도구증거)")

        for k in newly_confirmed:
            if k not in suspected:
                suspected.append(k)
                print(f"     📌 [토론R{rnd}] {k} 추가 (토론합의)")
        for k in conceded:
            if k in suspected:
                suspected.remove(k)
                print(f"     ❌ [토론R{rnd}] {k} 기각")

        adv_summary = (
            " | ".join(c.get("hypothesis", c.get("reason", c.get("challenge", "")))[:60] for c in claims[:3])
            if claims else ""
        )
        obs_summary = p.get("observer_summary", final_content[:200]) if p else final_content[:200]

        round_record = {
            "round":                rnd,
            "adv_claims_summary":   adv_summary[:200],
            "obs_response_summary": obs_summary[:200],
            "agreed_errors":        debate_agreed_this_round,  # 토론 합의만 (safety net 제외)
            "dismissed":            conceded,
        }
        history.append(round_record)

        debate_math = {
            f"debate_R{rnd}_{k}": {"note": v[:150]}
            for k, v in tool_ev.items()
            if k.startswith("tool_check_")
        }

        conv = log_conversation(state.get("conversation", []),
            f"🤝 토론합의[R{rnd}]",
            f"확정:{newly_confirmed} | 기각:{conceded} | 현재:{suspected}")
        print(f"     → R{rnd} 결과: 확정={newly_confirmed}, 기각={conceded}, 총의심={suspected}")

        return {
            **state,
            "suspected_misleaders":  suspected,
            "debate_history":        history,
            "debate_final_suspects": suspected,
            "debate_no_progress":    no_progress,
            "conversation":          conv,
            "math_checks": {**state.get("math_checks", {}), **debate_math},
        }

    # ─────────────────────────────────────────────────────
    # Self-Critique — 보수적 기준으로 증거 확인
    # ─────────────────────────────────────────────────────

    async def _node_self_critique(self, state: AgentState) -> AgentState:
        suspected   = list(state.get("suspected_misleaders", []))
        math_checks = state.get("math_checks", {})
        tool_ev     = state.get("tool_evidence", {})

        print(f"\n  🤔 최종 판정 ({len(suspected)}개)...")

        # ── 1단계: 토론 합의 목록 수집 ───────────────────────
        # 증거 없는 항목 제거는 토론의 conceded 결과로 대체
        debate_history = state.get("debate_history", [])
        debate_agreed: set[str] = set()
        for h in debate_history:
            debate_agreed.update(h.get("agreed_errors", []))

        # ── 2단계: 최종 misleaders 결정 (최대 5개, 우선순위 정렬) ─
        # 강도 순 정렬: math확인 > 토론합의 > 도구증거 > 의심
        def _priority(k: str) -> int:
            if self._math_confirms(k, math_checks):
                return 0
            if k in debate_agreed:
                return 1
            if any(self._tool_evidence_supports(k, tn, tv) for tn, tv in tool_ev.items()):
                return 2
            return 3

        final = sorted(suspected, key=_priority)[:5]

        # ── 3단계: 규칙 기반 verdict ──────────────────────
        # 오류: 수학 또는 토론으로 확증된 항목이 있을 때
        # 경고: 의심 항목이 있지만 수학/토론 확증은 없을 때
        # 정상: 최종 목록이 비어 있을 때
        math_confirmed_set  = {k for k in final if self._math_confirms(k, math_checks)}
        debate_confirmed_set = {k for k in final if k in debate_agreed}

        if math_confirmed_set or debate_confirmed_set:
            verdict = "오류"
        elif final:
            verdict = "경고"
        else:
            verdict = "정상"

        # ── 4단계: recheck 조건 (관찰자가 도구 미호출 시만) ─
        obs_tools      = state.get("observer_tool_call_count", 0)
        iteration      = state.get("iteration", 0)
        needs_recheck  = False
        recheck_reason = ""
        if obs_tools == 0 and iteration < 1 and not state.get("early_exit"):
            needs_recheck  = True
            recheck_reason = "FULL_RESTART: 관찰자 도구 미실행"

        # ── 5단계: 신뢰도 구조 (confidence scorer용) ────────
        # 설명(explanation)은 reporter가 직접 생성
        confidence = {
            k: {
                "score":            0.9 if self._math_confirms(k, math_checks) else
                                    0.8 if k in debate_agreed else
                                    0.65 if any(self._tool_evidence_supports(k, tn, tv)
                                                for tn, tv in tool_ev.items()) else 0.5,
                "math_confirmed":   self._math_confirms(k, math_checks),
                "debate_consensus": k in debate_agreed,
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
            raw = await asyncio.to_thread(
                call_vision_llm,
                "Re-analyze this chart for deliberate manipulation. Find ALL deceptions.",
                state["image_path"], system, 800, 0.1, False,   # use_thinking=False
            )
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
            raw = await asyncio.to_thread(
                call_vision_llm,
                f"Recheck: {reason}\nVerify: {state['suspected_misleaders']}",
                state["image_path"], system, 600, 0.1, False,   # use_thinking=False
            )
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

        error_lines = []
        for key in final:
            info = MISLEADER_TAXONOMY.get(key, {})
            error_lines.append(f"- {info.get('name', key)}: {info.get('desc', '')[:100]}")
        errors_str = "\n".join(error_lines) if error_lines else "발견된 조작 없음"

        debate_summary = "\n".join(
            f"  라운드{h['round']}: 합의={h.get('agreed_errors',[])} 기각={h.get('dismissed',[])}"
            for h in state.get("debate_history", [])
        )

        system = """
보고서 에이전트. 차트 조작 수사 결과를 전문적인 한국어로 작성.

구조:
1. 📊 분석 요약 (차트 유형, 판결, 수사 과정)
2. 🚨 발견된 조작 (각 조작: 이름 + 구체적 증거 + 독자 오도 방법)
3. 🕵️ 교묘한 오류 분석 (어떻게 숨겨져 있었는가)
4. ✅ 권고사항

정상 판정 시: 도구 검증 결과를 간결하게 요약.
절대 금지: 불필요한 기술 용어 나열
"""
        user = (
            f"차트 유형: {chart_type}\n판정: {verdict}\n"
            f"설명: {state.get('image_description','')[:300]}\n"
            f"오류:\n{errors_str}\n"
            f"토론 기록:\n{debate_summary or '없음'}\n"
            f"수학 검증: {json.dumps(state.get('math_checks',{}), ensure_ascii=False)[:400]}\n"
        )
        report = await asyncio.to_thread(
            call_text_llm, [{"role": "user", "content": user}], system, 2000, 0.2, False
        )
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
        obs_tools      = state.get("observer_tool_call_count", 0)
        debate_history = state.get("debate_history", [])

        debate_agreed: set[str] = set()
        for h in debate_history:
            debate_agreed.update(h.get("agreed_errors", []))

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

            if m in debate_agreed:
                score += 0.15
            breakdown["debate_consensus"] = m in debate_agreed

            tool_count = sum(
                1 for k in tool_evidence
                if self._tool_evidence_supports(m, k, tool_evidence[k])
            )
            score += min(tool_count * 0.04, 0.12)
            breakdown["supporting_tools"] = tool_count

            if state.get("early_exit") and math_ok:
                score += 0.03

            per_misleader[m] = {
                "score":     round(min(max(score, 0.0), 1.0), 3),
                "grade":     self._score_grade(round(min(max(score, 0.0), 1.0), 3)),
                "breakdown": breakdown,
                "reason":    conf_data.get(m, {}).get("reason", ""),
            }

        overall = self._compute_overall_confidence(state, per_misleader, debate_history)

        self_confidence = {
            "per_misleader":        per_misleader,
            "overall":              overall,
            "tool_call_count":      obs_tools,
            "math_tool_count":      len([k for k in math_checks if k.startswith("tool_")]),
            "debate_rounds":        len(debate_history),
            "agent_iterations":     state.get("iteration", 1),
            "debate_agreed_errors": list(debate_agreed),
            "early_exit":           state.get("early_exit", False),
        }

        print(f"     → 전체 신뢰도: {overall['score']:.0%} ({overall['grade']})")
        for m, d in per_misleader.items():
            name = MISLEADER_TAXONOMY.get(m, {}).get("name", m)
            cons = "✅합의" if d["breakdown"].get("debate_consensus") else ""
            math = "🔢수학" if d["breakdown"].get("math_confirmed") else ""
            print(f"       • {name}: {d['score']:.0%} ({d['grade']}) {cons}{math}")

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

    def _key_to_tool(self, key: str) -> str:
        mapping = {
            "truncated_axis":            "tool_check_axis_truncation",
            "misrepresentation":         "tool_check_pie_angles",
            "inconsistent_tick":         "tool_check_tick_intervals",
            "inappropriate_order":       "tool_check_item_order",
            "dual_axis":                 "tool_check_dual_axis",
            "log_scale_unlabeled":       "tool_check_log_scale",
            "area_distortion":           "tool_check_area_distortion",
            "data_visual_disproportion": "tool_check_label_value_match",
            "manipulated_annotation":    "tool_check_label_value_match",
            "selective_emphasis":        "tool_check_color_emphasis_bias",
            "aspect_ratio_manipulation": "tool_check_aspect_ratio",
            "inconsistent_binning":      "tool_check_bin_widths",
            "non_aligned_baseline":      "tool_check_baseline_alignment",
            "cherry_picking":            "tool_check_data_gap",
        }
        return mapping.get(key, "")

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

    async def _rag_refine(self, suspected, chart_type, image_desc, tool_evidence, rag_hint) -> list[str]:
        try:
            system = textwrap.dedent(f"""
Verify chart manipulation findings using MisViz evidence.
{rag_hint}
Output JSON only:
```json
{{"confirmed":[],"added":[],"removed":[],"reason":""}}
```
""").strip()
            user = (
                f"Chart: {chart_type}\nDesc: {image_desc[:400]}\n"
                f"Tools: {json.dumps({k: v[:80] for k, v in tool_evidence.items()}, ensure_ascii=False)[:400]}\n"
                f"Suspected: {suspected}\nRefine the list."
            )
            raw = await asyncio.to_thread(
                call_text_llm, [{"role": "user", "content": user}], system, 500, 0.1
            )
            rp = parse_json(raw)
            if rp:
                confirmed = [m for m in rp.get("confirmed", []) if m in MISLEADER_KEYS]
                added     = [m for m in rp.get("added", [])     if m in MISLEADER_KEYS]
                removed   = set(rp.get("removed", []))
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
            pairs = {
                "truncated_axis":            ("tool_check_axis_truncation",      "truncated",           True),
                "inconsistent_tick":         ("tool_check_tick_intervals",       "consistent",          False),
                "inappropriate_order":       ("tool_check_item_order",           "correct",             False),
                "dual_axis":                 ("tool_check_dual_axis",            "manipulation_likely", True),
                "log_scale_unlabeled":       ("tool_check_log_scale",            "is_log",              True),
                "area_distortion":           ("tool_check_area_distortion",      "distortion_detected", True),
                "data_visual_disproportion": ("tool_check_label_value_match",    "distortion_detected", True),
                "manipulated_annotation":    ("tool_check_label_value_match",    "distortion_detected", True),
                "selective_emphasis":        ("tool_check_color_emphasis_bias",  "emphasis_biased",     True),
                "aspect_ratio_manipulation": ("tool_check_aspect_ratio",         "aspect_manipulated",  True),
                "inconsistent_binning":      ("tool_check_bin_widths",           "inconsistent",        True),
                "non_aligned_baseline":      ("tool_check_baseline_alignment",   "misaligned",          True),
                "cherry_picking":            ("tool_check_data_gap",             "gaps_detected",       True),
                "unit_switch":               ("tool_check_unit_switch",           "unit_switch_detected", True),
                "cumulative_misrepresentation": ("tool_check_cumulative_vs_period", "misleading",         True),
            }
            if key not in pairs or tool_name != pairs[key][0]:
                return False
            _, field, expect_true = pairs[key]
            val = ev.get(field)
            return bool(val) == expect_true
        except Exception:
            return False

    def _tool_clearly_negative(self, key: str, tool_name: str, ev: dict) -> bool:
        """도구가 명확히 정상(오류 없음)을 확인한 경우에만 True."""
        clear_negative = {
            "tool_check_axis_truncation":    lambda e: e.get("truncated") is False and e.get("y_axis_start", 1) == 0,
            "tool_check_pie_sum":            lambda e: e.get("verdict") == "정상" and abs(e.get("total", 100) - 100) < 1,
            "tool_check_tick_intervals":     lambda e: e.get("consistent") is True and e.get("cv", 1) < 0.03,
            "tool_check_item_order":         lambda e: e.get("correct") is True,
            "tool_check_dual_axis":          lambda e: e.get("manipulation_likely") is False,
            "tool_check_log_scale":          lambda e: e.get("is_log") is False,
            "tool_check_area_distortion":    lambda e: e.get("distortion_detected") is False,
            "tool_check_label_value_match":  lambda e: e.get("distortion_detected") is False and e.get("max_diff_pct", 100) < 3,
            "tool_check_selective_annotation": lambda e: e.get("biased") is False,
            "tool_check_aspect_ratio":       lambda e: e.get("aspect_manipulated") is False,
            "tool_check_bin_widths":         lambda e: e.get("inconsistent") is False,
            "tool_check_baseline_alignment": lambda e: e.get("misaligned") is False,
            "tool_check_data_gap":           lambda e: e.get("gaps_detected") is False,
            "tool_check_color_emphasis_bias":      lambda e: e.get("emphasis_biased") is False,
            "tool_check_slope_distortion":         lambda e: e.get("distorted") is False,
            "tool_check_misleading_annotation":    lambda e: e.get("misleading") is False,
            "tool_check_cherry_pick_range":        lambda e: e.get("cherry_pick_detected") is False,
            "tool_check_unit_switch":              lambda e: e.get("unit_switch_detected") is False,
            "tool_check_cumulative_vs_period":     lambda e: e.get("misleading") is False,
        }
        checker = clear_negative.get(tool_name)
        if not checker:
            return False
        try:
            return bool(checker(ev))
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
        self, state: AgentState, per_misleader: dict, debate_history: list
    ) -> dict:
        obs_tools   = state.get("observer_tool_call_count", 0)
        math_tools  = len([k for k in state.get("math_checks", {}) if not k.startswith("obs_tool_")])
        debate_rnds = len(debate_history)
        final       = state.get("final_misleaders", [])

        if final:
            scores = [d["score"] for d in per_misleader.values()]
            base   = sum(scores) / len(scores) if scores else 0.5
            bonus  = min(obs_tools  * 0.008, 0.06)
            bonus += min(math_tools * 0.015, 0.06)
            bonus += min(debate_rnds * 0.02, 0.06)
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
                f"토론:{debate_rnds}라운드 | 판단반복:{state.get('iteration',1)}회"
            ),
        }
