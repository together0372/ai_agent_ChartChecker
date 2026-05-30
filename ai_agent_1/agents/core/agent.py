"""
DistortionDetectorAgent — LangGraph 기반 차트 오류 감지 멀티에이전트

그래프:
  observer_start → observer_llm ⇄ observer_tools → observer_finalize
  → math (비전 추출 + LLM 루프 + 직접 수학도구)
  → debate_adv → debate_obs_llm ⇄ debate_obs_tools → debate_synthesize [3라운드]
  → self_critique
  → judge → [recheck] → reporter → confidence → END

핵심 설계:
  - 수학(math)을 토론 앞으로 이동 → extracted_numbers 채워진 후 debate 시작
  - 토론 Observer: 전체 ReAct 루프 (이미지 재확인 + 도구 직접 호출)
  - 관찰자에 차트 유형별 심층 체크리스트 내장
  - self_critique: 보수적 기준 (명확한 반박 증거 있을 때만 제거)
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
MAX_OBSERVER_ROUNDS  = 8   # Observer ReAct 최대 도구 호출
MAX_DEBATE_ROUNDS    = 1   # 토론 최대 라운드
MAX_DEBATE_OBS_TOOLS = 6   # 토론 내 Observer 도구 호출 최대
MAX_MATH_LLM_LOOPS   = 4   # 수학 LLM 도구 루프 최대


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

        # Observer ReAct
        g.add_node("observer_start",    self._node_observer_start)
        g.add_node("observer_llm",      self._node_observer_llm)
        g.add_node("observer_tools",    self._node_observer_tools)
        g.add_node("observer_finalize", self._node_observer_finalize)

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

        # Observer 흐름
        g.add_edge(START, "observer_start")
        g.add_edge("observer_start", "observer_llm")
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
    # Observer — 초기화
    # ─────────────────────────────────────────────────────

    async def _node_observer_start(self, state: AgentState) -> AgentState:
        print("\n  🔍 관찰자 에이전트 — 사기 수사관 모드...")

        from PIL import Image
        img_info: dict = {}
        try:
            with Image.open(state["image_path"]) as img:
                img_info = {"w": img.size[0], "h": img.size[1]}
        except Exception:
            pass

        img_b64  = await asyncio.to_thread(image_to_base64, state["image_path"])
        use_rag  = state.get("use_rag", True)
        rag_line = (
            "⚠️ MANDATORY FIRST STEP: Call tool_query_misviz_db with your chart description BEFORE any other tool.\n"
            "This searches the MisViz database of known deceptive charts and MUST be called first.\n"
        ) if use_rag else ""

        system = textwrap.dedent(f"""
You are a chart manipulation detector. Call tools to find deceptions. Do NOT write analysis text — use tools.

{rag_line}
## STEP 1 — Call tools immediately. Required tools by chart type:

BAR chart:
- tool_check_axis_truncation(y_min, y_max, data_min)
- tool_check_label_value_match(label_values, visual_ratios, y_min, y_max)
- tool_check_color_emphasis_bias(highlighted_indices, all_values)
- tool_check_selective_annotation(annotated_indices, all_values)
- tool_check_baseline_alignment(bar_start_values)
- If bars go LEFT and RIGHT from center: tool_check_bar_scale_symmetry(left_values, left_px, right_values, right_px)
  where left_px/right_px = estimated bar length as 0.0–1.0 fraction of chart half-width

PIE/DONUT chart:
- tool_check_pie_sum(percentages)
- tool_check_pie_angles(pie_pcts, pie_angles_deg)
- tool_check_selective_annotation(annotated_indices=[index of prominently shown number], all_values=[all slice %s])
  CRITICAL: if the big displayed number is NOT the largest slice → selective_emphasis

LINE chart:
- tool_check_axis_truncation(y_min, y_max, data_min)
- tool_check_tick_intervals(ticks)
- tool_check_data_gap(shown_x_values)
- tool_check_dual_axis(left_axis_range, right_axis_range, description)
- tool_check_aspect_ratio(width_px, height_px, x_data_range, y_data_range)

ANY chart:
- tool_check_log_scale(ticks) if ticks look exponential
- tool_check_item_order(labels) if X-axis has dates/years
- tool_check_area_distortion(values, visual_areas) for bubble charts
- tool_check_bin_widths(bin_edges, visual_widths_px) for histograms

## STEP 2 — After calling AT LEAST 5 tools, output ONLY this JSON:
```json
{{
  "chart_type": "bar|line|pie|scatter|histogram|bidirectional_bar|donut",
  "image_description": "all numbers, labels, colors seen",
  "suspected_misleaders": ["key1", "key2"],
  "tool_evidence": {{"tool_name": "result summary"}}
}}
```

## MISLEADER KEYS:
{', '.join(sorted(MISLEADER_KEYS))}

RULE: Call tools first. Output JSON only after ≥5 tool calls.
""").strip()

        human_msg = HumanMessage(content=[
            {"type": "text", "text": (
                f"Chart size: {img_info.get('w','?')}×{img_info.get('h','?')}px. "
                "Call tools to check for manipulation. Start calling tools now."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ])

        return {"observer_messages": [SystemMessage(content=system), human_msg]}

    # ─────────────────────────────────────────────────────
    # Observer — LLM 호출
    # ─────────────────────────────────────────────────────

    async def _node_observer_llm(self, state: AgentState) -> AgentState:
        msgs = state.get("observer_messages", [])
        used = sum(1 for m in msgs if isinstance(m, ToolMessage))
        print(f"     [Observer LLM] 메시지:{len(msgs)} / 도구결과:{used}회")

        if 0 < used < 5:
            extra = (
                f"\n\nOnly {used} tool(s) called. Call more tools now. Do NOT output JSON yet."
            )
            invoke_msgs = list(msgs) + [HumanMessage(content=extra)]
        else:
            invoke_msgs = msgs

        available = self._get_available_tools()
        llm = ChatOllama(model=LLM_NAME, num_predict=4000, temperature=0.1,
                         num_image_tokens=IMAGE_TOKENS, think=False)
        llm_with_tools = llm.bind_tools(available)

        try:
            resp = await asyncio.to_thread(llm_with_tools.invoke, invoke_msgs)
        except Exception as e:
            print(f"     ❌ LLM 실패: {e}")
            resp = AIMessage(content="[LLM 호출 실패]")

        if getattr(resp, "tool_calls", None):
            print(f"     → 도구 호출: {[tc['name'] for tc in resp.tool_calls]}")
        else:
            print("     → 최종 답변 생성")

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

        similar: list[dict] = []
        rag_hint_text = ""
        if state.get("use_rag", True):
            try:
                rag_query    = image_desc or chart_type
                similar      = await asyncio.to_thread(query_vector_db, rag_query, 3)
                rag_hint_text = await asyncio.to_thread(format_rag_hint, rag_query, 8)
                suspected    = await self._rag_refine(suspected, chart_type, image_desc,
                                                      tool_evidence, rag_hint_text)
                if similar:
                    print(f"     📚 RAG 유사사례 {len(similar)}건 로드")
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
  "right_bar_values": [], "right_bar_px": []
}
```
Notes: visual_ratios = bar heights as 0-1 fraction of Y-axis range.
For bidirectional bar: left_bar_px/right_bar_px = estimated bar length 0.0-1.0.
For donut: annotated_indices = indices of prominently highlighted slice numbers.
""")
        raw = await asyncio.to_thread(
            call_vision_llm,
            f"Extract all numbers. Suspected: {state['suspected_misleaders']}",
            state["image_path"], extract_sys, 1200, 0.0,
        )
        nums = parse_json(raw)
        if not isinstance(nums, dict):
            nums = {}

        # ── LLM 도구 루프 (최대 MAX_MATH_LLM_LOOPS) ──────────
        math_sys = textwrap.dedent("""
You are a mathematical fraud verifier. Given chart numbers, call tools to verify manipulations.

Call ALL applicable tools:
- y_min > 0 → tool_check_axis_truncation
- pie_pcts → tool_check_pie_sum AND tool_check_pie_angles
- y_ticks ≥3 → tool_check_tick_intervals AND tool_check_log_scale
- has_dual_axis → tool_check_dual_axis
- visual_ratios + data_values → tool_check_label_value_match
- annotated_indices → tool_check_selective_annotation (CRITICAL for donut: checks if annotated ≠ max)
- bar_start_values → tool_check_baseline_alignment
- bin_edges → tool_check_bin_widths
- highlighted_indices → tool_check_color_emphasis_bias
- chart_width_px + chart_height_px → tool_check_aspect_ratio
- x_labels → tool_check_data_gap + tool_check_item_order
- is_bidirectional_bar=true → tool_check_bar_scale_symmetry(left_values, left_px, right_values, right_px)

After ALL applicable tools called, output:
```json
{"confirmed_misleaders": ["key1"], "math_summary": "summary of all findings"}
```
""").strip()

        math_msgs: list[Any] = [
            SystemMessage(content=math_sys),
            HumanMessage(content=(
                f"Numbers:\n{json.dumps(nums, ensure_ascii=False, indent=2)}\n\n"
                f"Suspected: {state['suspected_misleaders']}\n\n"
                "Call ALL applicable tools, then output JSON."
            )),
        ]

        math_evidence:  dict[str, str] = {}
        math_confirmed: list[str]      = []
        llm       = ChatOllama(model=LLM_NAME, num_predict=3000, temperature=0.0, think=False)
        llm_tools = llm.bind_tools(self._get_available_tools())

        for _ in range(MAX_MATH_LLM_LOOPS):
            try:
                resp = await asyncio.to_thread(llm_tools.invoke, math_msgs)
            except Exception as e:
                print(f"     ⚠️ 수학 LLM 실패: {e}")
                break
            math_msgs.append(resp)
            if not getattr(resp, "tool_calls", None):
                math_confirmed = [
                    m for m in parse_json(resp.content or "").get("confirmed_misleaders", [])
                    if m in MISLEADER_KEYS
                ]
                break
            print(f"     → 수학 도구: {[tc['name'] for tc in resp.tool_calls]}")
            for tc in resp.tool_calls:
                if tc["name"] in OBSERVER_TOOLS_BY_NAME:
                    try:
                        res = await asyncio.to_thread(
                            OBSERVER_TOOLS_BY_NAME[tc["name"]].invoke, tc["args"]
                        )
                        if not isinstance(res, str):
                            res = json.dumps(res, ensure_ascii=False)
                        math_evidence[tc["name"]] = res[:300]
                        print(f"       ✅ {tc['name']}: {res[:80]}...")
                    except Exception as te:
                        res = f"오류: {te}"
                        print(f"       ❌ {tc['name']}: {te}")
                else:
                    res = f"알 수 없는 도구: {tc['name']}"
                math_msgs.append(ToolMessage(tool_call_id=tc["id"], name=tc["name"], content=res))

        # 직접 수학 도구 실행
        mt     = MathTools()
        checks = self._run_math_tools(mt, nums)
        for k, v in math_evidence.items():
            checks[f"tool_{k}"] = {"note": v[:200], "source": "math_llm"}

        suspected = list(state.get("suspected_misleaders", []))
        suspected = self._update_suspected_from_math(suspected, checks)
        for mk in math_confirmed:
            if mk not in suspected:
                suspected.append(mk)

        clear_errors = sum([
            checks.get("axis_truncation", {}).get("truncated", False),
            checks.get("pie", {}).get("verdict") == "부적절",
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
            f"LLM도구:{list(math_evidence.keys())} | 확정:{math_confirmed} | 의심:{suspected}")

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

        prev_str = ""
        for h in history:
            prev_str += (
                f"\n[라운드 {h['round']}]\n"
                f"  반박자: {h.get('adv_claims_summary','')}\n"
                f"  관찰자: {h.get('obs_response_summary','')}\n"
                f"  합의 오류: {h.get('agreed_errors',[])}\n"
                f"  기각: {h.get('dismissed',[])}\n"
            )

        system = textwrap.dedent(f"""
Round {rnd}/{MAX_DEBATE_ROUNDS}. You are a devil's advocate. Challenge observer findings and find missed errors.

Previous rounds: {prev_str.strip() if prev_str else 'none'}

Output ONLY valid JSON (no markdown, no explanation):
{{
  "challenges": [{{"target_error": "<misleader_key>", "reason": "<why it may be wrong>"}}],
  "new_hypotheses": [{{"misleader_key": "<one of the MISLEADER_KEYS>", "hypothesis": "<specific claim>", "tool": "<tool_name>"}}],
  "adversarial_summary": "<overall: what was missed or overstated>"
}}

MISLEADER_KEYS examples: truncated_axis, selective_emphasis, dual_axis_manipulation, misrepresentation, data_gap, cherry_picking, log_scale_unlabeled, aspect_ratio_distortion, inverted_axis, baseline_manipulation
""").strip()

        rag_hint = state.get("rag_hint", "")
        rag_section = f"\n## MisViz RAG 유사 사례 (참고):\n{rag_hint[:600]}\n" if rag_hint else ""

        user = (
            f"Chart type: {chart_type}\n"
            f"Image description: {image_desc[:500]}\n"
            f"Observer's suspected errors: {suspected}\n"
            f"Tool evidence so far: {json.dumps({k: v[:60] for k, v in tool_ev.items()}, ensure_ascii=False)[:400]}\n"
            f"{rag_section}"
            f"Round {rnd}: Challenge findings and find what was missed."
        )

        raw = await asyncio.to_thread(
            call_text_llm, [{"role": "user", "content": user}], system, 1500, 0.3
        )

        p      = parse_json(raw)
        claims = []
        if p:
            for h in p.get("new_hypotheses", []):
                key = h.get("misleader_key", "")
                if key:
                    claims.append({"type": key, "hypothesis": h.get("hypothesis", ""), "tool": h.get("tool", "")})
            for c in p.get("challenges", []):
                target = c.get("target_error", "")
                if target:
                    claims.append({"type": "challenge", "target_error": target, "reason": c.get("reason", "")})
            print(f"     → 주장 {len(claims)}개: {[c.get('type','?') for c in claims]}")

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

        # 첫 메시지 — 초기화 (이미지 재첨부)
        if not msgs:
            print(f"\n  🔍 토론 Observer [라운드 {rnd}] 초기화...")

            img_b64    = await asyncio.to_thread(image_to_base64, state["image_path"])
            claims_str = json.dumps(claims, ensure_ascii=False, indent=2)
            nums_str   = json.dumps(nums, ensure_ascii=False)[:600]
            rag_hint   = state.get("rag_hint", "")
            rag_section = f"\n## MisViz RAG 유사 사례:\n{rag_hint[:500]}\n" if rag_hint else ""

            system = textwrap.dedent(f"""
Debate round {rnd}/{MAX_DEBATE_ROUNDS}. Use tools to verify or refute each adversarial claim.
{rag_section}
Numbers extracted: {nums_str}

For each claim below: call the relevant tool, then output JSON.

```json
{{
  "newly_confirmed": ["keys confirmed by tools"],
  "conceded": ["keys disproven by tools"],
  "observer_summary": "findings"
}}
```
""").strip()

            human_content: list[Any] = [
                {"type": "text", "text": (
                    f"Chart type: {chart_type}\n"
                    f"Image description: {image_desc[:300]}\n"
                    f"My current suspected errors: {suspected}\n\n"
                    f"Adversarial's claims this round:\n{claims_str[:800]}\n\n"
                    "Look at the image and use tools to respond to each claim. "
                    "Call at least 3 tools relevant to the adversarial's hypotheses."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]
            msgs = [SystemMessage(content=system), HumanMessage(content=human_content)]

        used = sum(1 for m in msgs if isinstance(m, ToolMessage))
        print(f"     [토론 Observer LLM R{rnd}] 메시지:{len(msgs)} / 도구결과:{used}")

        llm = ChatOllama(model=LLM_NAME, num_predict=4000, temperature=0.1,
                         num_image_tokens=IMAGE_TOKENS, think=False)
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

        # 도구 증거에서 자동 추출 (safety net)
        newly_from_tools = self._extract_from_evidence([], tool_ev)
        for k in newly_from_tools:
            if k not in newly_confirmed:
                newly_confirmed.append(k)

        for k in newly_confirmed:
            if k not in suspected:
                suspected.append(k)
                print(f"     📌 [토론R{rnd}] {k} 추가")
        for k in conceded:
            if k in suspected:
                suspected.remove(k)
                print(f"     ❌ [토론R{rnd}] {k} 기각")

        no_progress = (len(newly_confirmed) == 0 and len(conceded) == 0)

        adv_summary = (
            " | ".join(c.get("hypothesis", c.get("reason", c.get("challenge", "")))[:60] for c in claims[:3])
            if claims else ""
        )
        obs_summary = p.get("observer_summary", final_content[:200]) if p else final_content[:200]

        round_record = {
            "round":                rnd,
            "adv_claims_summary":   adv_summary[:200],
            "obs_response_summary": obs_summary[:200],
            "agreed_errors":        newly_confirmed,
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
        evidence    = dict(tool_ev)
        removed:    list[str] = []

        print(f"\n  🤔 자기비판 + 최종 판정 ({len(suspected)}개)...")

        # ── 1단계: 필터링 ──────────────────────────────────
        for key in list(suspected):
            if self._math_confirms(key, math_checks):
                print(f"     ✅ {key}: 수학 도구 확인 (유지)")
                continue

            already = any(
                self._tool_evidence_supports(key, tn, tv)
                for tn, tv in tool_ev.items()
            )
            if already:
                print(f"     ✅ {key}: 기존 도구 증거 지지 (유지)")
                continue

            tool_name = self._key_to_tool(key)
            if tool_name and tool_name in tool_ev:
                try:
                    ev = json.loads(tool_ev[tool_name])
                    if self._tool_clearly_negative(key, tool_name, ev):
                        print(f"     ❌ {key}: 도구 명확히 정상 → 제거")
                        suspected.remove(key)
                        removed.append(key)
                    else:
                        print(f"     ⚠️ {key}: 도구 불명확 (유지)")
                except Exception:
                    print(f"     ⚠️ {key}: 증거 파싱 오류 (유지)")
                continue

            if tool_name and tool_name in OBSERVER_TOOLS_BY_NAME:
                tool_args = self._infer_tool_args(tool_name, state)
                if tool_args and not all(
                    (v is None or v == [] or v == {}) for v in tool_args.values()
                ):
                    try:
                        result = await asyncio.to_thread(
                            OBSERVER_TOOLS_BY_NAME[tool_name].invoke, tool_args
                        )
                        if not isinstance(result, str):
                            result = json.dumps(result, ensure_ascii=False)
                        evidence[tool_name] = result[:400]
                        ev = json.loads(result)
                        if self._tool_clearly_negative(key, tool_name, ev):
                            print(f"     ❌ {key}: 검증 후 정상 → 제거")
                            suspected.remove(key)
                            removed.append(key)
                        else:
                            print(f"     ✅ {key}: 검증 통과 또는 불명확 (유지)")
                    except Exception as e:
                        print(f"     ⚠️ {key}: 도구 오류 ({e}) → 유지")
                    continue

            print(f"     ⚠️ {key}: 검증 불가 (유지)")

        print(f"     → 필터링 완료: {len(removed)}개 제거 | 남은 의심: {suspected}")

        # ── 2단계: 최종 misleaders 결정 (최대 5개) ──────────
        debate_history = state.get("debate_history", [])
        debate_agreed: set[str] = set()
        for h in debate_history:
            debate_agreed.update(h.get("agreed_errors", []))

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
        math_confirmed_set = {k for k in final if self._math_confirms(k, math_checks)}
        debate_confirmed_set = {k for k in final if k in debate_agreed}
        tool_confirmed_set = {
            k for k in final
            if any(self._tool_evidence_supports(k, tn, tv) for tn, tv in tool_ev.items())
        }

        if math_confirmed_set or debate_confirmed_set:
            verdict = "오류"
        elif tool_confirmed_set or len(final) >= 2:
            verdict = "경고"
        elif final:
            verdict = "경고"
        else:
            verdict = "정상"

        # ── 4단계: recheck 조건 (관찰자가 도구 미호출 시만) ─
        obs_tools     = state.get("observer_tool_call_count", 0)
        iteration     = state.get("iteration", 0)
        needs_recheck = False
        recheck_reason = ""
        if obs_tools == 0 and iteration < 1 and not state.get("early_exit"):
            needs_recheck  = True
            recheck_reason = "FULL_RESTART: 관찰자 도구 미실행"

        # ── 5단계: 설명 생성 ─────────────────────────────
        if final:
            error_names = [MISLEADER_TAXONOMY.get(k, {}).get("name", k) for k in final]
            evidence_tags = []
            for k in final:
                tags = []
                if self._math_confirms(k, math_checks):
                    tags.append("수학검증")
                if k in debate_agreed:
                    tags.append("토론합의")
                if tags:
                    evidence_tags.append(f"{k}({','.join(tags)})")
            explanation = (
                f"발견된 오류: {', '.join(error_names)}. "
                f"증거: {', '.join(evidence_tags) if evidence_tags else '도구 검증'}. "
                f"판정: {verdict}."
            )
        else:
            explanation = "모든 의심 항목이 도구 검증을 통해 정상으로 확인되었습니다."

        # 신뢰도 구조 (confidence scorer용)
        confidence = {
            k: {
                "score":          0.9 if self._math_confirms(k, math_checks) else
                                  0.8 if k in debate_agreed else
                                  0.65 if any(self._tool_evidence_supports(k, tn, tv)
                                              for tn, tv in tool_ev.items()) else 0.5,
                "math_confirmed": self._math_confirms(k, math_checks),
                "debate_consensus": k in debate_agreed,
                "reason": "auto-scored by self_critique",
            }
            for k in final
        }

        print(f"     → 최종: {final} | {verdict} | 재검토:{needs_recheck}")
        conv = log_conversation(state.get("conversation", []), "🤔 자기비판+판정",
            f"판결:{verdict} | 확정:{final} | 제거:{removed}")

        return {
            **state,
            "suspected_misleaders":  suspected,
            "final_misleaders":      final,
            "tool_evidence":         evidence,
            "self_critique_removed": removed,
            "verdict":               verdict,
            "explanation":           explanation,
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
                state["image_path"], system, 800, 0.1,
            )
            p = parse_json(raw)
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
                state["image_path"], system, 600, 0.1,
            )
            p = parse_json(raw)
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
                "score":       round(min(max(score, 0.0), 1.0), 3),
                "grade":       self._score_grade(round(min(max(score, 0.0), 1.0), 3)),
                "breakdown":   breakdown,
                "reason": conf_data.get(m, {}).get("reason", ""),
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
            checks["pie"] = mt.check_pie(pie)

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

        va = [v for v in (nums.get("visual_areas") or []) if isinstance(v, (int, float))]
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

        return checks

    # ─────────────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────────────

    def _key_to_tool(self, key: str) -> str:
        mapping = {
            "truncated_axis":            "tool_check_axis_truncation",
            "inappropriate_pie":         "tool_check_pie_sum",
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

    def _infer_tool_args(self, tool_name: str, state: AgentState) -> dict:
        nums = state.get("extracted_numbers", {})
        defaults: dict[str, dict] = {
            "tool_check_axis_truncation": {
                "y_min":    nums.get("y_min") if nums.get("y_min") is not None else 0,
                "y_max":    nums.get("y_max") if nums.get("y_max") is not None else 100,
                "data_min": min(nums.get("data_values", [0]) or [0]),
            },
            "tool_check_label_value_match": {
                "label_values":  nums.get("data_values", []),
                "visual_ratios": nums.get("visual_ratios", []),
                "y_min":         nums.get("y_min") if nums.get("y_min") is not None else 0,
                "y_max":         nums.get("y_max") if nums.get("y_max") is not None else 100,
            },
            "tool_check_selective_annotation": {
                "annotated_indices": nums.get("annotated_indices", []),
                "all_values":        nums.get("data_values", []),
            },
            "tool_check_aspect_ratio": {
                "width_px":     nums.get("chart_width_px") or 400,
                "height_px":    nums.get("chart_height_px") or 300,
                "x_data_range": 10,
                "y_data_range": nums.get("y_max") if nums.get("y_max") is not None else 100,
            },
            "tool_check_data_gap": {
                "shown_x_values": nums.get("x_labels", []),
            },
            "tool_check_baseline_alignment": {
                "bar_start_values": nums.get("bar_start_values", [0]),
            },
            "tool_check_color_emphasis_bias": {
                "highlighted_indices": nums.get("highlighted_indices", []),
                "all_values":          nums.get("data_values", []),
            },
            "tool_check_bin_widths": {
                "bin_edges":        nums.get("bin_edges", []),
                "visual_widths_px": nums.get("visual_bin_widths_px", []),
            },
            "tool_check_tick_intervals": {
                "ticks": nums.get("y_ticks", []),
            },
            "tool_check_pie_sum": {
                "percentages": nums.get("pie_pcts", []),
            },
            "tool_check_pie_angles": {
                "pie_pcts":       nums.get("pie_pcts", []),
                "pie_angles_deg": nums.get("pie_angles_deg", []),
            },
            "tool_check_log_scale": {
                "ticks": nums.get("y_ticks", []),
            },
            "tool_check_area_distortion": {
                "values":       nums.get("data_values", []),
                "visual_areas": nums.get("visual_areas", []),
            },
        }
        return defaults.get(tool_name, {})

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
            if tool_name == "tool_check_pie_sum"             and not ev.get("appropriate"):        _add("inappropriate_pie")
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
            "inappropriate_pie":         lambda c: c.get("pie", {}).get("verdict") == "부적절",
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
            "inconsistent_binning":      lambda c: c.get("bin_widths", {}).get("inconsistent", False),
            "aspect_ratio_manipulation": lambda c: c.get("aspect_ratio", {}).get("aspect_manipulated", False),
        }
        checker = mapping.get(key)
        return checker(math_checks) if checker else False

    def _tool_evidence_supports(self, key: str, tool_name: str, evidence: str) -> bool:
        try:
            ev = json.loads(evidence)
            pairs = {
                "truncated_axis":            ("tool_check_axis_truncation",      "truncated",           True),
                "inappropriate_pie":         ("tool_check_pie_sum",              "appropriate",         False),
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
            "tool_check_color_emphasis_bias": lambda e: e.get("emphasis_biased") is False,
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

        overall = round(min(base + bonus, 0.97), 3)
        return {
            "score": overall,
            "grade": DistortionDetectorAgent._score_grade(overall),
            "basis": (
                f"Observer도구:{obs_tools}회 | 수학도구:{math_tools}개 | "
                f"토론:{debate_rnds}라운드 | 판단반복:{state.get('iteration',1)}회"
            ),
        }
