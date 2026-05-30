import asyncio
import logging
import time
from pathlib import Path

from langchain_ollama import ChatOllama

from workflow import create_news_workflow
from config import Config
from state import ChartCheckState
from agents.core.rag_utils import init_vector_db
from agents.core.config import MISLEADER_TAXONOMY

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# ⚙️  설정 변수
# ──────────────────────────────────────────────────────────
USE_VECTOR_DB = True
# True  → RAG 벡터DB 사용 (tool_query_misviz_db, RAG 보강 활성화)
# False → RAG 완전 비활성화 (속도 향상, 오프라인 환경에 유용)

# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _collect_images(test_dir: str) -> list[str]:
    """test/ 폴더에서 이미지 파일 목록을 수집 (이름 순 정렬)."""
    folder = Path(test_dir)
    if not folder.exists():
        print(f"  ⚠️  test 폴더 없음: {test_dir}")
        return []
    return sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _get(state, key, default=None):
    """ChartCheckState(Pydantic) 또는 dict 모두 지원하는 안전한 필드 접근."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _print_report(state, image_path: str) -> None:
    """최종 분석 결과를 보기 좋게 출력."""
    _VE    = {"정상": "✅", "경고": "⚠️", "오류": "❌", "확인불가": "❓"}
    _GRADE = {"매우 높음": "🟢", "높음": "🔵", "중간": "🟡", "낮음": "🟠", "매우 낮음": "🔴"}
    fname  = Path(image_path).name

    # 차트가 아닌 경우
    if not _get(state, "is_chart", False):
        print(f"\n{'='*60}")
        print(f"  🚫  [{fname}] 차트/그래프 아님 — 스킵")
        print(f"{'='*60}")
        print(f"  사유: {_get(state, 'skip_reason', '')}")
        print(f"{'='*60}\n")
        return

    verdict = _get(state, "verdict") or "확인불가"
    icon    = _VE.get(verdict, "❓")

    print(f"\n{'='*60}")
    print(f"  {icon}  [{fname}] 최종 판정: {verdict}")
    print(f"{'='*60}")

    conf = _get(state, "confidence", "")
    if conf:
        print(f"  확신도: {conf}")

    sc      = _get(state, "self_confidence", {}) or {}
    overall = sc.get("overall", {})
    if overall:
        grade = overall.get("grade", "?")
        score = overall.get("score", 0.0)
        print(f"  분석신뢰: {_GRADE.get(grade,'⚪')} {score:.0%} ({grade})")
        print(f"           {overall.get('basis', '')}")

    final_misleaders = _get(state, "final_misleaders", []) or []
    visual_errors    = _get(state, "visual_errors", []) or []
    if final_misleaders:
        visual = visual_errors or final_misleaders
        print(f"\n  🚨 발견된 오류 {len(final_misleaders)}개:")
        for key, name in zip(final_misleaders, visual):
            info = MISLEADER_TAXONOMY.get(key, {})
            print(f"    • {name}")
            desc = info.get("desc", "")
            if desc:
                print(f"      {desc[:100]}")
    else:
        print("\n  ✅ 오류 없음")

    explanation = _get(state, "explanation", "")
    if explanation:
        print(f"\n{'─'*60}")
        print("  📋 분석 결과:")
        for line in explanation.split("\n"):
            if line.strip():
                print(f"  {line}")

    print(f"{'='*60}\n")


async def process_image(app, image_path: str):
    """이미지 하나를 워크플로우에서 처리. dict 또는 ChartCheckState 반환."""
    result = await app.ainvoke(ChartCheckState(chart_image_path=image_path))
    # LangGraph는 dict를 반환하므로 Pydantic 모델로 변환 시도
    if isinstance(result, dict):
        try:
            return ChartCheckState(**result)
        except Exception:
            return result  # 변환 실패 시 dict 그대로 반환 (_get으로 안전하게 접근)
    return result


async def main():
    print("""
╔══════════════════════════════════════════════════════╗
║   차트 왜곡 탐지 시스템 (멀티에이전트)               ║
╚══════════════════════════════════════════════════════╝
""")

    images = _collect_images("./test")
    if not images:
        print("  ❌ 처리할 이미지가 없습니다.")
        return

    print(f"  📁 발견된 이미지: {len(images)}개")
    for i, p in enumerate(images, 1):
        print(f"     {i}. {Path(p).name}")

    # RAG 벡터DB 초기화 (USE_VECTOR_DB=True일 때만)
    if USE_VECTOR_DB:
        print(f"\n{'='*60}")
        print("  RAG 벡터DB 초기화 중...")
        print(f"{'='*60}")
        init_vector_db()
    else:
        print(f"\n  ℹ️  USE_VECTOR_DB=False → RAG 벡터DB 스킵 (속도 우선 모드)")

    # 워크플로우 생성 (한 번만)
    llm = ChatOllama(
        model=Config.MODEL_NAME,
        temperature=Config.MODEL_TEMPERATURE,
        top_p=Config.MODEL_TOP_P,
        top_k=Config.MODEL_TOP_K,
    )
    app = create_news_workflow(llm, use_rag=USE_VECTOR_DB)

    results: list[dict] = []

    try:
        for idx, image_path in enumerate(images, 1):
            fname = Path(image_path).name
            print(f"\n\n{'#'*60}")
            print(f"  [{idx}/{len(images)}]  처리 중: {fname}")
            print(f"{'#'*60}")

            try:
                t_start = time.time()
                final_state = await process_image(app, image_path)
                elapsed = time.time() - t_start

                _print_report(final_state, image_path)
                print(f"  ⏱️  소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)\n")

                is_chart = _get(final_state, "is_chart", False)
                results.append({
                    "file":        fname,
                    "is_chart":    is_chart,
                    "verdict":     _get(final_state, "verdict", "") if is_chart else "스킵",
                    "skip_reason": _get(final_state, "skip_reason", ""),
                    "misleaders":  _get(final_state, "final_misleaders", []) or [],
                    "elapsed":     elapsed,
                })
            except Exception as e:
                logger.exception(f"[{fname}] 처리 중 오류")
                print(f"\n  ❌ [{fname}] 오류: {e}\n")
                results.append({"file": fname, "verdict": "오류", "error": str(e), "elapsed": 0})

    except KeyboardInterrupt:
        print("\n\n  사용자에 의해 중단되었습니다.")

    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    """전체 처리 결과 요약."""
    if not results:
        return

    charts     = [r for r in results if r.get("is_chart")]
    non_charts = [r for r in results if r.get("verdict") == "스킵"]
    proc_err   = [r for r in results if r.get("verdict") == "오류"]
    err_found  = [r for r in charts  if r.get("verdict") in ("오류", "경고")]
    clean      = [r for r in charts  if r.get("verdict") == "정상"]

    print(f"\n{'#'*60}")
    print("  📊 전체 처리 요약")
    print(f"{'#'*60}")
    print(f"  총 이미지 : {len(results)}개")
    print(f"  차트 판별 : {len(charts)}개  |  비차트 스킵: {len(non_charts)}개  |  처리오류: {len(proc_err)}개")
    if charts:
        print(f"  차트 결과 : 오류/경고 {len(err_found)}개  |  정상 {len(clean)}개")

    total_time = sum(r.get("elapsed", 0) for r in results)
    print(f"  총 소요시간: {total_time:.1f}초 ({total_time/60:.1f}분)")

    print(f"\n  {'파일명':<25} {'판정':<8} {'시간':>6}  {'비고'}")
    print(f"  {'─'*62}")
    _icon = {"오류": "❌", "경고": "⚠️", "정상": "✅", "스킵": "🚫"}
    for r in results:
        v    = r.get("verdict", "?")
        icon = _icon.get(v, "❓")
        sec  = f"{r.get('elapsed',0):.0f}s"
        note = (
            ", ".join(r.get("misleaders", [])) or "-"
            if v not in ("스킵", "오류")
            else r.get("skip_reason", r.get("error", ""))[:35]
        )
        print(f"  {icon} {r['file']:<23} {v:<8} {sec:>6}  {note}")

    print(f"\n{'#'*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
