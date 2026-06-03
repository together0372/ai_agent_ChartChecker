"""
MisViz 벡터 DB (ChromaDB) 초기화 및 RAG 검색 유틸리티
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .config import (
    HF_TOKEN,
    MISLEADER_KEYS,
    MISLEADER_TAXONOMY,
    MISVIZ_CONTENT_SCHEMA,
    MISVIZ_DATASET,
    MISVIZ_DB_DIR,
    MISVIZ_EMBED,
    MISVIZ_EXAMPLES,
    MISVIZ_IMG_DIR,
    normalize_chart_type,
    normalize_misleader_label,
)

# 선택적 의존성
try:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from datasets import load_dataset as hf_load_dataset
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

_rag_retriever: Any = None


# ─────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────

def _misleader_enrichment_text(keys: list[str]) -> str:
    parts: list[str] = []
    for k in keys:
        tax = MISLEADER_TAXONOMY.get(k)
        if not tax:
            continue
        parts.append(
            f"- {tax['name']} [{k}]: {tax['desc']} "
            f"(은폐 패턴: {tax.get('hide_tip', '')})"
        )
    return "\n".join(parts)


def _build_document_from_example(example: dict, split_name: str, idx: int) -> "Document":
    import uuid as _uuid

    image = example.get("image")
    image_path = Path(MISVIZ_IMG_DIR) / f"{split_name}_{idx}.png"
    if isinstance(image, Image.Image):
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)

    chart_types_raw = example.get("chart_type") or example.get("chart_types") or []
    misleaders_raw  = example.get("misleader")  or example.get("misleaders") or []
    bbox            = example.get("bbox") or []

    if isinstance(chart_types_raw, str):
        chart_types_raw = [chart_types_raw]
    if isinstance(misleaders_raw, str):
        misleaders_raw = [misleaders_raw]

    chart_types    = [normalize_chart_type(c) for c in chart_types_raw]
    misleader_keys = [m for m in (normalize_misleader_label(x) for x in misleaders_raw) if m]
    is_normal      = len(misleader_keys) == 0
    enrichment     = _misleader_enrichment_text(misleader_keys)

    text = (
        f"차트 유형(Chart Types): {', '.join(chart_types) or 'unknown'}\n"
        f"오류 유형(Misleaders): {', '.join(misleader_keys) if misleader_keys else '없음(정상 차트)'}\n"
        f"원본 라벨(raw): {', '.join(str(m) for m in misleaders_raw) or 'none'}\n"
        + (
            f"오류 상세 설명:\n{enrichment}\n"
            if enrichment
            else "이 차트에는 알려진 오류가 없습니다(정상 사례).\n"
        )
    )

    metadata = {
        "id":             str(_uuid.uuid4()),
        "split":          split_name,
        "chart_types":    json.dumps(chart_types, ensure_ascii=False),
        "misleaders":     json.dumps(misleader_keys, ensure_ascii=False),
        "misleaders_raw": json.dumps([str(m) for m in misleaders_raw], ensure_ascii=False),
        "is_normal":      is_normal,
        "bbox":           json.dumps(bbox, ensure_ascii=False),
        "image_path":     str(image_path),
    }
    return Document(page_content=text, metadata=metadata)


def _robust_rmtree(path: Path, max_retries: int = 5, retry_delay: float = 0.5) -> None:
    """Windows [WinError 5] 권한 오류를 처리하는 안전한 디렉토리 삭제."""
    def _handle_error(func, fpath, exc_info):
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception:
            pass

    for attempt in range(1, max_retries + 1):
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(path, onexc=_handle_error)
            else:
                shutil.rmtree(path, onerror=_handle_error)
            return
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"  ⚠️  DB 삭제 재시도 {attempt}/{max_retries}: {e}")
            time.sleep(retry_delay)


def _db_marker_path(db_path: Path) -> Path:
    return db_path / "misviz_db_meta.json"


def _read_db_marker(db_path: Path) -> dict:
    try:
        return json.loads(_db_marker_path(db_path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_db_marker(db_path: Path, count: int, source: str) -> None:
    try:
        _db_marker_path(db_path).write_text(
            json.dumps({
                "embed_model":    MISVIZ_EMBED,
                "content_schema": MISVIZ_CONTENT_SCHEMA,
                "count":          count,
                "source":         source,
                "built_at":       datetime.now().isoformat(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"  ⚠️  DB 마커 기록 실패(무시): {e}")


# ─────────────────────────────────────────────────────────
# 공개 함수
# ─────────────────────────────────────────────────────────

def init_vector_db() -> bool:
    global _rag_retriever

    if not CHROMA_AVAILABLE:
        print("  ⚠️  ChromaDB/HuggingFace 라이브러리 없음 → 내장 예시 사용")
        return False

    MISVIZ_RESET = os.environ.get("MISVIZ_RESET", "0") == "1"

    try:
        embedding_fn = HuggingFaceEmbeddings(model_name=MISVIZ_EMBED)
        db_path = Path(MISVIZ_DB_DIR)

        if MISVIZ_RESET and db_path.exists():
            _robust_rmtree(db_path)
            print("  🗑️  기존 벡터DB 삭제 (MISVIZ_RESET=1)")

        if (db_path / "chroma.sqlite3").exists():
            marker = _read_db_marker(db_path)
            stale = (
                marker.get("embed_model")    != MISVIZ_EMBED
                or marker.get("content_schema") != MISVIZ_CONTENT_SCHEMA
            )
            if stale:
                print("  ♻️  임베딩 모델/콘텐츠 스키마 변경 감지 → 벡터DB 자동 재구축")
                _robust_rmtree(db_path)
            else:
                vector_db = Chroma(
                    collection_name="misviz",
                    embedding_function=embedding_fn,
                    persist_directory=str(db_path),
                )
                cnt = vector_db._collection.count()
                if cnt > 20:
                    _rag_retriever = vector_db.as_retriever(
                        search_type="similarity", search_kwargs={"k": 5}
                    )
                    print(f"  ✅ 실제 MisViz 벡터DB 로드 ({cnt}건) ← {db_path}")
                    return marker.get("source", "real") == "real"
                else:
                    print(f"  ⚠️  기존 DB가 내장 예시({cnt}건) → 실제 MisViz 재다운로드 시도")
                    _robust_rmtree(db_path)

        db_path.mkdir(parents=True, exist_ok=True)
        Path(MISVIZ_IMG_DIR).mkdir(parents=True, exist_ok=True)

        print(f"  📥 UKPLab/misviz 다운로드 중...")
        try:
            load_kwargs: dict = {"path": MISVIZ_DATASET, "trust_remote_code": True}
            if HF_TOKEN:
                load_kwargs["token"] = HF_TOKEN

            ds = hf_load_dataset(**load_kwargs)
            all_docs: list[Document] = []

            for split_name in ds.keys():
                split_ds = ds[split_name]
                print(f"     [{split_name}] {len(split_ds)}건 처리 중...")
                ok = skip = 0
                for idx, example in enumerate(split_ds):
                    try:
                        all_docs.append(_build_document_from_example(example, split_name, idx))
                        ok += 1
                    except Exception as ex_err:
                        skip += 1
                        if skip <= 3:
                            print(f"     ⚠️  {split_name}-{idx} 건너뜀: {ex_err}")
                print(f"     [{split_name}] 완료 — 성공:{ok} / 건너뜀:{skip}")

            if not all_docs:
                raise ValueError("변환된 문서가 0건")

            print(f"  🧠 임베딩 & ChromaDB 저장 중 ({len(all_docs)}건)...")
            vector_db = Chroma(
                collection_name="misviz",
                embedding_function=embedding_fn,
                persist_directory=str(db_path),
            )
            BATCH = 200
            for i in range(0, len(all_docs), BATCH):
                vector_db.add_documents(all_docs[i:i + BATCH])

            _rag_retriever = vector_db.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )
            _write_db_marker(db_path, len(all_docs), "real")
            print(f"  ✅ 실제 MisViz 벡터DB 구축 완료 ({len(all_docs)}건)")
            return True

        except Exception as dl_err:
            print(f"\n  ❌ MisViz 다운로드 실패: {dl_err}")
            print("  ⚠️  【폴백】 내장 예시 20건 사용")

            if db_path.exists():
                _robust_rmtree(db_path)
            db_path.mkdir(parents=True, exist_ok=True)

            builtin_docs = [
                Document(
                    page_content=(
                        f"차트 유형(Chart Types): {normalize_chart_type(ex['chart_type'])}\n"
                        f"오류 유형(Misleaders): {', '.join(ex['misleaders'])}\n"
                        f"사례 설명: {ex['description']}\n"
                        + (_misleader_enrichment_text(ex["misleaders"]) or "")
                    ),
                    metadata={
                        "id":             ex["id"],
                        "split":          "builtin",
                        "chart_types":    json.dumps([normalize_chart_type(ex["chart_type"])], ensure_ascii=False),
                        "misleaders":     json.dumps(ex["misleaders"], ensure_ascii=False),
                        "misleaders_raw": json.dumps(ex["misleaders"], ensure_ascii=False),
                        "is_normal":      len(ex["misleaders"]) == 0,
                        "bbox":           "[]",
                        "image_path":     "",
                    },
                )
                for ex in MISVIZ_EXAMPLES
            ]
            vector_db = Chroma(
                collection_name="misviz",
                embedding_function=embedding_fn,
                persist_directory=str(db_path),
            )
            vector_db.add_documents(builtin_docs)
            _rag_retriever = vector_db.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )
            _write_db_marker(db_path, len(builtin_docs), "builtin")
            print(f"  ⚠️  내장 예시 벡터DB 구성 완료 ({len(builtin_docs)}건)")
            return False

    except Exception as e:
        print(f"  ❌ 벡터DB 초기화 전체 실패: {e}")
        import traceback; traceback.print_exc()
        return False


def query_vector_db(query_text: str, n_results: int = 3) -> list[dict]:
    """MisViz 유사 사례 검색. retriever 없으면 내장 예시 키워드 매칭 폴백."""
    if _rag_retriever is not None:
        try:
            docs = _rag_retriever.invoke(query_text)[:n_results]
            out = []
            for d in docs:
                meta = d.metadata
                try:
                    misleaders = json.loads(meta.get("misleaders", "[]"))
                except Exception:
                    misleaders = []
                try:
                    chart_types = json.loads(meta.get("chart_types", '["unknown"]'))
                    chart_type  = chart_types[0] if chart_types else "unknown"
                except Exception:
                    chart_type = "unknown"
                out.append({
                    "description": d.page_content[:300],
                    "misleaders":  misleaders,
                    "chart_type":  chart_type,
                    "is_normal":   bool(meta.get("is_normal", False)),
                    "image_path":  meta.get("image_path", ""),
                    "split":       meta.get("split", ""),
                })
            return out
        except Exception:
            pass

    q_lower = query_text.lower()
    scored = [
        (
            sum(
                w in ex["description"].lower() or w in " ".join(ex["misleaders"])
                for w in q_lower.split()
            ),
            ex,
        )
        for ex in MISVIZ_EXAMPLES
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "description": ex["description"],
            "misleaders":  ex["misleaders"],
            "chart_type":  ex["chart_type"],
            "is_normal":   len(ex["misleaders"]) == 0,
            "image_path":  "",
            "split":       "builtin",
        }
        for _, ex in scored[:n_results]
    ]


def misviz_prior_from_neighbors(query_text: str, k: int = 8) -> dict:
    """유사 MisViz 사례 k건에서 misleader 출현 빈도를 집계한 prior."""
    neighbors = query_vector_db(query_text, n_results=k)
    counts: dict[str, int] = {}
    normal = 0
    for nb in neighbors:
        if nb.get("is_normal") or not nb.get("misleaders"):
            normal += 1
        for m in nb.get("misleaders", []):
            key = m if m in MISLEADER_KEYS else normalize_misleader_label(str(m))
            if key in MISLEADER_KEYS:
                counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    n = len(neighbors)
    return {
        "counts":       counts,
        "ranked":       ranked,
        "normal_ratio": round(normal / n, 2) if n else 0.0,
        "n":            n,
    }


def format_rag_hint(query_text: str, k: int = 8, top: int = 6) -> str:
    """RAG 검색 결과를 LLM 프롬프트에 주입할 한국어 힌트 문자열로 포맷."""
    neighbors = query_vector_db(query_text, n_results=min(k, 5))
    prior = misviz_prior_from_neighbors(query_text, k=k)
    if not neighbors and not prior["ranked"]:
        return "(유사 MisViz 사례를 찾지 못함)"

    lines = ["[유사 MisViz 실제 사례 — 검색 결과]"]
    for i, nb in enumerate(neighbors, 1):
        names = ", ".join(
            MISLEADER_TAXONOMY.get(m, {}).get("name", m) for m in nb.get("misleaders", [])
        ) or "정상(오류 없음)"
        lines.append(f"  {i}. [{nb.get('chart_type', '?')}] 오류: {names}")

    if prior["ranked"]:
        lines.append(f"\n[빈도 통계 — 유사 사례 {prior['n']}건 중 자주 나타난 오류]")
        for key, cnt in prior["ranked"][:top]:
            name = MISLEADER_TAXONOMY.get(key, {}).get("name", key)
            lines.append(f"  • {name} [{key}]: {cnt}건")
        lines.append(f"  • 정상(오류 없음) 비율: {prior['normal_ratio'] * 100:.0f}%")
    lines.append(
        "\n※ 위는 참고용 통계일 뿐입니다. 반드시 이미지를 직접 보고 도구로 검증한 뒤,"
        " 실제로 관찰되는 오류만 채택하세요."
    )
    return "\n".join(lines)
