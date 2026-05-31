"""
웹 검색 유틸리티 — DuckDuckGo 기반
"""
from __future__ import annotations

from .config import MISLEADER_TAXONOMY


def _ddg_search(query: str, max_results: int = 3) -> list[dict]:
    """
    DuckDuckGo 검색 (duckduckgo_search 패키지 우선, 없으면 requests 폴백).
    pip install duckduckgo-search
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title":   r.get("title", ""),
                "snippet": r.get("body", "")[:200],
                "url":     r.get("href", ""),
            }
            for r in results
        ]
    except ImportError:
        pass
    except Exception:
        pass

    try:
        import re
        import urllib.parse
        import requests

        encoded = urllib.parse.quote_plus(query)
        resp = requests.get(
            f"https://html.duckduckgo.com/html/?q={encoded}",
            headers={"User-Agent": "Mozilla/5.0 (compatible; ChartErrorDetector/1.0)"},
            timeout=8,
        )
        titles   = re.findall(r'class="result__title"[^>]*>.*?<a[^>]*>(.*?)</a>', resp.text, re.DOTALL)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</span>', resp.text, re.DOTALL)
        urls     = re.findall(r'class="result__url"[^>]*>(.*?)</span>',     resp.text, re.DOTALL)

        def _clean(s: str) -> str:
            return re.sub(r"<[^>]+>", "", s).strip()

        out = []
        for t, s, u in zip(titles, snippets, urls):
            out.append({"title": _clean(t), "snippet": _clean(s), "url": _clean(u)})
            if len(out) >= max_results:
                break
        return out
    except Exception:
        return []


def search_misleader_evidence(misleader_key: str, chart_description: str) -> str:
    """특정 misleader 유형에 대한 실제 사례·증거를 Google/DDG에서 검색."""
    tax = MISLEADER_TAXONOMY.get(misleader_key, {})
    name_en = misleader_key.replace("_", " ")
    name_ko = tax.get("name", "").split("(")[0].strip()

    query_en = f"misleading chart {name_en} example data visualization error"
    query_ko = f"{name_ko} 차트 왜곡 오류 사례"

    results = _ddg_search(query_en, 2) + _ddg_search(query_ko, 1)

    if not results:
        return (
            f"[{tax.get('name', misleader_key)}]\n"
            f"설명: {tax.get('desc', '')}\n"
            f"은폐 방법: {tax.get('hide_tip', '')}"
        )
    lines = [f"• {r['title']}: {r['snippet'][:120]}" for r in results[:3]]
    return f"[{tax.get('name', misleader_key)}]\n" + "\n".join(lines)


def search_chart_context(chart_description: str) -> str:
    """차트 내용 기반으로 관련 뉴스/맥락 검색."""
    keywords = []
    for kw in ["선거", "election", "경제", "economy", "코로나", "covid",
               "총기", "gun", "기후", "climate", "주가", "stock"]:
        if kw in chart_description.lower():
            keywords.append(kw)

    if not keywords:
        keywords = ["misleading chart visualization"]

    query = " ".join(keywords[:3]) + " misleading data visualization"
    results = _ddg_search(query, 2)
    if not results:
        return "맥락 정보 검색 실패 (네트워크 제한)"
    return "\n".join(f"• {r['title']}: {r['snippet'][:100]}" for r in results[:2])
