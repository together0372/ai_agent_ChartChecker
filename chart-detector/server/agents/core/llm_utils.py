"""
LLM 호출, 응답 파싱, 이미지 인코딩 유틸리티
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .config import IMAGE_TOKENS, LLM_NAME, _RESPONSE_CACHE


# ─────────────────────────────────────────────────────────
# 이미지 / 캐시
# ─────────────────────────────────────────────────────────

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def cache_key(prompt: str, image_path: str = "") -> str:
    return hashlib.md5(f"{prompt}{image_path}".encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────
# LLM 호출
# ─────────────────────────────────────────────────────────

def call_vision_llm(
    prompt: str,
    image_path: str,
    system: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.1,
    use_thinking: bool = True,
) -> str:
    """ChatOllama Vision 호출 (Gemma4 Thinking Mode 지원)."""
    ck = cache_key(prompt + system, image_path)
    if ck in _RESPONSE_CACHE:
        return _RESPONSE_CACHE[ck]

    llm = ChatOllama(
        model=LLM_NAME,
        num_predict=max_tokens,
        temperature=temperature,
        num_image_tokens=IMAGE_TOKENS,
    )

    messages = []
    if system:
        messages.append(SystemMessage(content=system))

    img_b64 = image_to_base64(image_path)
    messages.append(HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
    ]))

    try:
        resp = llm.invoke(messages)
        result = resp.content or ""
        if "<channel|>" in result:
            result = result.split("<channel|>")[-1].strip()
        _RESPONSE_CACHE[ck] = result
        return result
    except Exception as e:
        raise RuntimeError(
            f"Vision LLM 호출 실패: {e}\n  모델: {LLM_NAME}\n  → ollama pull {LLM_NAME}"
        ) from e


def call_text_llm(
    messages: list[dict],
    system: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.1,
    use_thinking: bool = True,
) -> str:
    """ChatOllama 텍스트 호출 (Thinking Mode 지원)."""
    ck = cache_key(str(messages) + system)
    if ck in _RESPONSE_CACHE:
        return _RESPONSE_CACHE[ck]

    llm = ChatOllama(
        model=LLM_NAME,
        num_predict=max_tokens,
        temperature=temperature,
    )

    lc_messages = []
    if system:
        lc_messages.append(SystemMessage(content=system))
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    try:
        resp = llm.invoke(lc_messages)
        result = resp.content or ""
        if "<channel|>" in result:
            result = result.split("<channel|>")[-1].strip()
        _RESPONSE_CACHE[ck] = result
        return result
    except Exception as e:
        raise RuntimeError(f"텍스트 LLM 호출 실패: {e}") from e


# ─────────────────────────────────────────────────────────
# 파싱
# ─────────────────────────────────────────────────────────

def parse_json(text: str) -> dict:
    """LLM 응답에서 JSON 추출. 항상 dict 반환."""
    for pat in [
        r"```json\s*(.*?)\s*```",
        r"```\s*(\{[\s\S]*?\})\s*```",
        r"(\{[\s\S]*\})",
        r"(\[[\s\S]*\])",
    ]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if isinstance(parsed, list):
                    return {"_list": parsed}
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return {}


# ─────────────────────────────────────────────────────────
# 대화 로그
# ─────────────────────────────────────────────────────────

def log_conversation(conversation: list[dict], agent: str, msg: str) -> list[dict]:
    conversation = list(conversation)
    conversation.append({
        "agent":     agent,
        "message":   msg,
        "timestamp": datetime.now().isoformat(),
    })
    return conversation
