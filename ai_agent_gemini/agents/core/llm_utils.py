"""
LLM 호출, 응답 파싱, 이미지 인코딩 유틸리티
Gemini API (Google) 버전 — gemini-3.5-flash 최적화

변경 사항 (Gemini 3.5 Flash 기준):
  - temperature / top_p / top_k → deprecated, 사용 안 함
  - thinking_level 파라미터 추가 ("minimal" | "low" | "medium" | "high")
  - system_instruction 네이티브 지원 → SystemMessage 직접 사용 가능
  - 컨텍스트 윈도우 100만 토큰, 최대 출력 65,000 토큰
  - SDK: langchain-google-genai (내부적으로 google-genai >= 1.55.0 사용)
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Callable, TypeVar

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import IMAGE_TOKENS, LLM_NAME, THINKING_LEVEL, _RESPONSE_CACHE

_T = TypeVar("_T")

# ─────────────────────────────────────────────────────────
# Rate Limit 방어 — Exponential Backoff + 전역 Throttle
# ─────────────────────────────────────────────────────────
#
# Gemini Free Tier 제한 (2025 기준):
#   gemini-3.5-flash : 10 RPM  →  최소 요청 간격 7초 권장
#   gemini-2.5-flash : 10 RPM  →  동일
#
# 환경변수로 조정 가능:
#   GEMINI_MIN_INTERVAL=7.0   (기본값, Free Tier)
#   GEMINI_MIN_INTERVAL=1.0   (Paid Tier — 속도 우선)
#
# 블로그 권고안 (tenacity 방식) 반영:
#   wait_random_exponential(min=10, max=120), stop_after_attempt(8)

_MIN_REQUEST_INTERVAL: float = float(os.environ.get("GEMINI_MIN_INTERVAL", "7.0"))
_last_request_time: float    = 0.0   # 마지막 실제 요청 시각 (monotonic)

# 429 발생 시 자동 쿨다운 — 연속 오류면 더 길게 쉰다
_consecutive_429:    int   = 0
_COOLDOWN_PER_429:   float = 15.0   # 429 1회당 추가 쿨다운(초)


def _is_rate_limit_error(e: Exception) -> bool:
    """429 / ResourceExhausted 여부 판별."""
    msg = str(e).lower()
    return any(k in msg for k in (
        "429", "resource_exhausted", "resourceexhausted",
        "rate limit", "quota", "too many requests",
        "ratelimitexceeded",
    ))


def _parse_retry_after(e: Exception) -> float:
    """
    예외 메시지에서 Retry-After 대기 시간(초)을 추출한다.
    찾지 못하면 0.0 반환.
    """
    txt = str(e)
    for pat in [
        r"retry[_\-]delay\s*\{[^}]*seconds:\s*(\d+)",
        r"retry[_\s\-]after[:\s]+(\d+)",
        r'"retryDelay"\s*:\s*"(\d+)s"',
        r"retry after (\d+) second",
    ]:
        m = re.search(pat, txt, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return 0.0


def _throttle() -> None:
    """
    전역 요청 간격 제한.
    - 기본: 마지막 요청 후 _MIN_REQUEST_INTERVAL 초 대기
    - 429 연속 발생 시: 추가 쿨다운(_COOLDOWN_PER_429 × 연속횟수) 추가
    """
    global _last_request_time
    now      = time.monotonic()
    gap      = now - _last_request_time
    cooldown = _MIN_REQUEST_INTERVAL + (_consecutive_429 * _COOLDOWN_PER_429)
    if gap < cooldown:
        wait = cooldown - gap
        if wait > 2:
            print(f"     🕐 요청 간격 대기 {wait:.1f}초 (rate limit 방어)...")
        time.sleep(wait)
    _last_request_time = time.monotonic()


def _call_with_backoff(fn: Callable[[], _T], label: str = "LLM") -> _T:
    """
    fn()을 호출하되, 429 오류 시 Exponential Backoff + Jitter 로 재시도.

    블로그 권고안(tenacity wait_random_exponential) 반영:
      - 최대 8회 재시도
      - 대기: Retry-After 헤더 우선, 없으면 min(10 * 2^attempt + jitter, 180)
      - 연속 429 발생 시 _consecutive_429 카운터 누적 → _throttle() 자동 강화
      - 비 429 오류는 즉시 예외 전파
    """
    global _consecutive_429

    max_retries = 8
    base_wait   = 10.0   # 초 (free tier 기준으로 conservative)

    for attempt in range(max_retries + 1):
        _throttle()
        try:
            result = fn()
            # 성공 시 연속 429 카운터 리셋
            _consecutive_429 = 0
            return result
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise  # 429 외 오류는 즉시 전파

            _consecutive_429 += 1  # 연속 429 누적

            if attempt == max_retries:
                _consecutive_429 = 0  # 포기할 때는 리셋
                raise RuntimeError(
                    f"[{label}] 429 재시도 {max_retries}회 초과 → 포기\n"
                    f"원인: {e}\n"
                    f"💡 Free Tier 한도 초과 가능성. GEMINI_MIN_INTERVAL 값을 높여보세요."
                ) from e

            # Retry-After 헤더 파싱 우선, 없으면 Exponential Backoff + jitter
            retry_after = _parse_retry_after(e)
            if retry_after > 0:
                wait = retry_after + random.uniform(1, 5)
            else:
                # tenacity wait_random_exponential(min=10, max=180) 방식
                wait = min(base_wait * (2 ** attempt) + random.uniform(1, 5), 180.0)

            print(f"     ⏳ [{label}] 429 Rate Limit "
                  f"(연속 {_consecutive_429}회) — {wait:.0f}초 대기 후 재시도 "
                  f"({attempt + 1}/{max_retries})...")
            time.sleep(wait)

    raise RuntimeError(f"[{label}] 예상치 못한 backoff 종료")


# ─────────────────────────────────────────────────────────
# 이미지 / 캐시
# ─────────────────────────────────────────────────────────

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def cache_key(prompt: str, image_path: str = "") -> str:
    return hashlib.md5(f"{prompt}{image_path}".encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────
# 이미지 미디어 타입 감지
# ─────────────────────────────────────────────────────────

def _media_type(image_path: str) -> str:
    ext = image_path.lower().rsplit(".", 1)[-1]
    return {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "webp": "image/webp",
        "gif":  "image/gif",
        "bmp":  "image/bmp",
    }.get(ext, "image/png")


# ─────────────────────────────────────────────────────────
# content 추출 헬퍼
# ─────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """LangChain message content → 순수 문자열 변환."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content)


# ─────────────────────────────────────────────────────────
# LLM 인스턴스 생성 헬퍼
# ─────────────────────────────────────────────────────────

def _make_llm(
    max_tokens: int,
    thinking_level: str = THINKING_LEVEL,
) -> ChatGoogleGenerativeAI:
    """
    ChatGoogleGenerativeAI 인스턴스를 생성한다.

    Gemini 3.5 Flash API 주의사항:
      - temperature / top_p / top_k 는 deprecated → 사용 안 함
      - thinking_level 로 추론 품질/속도 조정
          "minimal" → 최저 지연 (단순 응답)
          "low"     → 빠름
          "medium"  → 기본값 (균형)
          "high"    → 최고 품질 (느림·비쌈)
      - GOOGLE_API_KEY 환경변수 필수
    """
    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "  1) 프로젝트 루트의 .env 파일에 GOOGLE_API_KEY=AIzaSy... 를 추가하거나\n"
            "  2) 터미널에서 직접 설정: set GOOGLE_API_KEY=AIzaSy...  (Windows)\n"
            "                         export GOOGLE_API_KEY=AIzaSy...  (Linux/Mac)\n"
            "  Google AI Studio: https://aistudio.google.com/app/apikey"
        )
    return ChatGoogleGenerativeAI(
        model=LLM_NAME,
        google_api_key=api_key,
        max_output_tokens=max_tokens,
        # Gemini 3.5 Flash: thinking_level 로 품질 조정
        # (구 temperature 대신 사용 — LangChain model_kwargs 경유)
        model_kwargs={
            "generation_config": {
                "thinking_config": {"thinking_level": thinking_level},
            },
        },
        # 안전 필터 완화 — 차트 분석 컨텍스트에서 불필요한 차단 방지
        safety_settings={
            "HARM_CATEGORY_HARASSMENT":        "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH":       "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        },
    )


# ─────────────────────────────────────────────────────────
# LLM 호출
# ─────────────────────────────────────────────────────────

def call_vision_llm(
    prompt: str,
    image_path: str,
    system: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.1,    # Gemini 3.5에서 deprecated — 무시됨, 인터페이스 호환용
    use_thinking: bool = False,  # 인터페이스 호환용 (미사용)
) -> str:
    """
    ChatGoogleGenerativeAI Vision 호출 (Gemini 3.5 Flash 멀티모달).

    Gemini 3.5 Flash는 system_instruction 네이티브 지원:
      → SystemMessage를 messages 목록에 직접 포함한다.
    """
    ck = cache_key(prompt + system, image_path)
    if ck in _RESPONSE_CACHE:
        cached = _RESPONSE_CACHE[ck]
        if cached:
            return cached

    # 빠른 분류 호출엔 "low" 수준으로 지연 최소화
    llm = _make_llm(max_tokens, thinking_level="low")

    img_b64  = image_to_base64(image_path)
    mtype    = _media_type(image_path)

    messages = []

    # Gemini 3.5: SystemMessage 네이티브 지원
    if system:
        messages.append(SystemMessage(content=system))

    messages.append(HumanMessage(content=[
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mtype};base64,{img_b64}"},
        },
    ]))

    def _invoke() -> str:
        resp   = llm.invoke(messages)
        result = _extract_text(resp.content)
        if not result:
            raise ValueError("Vision LLM 빈 응답")
        return result

    try:
        result = _call_with_backoff(_invoke, label="Vision LLM")
        _RESPONSE_CACHE[ck] = result
        return result
    except Exception as e:
        raise RuntimeError(
            f"Vision LLM 호출 최종 실패: {e}\n  모델: {LLM_NAME}\n"
            "  → GOOGLE_API_KEY 환경변수를 확인하세요."
        ) from e


def call_text_llm(
    messages: list[dict],
    system: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.1,    # Gemini 3.5에서 deprecated — 무시됨, 인터페이스 호환용
    use_thinking: bool = False,  # 인터페이스 호환용 (미사용)
) -> str:
    """
    ChatGoogleGenerativeAI 텍스트 호출.

    Gemini 3.5 Flash는 system_instruction 네이티브 지원:
      → SystemMessage를 messages 목록 맨 앞에 추가한다.
    """
    ck = cache_key(str(messages) + system)
    if ck in _RESPONSE_CACHE:
        cached = _RESPONSE_CACHE[ck]
        if cached:
            return cached

    llm = _make_llm(max_tokens, thinking_level=THINKING_LEVEL)

    lc_messages = []

    # Gemini 3.5: SystemMessage 네이티브 지원 → 직접 prepend
    if system:
        lc_messages.append(SystemMessage(content=system))

    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    if not lc_messages:
        return ""

    def _invoke() -> str:
        resp = llm.invoke(lc_messages)
        return _extract_text(resp.content)

    try:
        result = _call_with_backoff(_invoke, label="Text LLM")
        if result:
            _RESPONSE_CACHE[ck] = result
        return result
    except Exception as e:
        raise RuntimeError(f"텍스트 LLM 호출 최종 실패: {e}") from e


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
