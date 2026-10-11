import time

import requests

from config import (
    MAX_NEW_TOKENS_GEN,
    ONLINE_LLM_API_KEY,
    ONLINE_LLM_BASE_URL,
    ONLINE_LLM_MAX_RETRIES,
    ONLINE_LLM_MODEL,
    ONLINE_LLM_RETRY_BACKOFF,
    ONLINE_LLM_TIMEOUT,
    TEMPERATURE,
    TOP_P,
)


class OnlineLLMError(RuntimeError):
    pass


def chat_completion(messages, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS_GEN):
    """Call an OpenAI-compatible online chat completion endpoint."""
    if not ONLINE_LLM_API_KEY:
        raise OnlineLLMError("ONLINE_LLM_API_KEY is not set")

    url = ONLINE_LLM_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": ONLINE_LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {ONLINE_LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    transient_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
    last_error = None
    for attempt in range(ONLINE_LLM_MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=(10, ONLINE_LLM_TIMEOUT),
            )
            if response.status_code in transient_statuses and attempt < ONLINE_LLM_MAX_RETRIES:
                retry_after = response.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.replace(".", "", 1).isdigit()
                    else ONLINE_LLM_RETRY_BACKOFF * (2**attempt)
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            content = str(content).strip()
            if not content:
                raise OnlineLLMError("Online LLM returned empty content")
            return content
        except requests.RequestException as exc:
            last_error = exc
            if attempt < ONLINE_LLM_MAX_RETRIES:
                time.sleep(ONLINE_LLM_RETRY_BACKOFF * (2**attempt))
                continue
            raise OnlineLLMError(f"Online LLM request failed: {exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise OnlineLLMError("Unexpected online LLM response format") from exc

    raise OnlineLLMError(f"Online LLM request failed after retries: {last_error}")
