import requests

from config import (
    MAX_NEW_TOKENS_GEN,
    ONLINE_LLM_API_KEY,
    ONLINE_LLM_BASE_URL,
    ONLINE_LLM_MODEL,
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

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=ONLINE_LLM_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        if not content:
            raise OnlineLLMError("Online LLM returned empty content")
        return content
    except requests.RequestException as exc:
        raise OnlineLLMError(f"Online LLM request failed: {exc}") from exc
    except (KeyError, IndexError, TypeError) as exc:
        raise OnlineLLMError("Unexpected online LLM response format") from exc
