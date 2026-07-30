import streamlit as st
from langchain_openai import ChatOpenAI
from chatbot.config import (
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    GEMINI_DEFAULT_KEY, GEMINI_DEFAULT_MODEL
)

def warmup_ollama_model(llm: ChatOpenAI, provider: str):
    """Gửi 1 request tối thiểu để nạp model vào bộ nhớ trước khi user chat lần đầu."""
    import time
    import logging
    logger = logging.getLogger(__name__)

    if provider not in ["Ollama Server", "Local LLM"]:
        return

    try:
        t0 = time.monotonic()
        warmup_llm = llm.bind(max_tokens=10) if hasattr(llm, "bind") else llm
        warmup_llm.invoke("Xin chào")
        logger.info("[warmup] Ollama model loaded in %.2fs", time.monotonic() - t0)
    except Exception as e:
        logger.warning("[warmup] Failed to warm up model: %s", e)


@st.cache_resource
def get_llm_client(provider: str, api_key: str, model_name: str, base_url: str = None, max_tokens: int = 2048) -> ChatOpenAI:
    """
    Nạp và lưu trữ đối tượng ChatOpenAI (Singleton-like) thông qua st.cache_resource.
    Hỗ trợ gọi qua Local OpenAI endpoint, Ollama Server hoặc Gemini OpenAI Compatibility endpoint.
    """
    extra_body = None
    if provider == "Gemini API":
        # Sử dụng API OpenAI compatibility của Google Gemini
        url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        key = api_key if api_key else GEMINI_DEFAULT_KEY
        model = model_name if model_name else GEMINI_DEFAULT_MODEL
    elif provider == "Ollama Server":
        # Kết nối Ollama Server
        url = base_url if base_url else OLLAMA_BASE_URL
        key = api_key if api_key else "any"
        model = model_name if model_name else OLLAMA_MODEL
        extra_body = {"keep_alive": "30m"}
    else:
        # Local LLM
        url = base_url if base_url else LLM_BASE_URL
        key = api_key if api_key else LLM_API_KEY
        model = model_name if model_name else LLM_MODEL
        extra_body = {"keep_alive": "30m"}

    kwargs = {
        "openai_api_key": key,
        "openai_api_base": url,
        "model_name": model,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "timeout": 300.0,
        "max_retries": 2
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    client = ChatOpenAI(**kwargs)
    object.__setattr__(client, "provider", provider)
    warmup_ollama_model(client, provider)
    return client
