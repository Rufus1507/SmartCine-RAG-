import streamlit as st
from langchain_openai import ChatOpenAI
from chatbot.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, GEMINI_DEFAULT_KEY, GEMINI_DEFAULT_MODEL

@st.cache_resource
def get_llm_client(provider: str, api_key: str, model_name: str, base_url: str = None) -> ChatOpenAI:
    """
    Nạp và lưu trữ đối tượng ChatOpenAI (Singleton-like) thông qua st.cache_resource.
    Hỗ trợ gọi qua Local OpenAI endpoint hoặc Gemini OpenAI Compatibility endpoint.
    """
    if provider == "Gemini API":
        # Sử dụng API OpenAI compatibility của Google Gemini
        url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        key = api_key if api_key else GEMINI_DEFAULT_KEY
        model = model_name if model_name else GEMINI_DEFAULT_MODEL
    else:
        # Local LLM
        url = base_url if base_url else LLM_BASE_URL
        key = api_key if api_key else LLM_API_KEY
        model = model_name if model_name else LLM_MODEL

    # Trả về đối tượng ChatOpenAI của LangChain
    return ChatOpenAI(
        openai_api_key=key,
        openai_api_base=url,
        model_name=model,
        temperature=0.7,
        max_tokens=2048,
        timeout=30.0,
        max_retries=2
    )
