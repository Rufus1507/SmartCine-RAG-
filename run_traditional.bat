@echo off
echo ============================================================
echo   Khoi chay Traditional RAG UI/UX (Port 8502)
echo ============================================================
echo.

where uv >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    uv run streamlit run chatbot/app_traditional.py --server.port 8502
) else (
    streamlit run chatbot/app_traditional.py --server.port 8502
)

pause
