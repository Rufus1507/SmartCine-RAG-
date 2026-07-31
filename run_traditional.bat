@echo off
echo ============================================================
echo   Khoi chay Traditional RAG UI/UX (Port 8502)
echo ============================================================
echo.

python -m streamlit run chatbot/app_traditional.py --server.port 8502

pause
