@echo off
echo ============================================================
echo   Khoi chay Song Song: CineBot V3 (8501) & Traditional RAG (8502)
echo ============================================================
echo.

start "CineBot V3 (Port 8501)" cmd /k python -m streamlit run chatbot/app.py --server.port 8501
start "Traditional RAG (Port 8502)" cmd /k python -m streamlit run chatbot/app_traditional.py --server.port 8502

echo Dang mo 2 ung dung...
echo - CineBot V3: http://localhost:8501
echo - Traditional RAG: http://localhost:8502
echo.
pause
