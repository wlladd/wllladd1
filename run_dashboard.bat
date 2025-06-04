@echo off
REM =============================================================
REM run_dashboard.bat — activate venv311, install TA-Lib if needed, then start Streamlit
REM =============================================================

REM --- 1) Change to project folder ---
cd /d D:\aaa
if ERRORLEVEL 1 (
    echo [ERROR] Cannot change directory to D:\aaa
    pause
    exit /b 1
)

REM --- 2) Activate virtual environment venv311 ---
call venv311\Scripts\activate.bat
if ERRORLEVEL 1 (
    echo [ERROR] Cannot activate venv311.
    pause
    exit /b 1
)



REM --- 4) Run Streamlit app ---
echo.
echo ============================================================
echo  Starting Streamlit...
echo ============================================================
streamlit run src\streamlit_app.py

REM --- 5) Keep window open after Streamlit exits ---
pause
