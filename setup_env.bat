@echo on
REM ============================================================
REM  setup_env.bat — create venv311 (Python 3.11) and install packages
REM ============================================================

echo.
echo ============================================================
echo   Step 1) Checking for Python 3.11...
echo ============================================================
py -3.11 --version
if ERRORLEVEL 1 (
    echo.
    echo [ERROR] Python 3.11 not found. Please install Python 3.11 and add it to PATH.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 2) Removing old venv311 if it exists...
echo ============================================================
if exist venv311 (
    echo    Deleting old venv311...
    rmdir /s /q venv311
    if ERRORLEVEL 1 (
        echo [ERROR] Cannot delete venv311.
        pause
        exit /b 1
    )
)

echo.
echo ============================================================
echo   Step 3) Creating new venv311...
echo ============================================================
py -3.11 -m venv venv311
if ERRORLEVEL 1 (
    echo [ERROR] Cannot create venv311.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 4) Activating venv311...
echo ============================================================
call venv311\Scripts\activate.bat
if ERRORLEVEL 1 (
    echo [ERROR] Cannot activate venv311.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 5) Upgrading pip, setuptools, wheel...
echo ============================================================
python -m pip install --upgrade pip setuptools wheel
if ERRORLEVEL 1 (
    echo [ERROR] Cannot upgrade pip, setuptools, or wheel.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 6) Installing numpy==1.25.2...
echo ============================================================
pip install numpy==1.25.2
if ERRORLEVEL 1 (
    echo [ERROR] Cannot install numpy.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 7) Installing pandas==1.5.3...
echo ============================================================
pip install pandas==1.5.3
if ERRORLEVEL 1 (
    echo [ERROR] Cannot install pandas.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 8) Installing streamlit==1.23.1...
echo ============================================================
pip install streamlit==1.23.1
if ERRORLEVEL 1 (
    echo [ERROR] Cannot install streamlit.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 9) Installing pandas_ta (latest available)...
echo ============================================================
pip install pandas_ta
if ERRORLEVEL 1 (
    echo [ERROR] Cannot install pandas_ta.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 10) Installing SQLAlchemy==2.0.18...
echo ============================================================
pip install SQLAlchemy==2.0.18
if ERRORLEVEL 1 (
    echo [ERROR] Cannot install SQLAlchemy.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 11) Installing lightgbm==3.3.5...
echo ============================================================
pip install lightgbm==3.3.5
if ERRORLEVEL 1 (
    echo [ERROR] Cannot install lightgbm.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 12) Installing tensorflow==2.12.0...
echo ============================================================
pip install tensorflow==2.12.0
if ERRORLEVEL 1 (
    echo [ERROR] Cannot install tensorflow.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 13) Installing catboost (latest available)...
echo ============================================================
pip install catboost
if ERRORLEVEL 1 (
    echo [ERROR] Cannot install catboost.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 14) Installing scikit-learn==1.3.0...
echo ============================================================
pip install scikit-learn==1.3.0
if ERRORLEVEL 1 (
    echo [ERROR] Cannot install scikit-learn.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 15) Installing requests==2.31.0...
echo ============================================================
pip install requests==2.31.0
if ERRORLEVEL 1 (
    echo [ERROR] Cannot install requests.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 16) Installing PyYAML==6.0...
echo ============================================================
pip install PyYAML==6.0
if ERRORLEVEL 1 (
    echo [ERROR] Cannot install PyYAML.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Step 17) Installing yfinance==0.2.26...
echo ============================================================
pip install yfinance==0.2.26
if ERRORLEVEL 1 (
    echo [ERROR] Cannot install yfinance.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Deactivating venv311...
echo ============================================================
deactivate

echo.
echo ============================================================
echo   All packages installed successfully.
echo   To activate the environment, run:
echo     venv311\Scripts\activate.bat
echo ============================================================
pause
