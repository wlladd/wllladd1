@echo off
REM ---------------------------------------------------------------------
REM Batch script to prepare and push Python project to GitHub
REM (removes venv311 from tracking, updates .gitignore, handles index.lock)
REM ---------------------------------------------------------------------

REM 1) Set variables (change these values as needed):
set "PROJECT_DIR=D:\AAA"
set "VENV_FOLDER=venv311"
set "GIT_REMOTE_URL=https://github.com/wlladd/wllladd1.git"

REM ---------------------------------------------------------------------
REM 2) Change to the project directory
cd /d "%PROJECT_DIR%"
if NOT "%CD%"=="%PROJECT_DIR%" (
    echo ERROR: Failed to change directory to "%PROJECT_DIR%"
    pause
    exit /b 1
)
echo Current directory: %CD%

REM ---------------------------------------------------------------------
REM 3) Activate the virtual environment
if exist "%VENV_FOLDER%\Scripts\activate.bat" (
    echo Activating virtual environment "%VENV_FOLDER%"...
    call "%VENV_FOLDER%\Scripts\activate.bat"
) else (
    echo ERROR: "%VENV_FOLDER%\Scripts\activate.bat" not found.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------
REM 4) Generate or update requirements.txt
echo Generating requirements.txt...
pip freeze > requirements.txt
if errorlevel 1 (
    echo ERROR: pip freeze failed.
    pause
    exit /b 1
)
echo requirements.txt generated, size:
for %%I in (requirements.txt) do echo     %%~zI bytes

REM ---------------------------------------------------------------------
REM 5) Create or update .gitignore to exclude virtual environment and common patterns
echo Creating or updating .gitignore...
(
    echo # Byte-compiled / optimized / DLL files
    echo __pycache__/
    echo *.py[cod]
    echo *.so
    echo 
    echo # Virtual environment folder
    echo %VENV_FOLDER%/
    echo 
    echo # VSCode / IDE settings
    echo .vscode/
    echo .idea/
    echo 
    echo # Database and log files
    echo data/project.db
    echo logs/
    echo 
    echo # Environment files
    echo *.env
) > .gitignore
echo .gitignore contents:
type .gitignore

REM ---------------------------------------------------------------------
REM 6) Initialize Git repository if not already initialized
if exist ".git\HEAD" (
    echo Git repository already initialized.
) else (
    echo Initializing Git repository...
    git init
    if errorlevel 1 (
        echo ERROR: git init failed.
        pause
        exit /b 1
    )
)

REM ---------------------------------------------------------------------
REM 7) Add remote origin if not already present
git remote get-url origin >nul 2>&1
if %errorlevel% equ 0 (
    echo Remote "origin" already exists.
) else (
    echo Adding remote origin "%GIT_REMOTE_URL%"...
    git remote add origin %GIT_REMOTE_URL%
    if errorlevel 1 (
        echo ERROR: git remote add failed.
        pause
        exit /b 1
    )
)

REM ---------------------------------------------------------------------
REM 8) Remove any stale .git\index.lock before staging files
if exist ".git\index.lock" (
    echo Found ".git\index.lock". Attempting to delete...
    del /f ".git\index.lock" 2>nul
    if exist ".git\index.lock" (
        echo ERROR: Unable to delete ".git\index.lock". Make sure no other Git process is running.
        echo Manually delete "%PROJECT_DIR%\.git\index.lock" then rerun this script.
        pause
        exit /b 1
    ) else (
        echo ".git\index.lock" deleted successfully.
    )
)

REM ---------------------------------------------------------------------
REM 9) Remove the virtual environment folder from Git tracking if it was previously added
git ls-files --error-unmatch "%VENV_FOLDER%/*" >nul 2>&1
if %errorlevel% equ 0 (
    echo Virtual environment "%VENV_FOLDER%" is tracked. Removing from index...
    git rm -r --cached "%VENV_FOLDER%"
    if errorlevel 1 (
        echo ERROR: git rm --cached "%VENV_FOLDER%" failed.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment "%VENV_FOLDER%" is not currently tracked.
)

REM ---------------------------------------------------------------------
REM 10) Stage all changes except venv311
echo Staging files...
git add .
if errorlevel 1 (
    echo ERROR: git add failed.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------
REM 11) Commit changes (if any)
echo Committing changes...
git commit -m "Update project: remove venv, update gitignore, freeze dependencies" 2>nul

REM ---------------------------------------------------------------------
REM 12) Rename or create branch main (GitHub default)
git branch -M main 2>nul

REM ---------------------------------------------------------------------
REM 13) Push to GitHub
echo Pushing to origin/main...
git push -u origin main
if errorlevel 1 (
    echo ERROR: git push failed. Check remote URL and credentials.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------
REM 14) Deactivate virtual environment
echo Deactivating virtual environment...
deactivate

echo All steps completed successfully.
pause
