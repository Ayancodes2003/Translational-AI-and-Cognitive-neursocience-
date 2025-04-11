@echo off
echo Setting up Minimal Mental Health AI Demo...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install minimal dependencies
echo Installing minimal dependencies...
pip install numpy matplotlib scikit-learn

REM Run the minimal demo
echo Running minimal demo...
python minimal_demo.py

echo Demo completed successfully!
echo Results are available in the minimal_results directory.
echo.
echo Press any key to exit...
pause > nul
