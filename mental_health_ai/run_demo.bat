@echo off
echo Setting up Mental Health AI Demo...

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

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
mkdir data\eeg\raw data\eeg\processed 2>nul
mkdir data\audio\raw data\audio\processed 2>nul
mkdir data\text\raw data\text\processed 2>nul
mkdir models 2>nul
mkdir results\eeg results\audio results\text results\fusion 2>nul
mkdir visualizations 2>nul

REM Run the quick demo
echo Running quick demo...
python quick_demo.py

REM Generate visualizations
echo Generating visualizations...
python generate_visualizations.py

echo Demo completed successfully!
echo Results are available in the demo_results directory.
echo Visualizations are available in the visualizations directory.
echo.
echo Press any key to exit...
pause > nul
