@echo off
echo =======================================================
echo     DocSecure - AI Document Fraud Detection System
echo =======================================================
echo.
echo Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b %errorlevel%
)
echo.
echo =======================================================
echo Starting the DocSecure Server...
echo The model will take ~30-60 seconds to load.
echo.
echo Once started, open: http://127.0.0.1:5000
echo Keep this window open.
echo =======================================================
echo.
start "" http://127.0.0.1:5000
python app.py
pause
