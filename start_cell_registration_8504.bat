@echo off
cd /d "%~dp0"
echo Starting Cell Registration Prototype on http://127.0.0.1:8504
echo Working directory: %CD%
echo.
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
python -m streamlit run app.py --server.port 8504 --server.address 127.0.0.1 --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false
echo.
echo Streamlit exited with code %ERRORLEVEL%.
pause
