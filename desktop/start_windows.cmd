@echo off
setlocal
set "SIGNALSCOPE_PYTHON=%~dp0..\.local\desktop_env\Scripts\pythonw.exe"
if not exist "%SIGNALSCOPE_PYTHON%" (
  echo SignalScope environment missing. See docs/lab_setup.md.
  pause
  exit /b 1
)
start "" "%SIGNALSCOPE_PYTHON%" "%~dp0run.py"
