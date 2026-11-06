@echo off
REM Paneli baslatir. Cift tikla. Tarayici kendiliginden acilir.
cd /d "%~dp0"
streamlit run kod/panel.py
pause