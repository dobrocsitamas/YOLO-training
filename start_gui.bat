@echo off
chcp 65001 >nul
setlocal

:: ── Frame Extractor GUI – indítófájl ──────────────────────────────────────
:: Dupla kattintással futtatható a YOLO-training mappából.
:: Python 3.10+ és a requirements.txt csomagok szükségesek.

set "SCRIPT_DIR=%~dp0"
set "GUI_SCRIPT=%SCRIPT_DIR%scripts\extract_frames_gui.py"

:: ── Python keresés ──────────────────────────────────────────────────────────
:: 1. Virtuális környezet (venv / .venv a projektmappában)
if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe"
    goto :run
)
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
    goto :run
)

:: 2. Conda-aktivált környezet (ha már aktív terminálból indítják)
if defined CONDA_PREFIX (
    if exist "%CONDA_PREFIX%\python.exe" (
        set "PYTHON=%CONDA_PREFIX%\python.exe"
        goto :run
    )
)

:: 3. Rendszer Python (PATH-ban lévő python)
where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON=python"
    goto :run
)

:: 4. python3 parancs
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON=python3"
    goto :run
)

:: Nem található Python
echo.
echo  [HIBA] Nem talalhato Python telepites!
echo.
echo  Telepitsd a Pythont (3.10+) innen: https://www.python.org/downloads/
echo  Vagy aktivald a virtualis kornyezetet, majd futtasd ujra ezt a .bat-ot.
echo.
pause
exit /b 1

:run
:: ── GUI indítása ─────────────────────────────────────────────────────────────
echo  Python: %PYTHON%
echo  Script: %GUI_SCRIPT%
echo.

"%PYTHON%" "%GUI_SCRIPT%"

if %errorlevel% neq 0 (
    echo.
    echo  [HIBA] A program hibával állt le (hibakód: %errorlevel%).
    echo  Ellenőrizd, hogy a requirements.txt csomagok telepítve vannak-e:
    echo.
    echo    pip install -r "%SCRIPT_DIR%requirements.txt"
    echo.
    pause
)

endlocal
