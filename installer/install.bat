@echo off
rem Installs fibsemOS for this user on Windows 10 or newer.
rem Double-click it. It installs uv, puts Python and fibsem[ui] in
rem %USERPROFILE%\fibsem, makes a desktop shortcut and opens the app.
rem Run it again to upgrade. No admin rights are needed.

setlocal
if not defined FIBSEM_ENV set "FIBSEM_ENV=%USERPROFILE%\fibsem"
if not defined FIBSEM_PACKAGE set "FIBSEM_PACKAGE=fibsem[ui]"
set "PYTHON_VERSION=3.11"
rem A fixed uv release, so the installer it runs never changes under it.
set "UV_VERSION=0.11.32"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

where uv >nul 2>nul
if errorlevel 1 (
    echo Installing uv...
    set "UV_NO_MODIFY_PATH=1"
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/%UV_VERSION%/install.ps1 | iex"
    if errorlevel 1 goto failed
)

if not exist "%FIBSEM_ENV%\Scripts\python.exe" (
    echo Creating the environment in %FIBSEM_ENV%...
    uv venv "%FIBSEM_ENV%" --python %PYTHON_VERSION% --seed
    if errorlevel 1 goto failed
)

echo Installing %FIBSEM_PACKAGE%...
uv pip install --python "%FIBSEM_ENV%\Scripts\python.exe" --upgrade "%FIBSEM_PACKAGE%"
if errorlevel 1 goto failed

echo Making the desktop shortcut...
"%FIBSEM_ENV%\Scripts\python.exe" -c "from fibsem.tools.desktop_shortcut import create_desktop_shortcut; print(create_desktop_shortcut(overwrite=True))"
if errorlevel 1 goto failed

echo.
echo fibsemOS is installed. Opening AutoLamella...
start "" "%FIBSEM_ENV%\Scripts\fibsem-autolamella-ui.exe"
pause
exit /b 0

:failed
echo.
echo The install did not finish. See the messages above, or
echo https://www.fibsemos.org/docs/installation/ for the manual steps.
pause
exit /b 1
