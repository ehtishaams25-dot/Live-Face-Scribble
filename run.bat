@echo off
title Scribble Camera
cd /d "%~dp0"

echo ===================================================
echo           REAL-TIME 3D SCRIBBLE CAMERA
echo ===================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [Setup] Virtual environment not found. Creating with uv...
    uv venv .venv --python 3.11
    if errorlevel 1 (
        python -m venv .venv
    )
    echo [Setup] Installing dependencies...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo [Running] Launching Scribble Camera...
.venv\Scripts\python.exe app\main.py

if errorlevel 1 (
    echo.
    echo [Error] Application encountered an issue.
    pause
)
