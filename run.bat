@echo off
title Bitkub Hybrid AI Bot Runner

echo ========================================
echo  Bitkub Hybrid AI Bot Installer & Runner
echo ========================================
echo.

REM --- Step 1: Check for Python ---
echo [1/3] Checking for Python installation...
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.10 or newer and try again.
    pause
    exit /b
)
echo Python found!
echo.

REM --- Step 2: Install required libraries ---
echo [2/3] Installing required libraries from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install libraries. Please check your internet connection or pip installation.
    pause
    exit /b
)
echo Libraries installed successfully.
echo.

REM --- Step 3: Run the bot ---
echo [3/3] Starting the bot...
echo To stop the bot, press CTRL+C in this window.
echo.
python main.py

echo.
echo Bot has been stopped.
pause
