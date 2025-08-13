@echo off
echo MagicSample - YouTube Playlist Fix Script
echo =========================================
echo.

REM Check if we're in the right directory
if not exist "MagicSample.py" (
    echo Error: MagicSample.py not found!
    echo Please run this script from the MagicSample folder.
    pause
    exit /b 1
)

echo Checking current yt-dlp version...
pip show yt-dlp

echo.
echo Updating yt-dlp to latest version...
pip install --upgrade yt-dlp

echo.
echo Checking FFmpeg installation...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo Warning: FFmpeg not found in PATH
    echo This may cause audio conversion issues
    echo Consider installing FFmpeg from: https://ffmpeg.org/download.html
) else (
    echo FFmpeg is installed and working
)

echo.
echo Testing YouTube playlist functionality...
python test_youtube_playlist.py

echo.
echo If the test shows errors, try these solutions:
echo 1. Update yt-dlp: pip install --upgrade yt-dlp
echo 2. Install FFmpeg: https://ffmpeg.org/download.html
echo 3. Check your internet connection
echo 4. Try a different YouTube playlist URL
echo.
echo To test with your own playlist URL, edit test_youtube_playlist.py
echo and replace the test URLs with your playlist URL.
echo.
pause
