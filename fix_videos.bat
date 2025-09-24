@echo off
setlocal enabledelayedexpansion

:: 원본 영상이 있는 루트 디렉토리
set "DATADIR=data"

for /r "%DATADIR%" %%f in (*.mp4 *.avi *.mov) do (
    echo Processing: %%f
    ffmpeg -i "%%f" -c copy -movflags +faststart "%%~dpnf_fixed%%~xf"
)
echo All videos processed!
pause
