@echo off
echo 執行 Gym Environment Smoke Test...
cd /d "%~dp0"

echo 啟動虛擬環境...
call "C:\Users\user\Desktop\environment\stock\Scripts\activate"

python test_smoke.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Gym Environment Smoke Test 通過！
) else (
    echo.
    echo ❌ Gym Environment Smoke Test 失敗！
)
pause