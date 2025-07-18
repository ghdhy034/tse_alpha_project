@echo off
echo 執行 fetch_minute.py Smoke Test...
cd /d "%~dp0"

echo 啟動虛擬環境...
call "C:\Users\user\Desktop\environment\stock\Scripts\activate"

python smoke_test.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Smoke Test 通過！
) else (
    echo.
    echo ❌ Smoke Test 失敗！
)
pause