@echo off
echo === API 連接測試 ===
cd /d "%~dp0"

echo 啟動虛擬環境...
call "C:\Users\user\Desktop\environment\stock\Scripts\activate"

echo 執行 API 測試...
python tmp_rovodev_api_test.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ API 測試完成！
) else (
    echo.
    echo ❌ API 測試失敗！
)
pause