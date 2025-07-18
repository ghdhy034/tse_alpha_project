@echo off
echo === 特徵工程模組測試 ===

cd /d "%~dp0"

echo 啟動虛擬環境...
call "C:\Users\user\Desktop\environment\stock\Scripts\activate"

echo 1. 執行煙霧測試...
python smoke_test.py
if %ERRORLEVEL% neq 0 (
    echo 煙霧測試失敗！
    pause
    exit /b 1
)

echo.
echo 2. 執行特徵工程測試...
python test_features.py
if %ERRORLEVEL% neq 0 (
    echo 特徵工程測試失敗！
    pause
    exit /b 1
)

echo.
echo ✅ 所有測試通過！
pause