@echo off
chcp 65001 >nul
echo ========================================
echo TSE Alpha 回測引擎測試
echo ========================================
echo 啟動時間: %date% %time%
echo.

REM 激活虛擬環境
echo 🔄 激活虛擬環境...
call "C:\Users\user\Desktop\environment\stock\Scripts\activate"

REM 檢查環境是否成功激活
if errorlevel 1 (
    echo ❌ 虛擬環境激活失敗！
    pause
    exit /b 1
)

echo ✅ 虛擬環境激活成功
echo.

REM 設定輸出檔案名稱（包含時間戳）
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do set mydate=%%d%%b%%c
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a%%b
set timestamp=%mydate%_%mytime%
set output_file=backtest_engine_test_%timestamp%.txt

echo 📄 測試結果將輸出到: %output_file%
echo.

REM 執行回測引擎測試並輸出到檔案
echo 🚀 開始執行回測引擎測試...
python tmp_rovodev_test_fixed_backtest_20250110.py > %output_file% 2>&1

REM 檢查執行結果
if errorlevel 1 (
    echo ❌ 測試執行過程中發生錯誤！
    echo 請查看 %output_file% 了解詳細錯誤信息
) else (
    echo ✅ 測試執行完成！
)

echo.
echo 📋 測試結果摘要:
echo ----------------------------------------
REM 顯示結果檔案的最後幾行
powershell -Command "Get-Content '%output_file%' | Select-Object -Last 10"

echo.
echo 📄 完整測試結果已保存至: %output_file%
echo 🕒 完成時間: %date% %time%
echo.

pause