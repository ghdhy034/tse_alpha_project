@echo off
chcp 65001 >nul
echo ========================================
echo TSE Alpha 單一模組測試
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

REM 詢問用戶要測試哪個模組
echo 請選擇要測試的模組:
echo 1. model     - 模型架構
echo 2. data      - 資料載入器
echo 3. trainer   - 訓練器
echo 4. env       - 交易環境
echo 5. backtest  - 回測引擎
echo 6. features  - 特徵工程
echo 7. all       - 所有模組
echo.

set /p choice="請輸入選項 (1-7): "

REM 根據選擇設定模組名稱
if "%choice%"=="1" set module=model
if "%choice%"=="2" set module=data
if "%choice%"=="3" set module=trainer
if "%choice%"=="4" set module=env
if "%choice%"=="5" set module=backtest
if "%choice%"=="6" set module=features
if "%choice%"=="7" set module=all

REM 檢查是否為有效選擇
if "%module%"=="" (
    echo ❌ 無效的選擇！
    pause
    exit /b 1
)

echo 📋 選擇的模組: %module%
echo.

REM 設定輸出檔案名稱（包含時間戳和模組名稱）
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do set mydate=%%d%%b%%c
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a%%b
set timestamp=%mydate%_%mytime%
set output_file=single_module_test_%module%_%timestamp%.txt

echo 📄 測試結果將輸出到: %output_file%
echo.

REM 執行單一模組測試並輸出到檔案
echo 🚀 開始執行 %module% 模組測試...
python tmp_rovodev_single_module_test_20250110.py --module %module% > %output_file% 2>&1

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