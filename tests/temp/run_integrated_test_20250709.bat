@echo off
echo === TSE Alpha 整合測試 ===

REM 啟動虛擬環境
call "C:\Users\user\Desktop\environment\stock\Scripts\activate"

REM 執行整合測試
python tmp_rovodev_integrated_test.py

echo.
echo === 測試完成 ===
pause