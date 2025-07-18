@echo off
echo === TSE Alpha 完整功能測試 ===

REM 啟動虛擬環境
call "C:\Users\user\Desktop\environment\stock\Scripts\activate"

REM 執行完整測試
python tmp_rovodev_complete_test.py

echo.
echo === 測試完成 ===
pause