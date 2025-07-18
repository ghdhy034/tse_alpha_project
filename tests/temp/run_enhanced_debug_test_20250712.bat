@echo off
echo ========================================
echo TSE Alpha 增強調試測試
echo ========================================
echo.

REM 啟動虛擬環境
echo 啟動虛擬環境...
call C:\Users\user\Desktop\environment\stock\Scripts\activate

REM 執行增強調試測試
echo.
echo 執行增強調試測試...
python tmp_rovodev_final_test_20250110.py

echo.
echo ========================================
echo 測試完成，請檢查輸出結果
echo ========================================
pause