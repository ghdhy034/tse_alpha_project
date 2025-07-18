@echo off
chcp 65001 > nul
echo 🚀 執行階段3: 小規模多股票測試
echo ================================

REM 啟動虛擬環境
call C:\Users\user\Desktop\environment\stock\Scripts\activate

REM 執行階段3測試
python tmp_rovodev_stage3_multi_stock_test_20250115.py

echo.
echo ✅ 階段3測試完成
pause