@echo off
chcp 65001 > nul
echo 🚀 執行階段1: 基礎驗證測試
echo ================================

REM 啟動虛擬環境
call C:\Users\user\Desktop\environment\stock\Scripts\activate

REM 執行階段1測試
python tmp_rovodev_stage1_basic_verification_20250115.py

echo.
echo ✅ 階段1測試完成
pause