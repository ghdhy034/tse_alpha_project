@echo off
chcp 65001 > nul
echo 🔧 資料載入器修復測試
echo ================================

REM 啟動虛擬環境
call C:\Users\user\Desktop\environment\stock\Scripts\activate

echo.
echo 🧪 執行資料載入器索引越界修復測試...
echo.

REM 執行修復測試
python tmp_rovodev_dataloader_fix_test_20250115.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 資料載入器修復測試成功！
    echo.
    echo 💡 修復內容:
    echo    - 修復MultiStockDataset.__len__()返回0的問題
    echo    - 修復索引越界問題 (price_frame邊界檢查)
    echo    - 改善NaN處理 (填充而不是丟棄)
    echo    - 擴大測試日期範圍 (確保足夠資料)
    echo    - 添加詳細診斷信息
    echo.
    echo 🚀 建議下一步:
    echo    1. 重新執行階段4: python tmp_rovodev_stage4_training_validation_20250115.py
    echo    2. 或執行完整測試: run_complete_smoke_test_20250115.bat
    echo.
    echo 📋 修復報告: tmp_rovodev_comprehensive_dataloader_fix_20250115.md
    echo.
) else (
    echo.
    echo ❌ 資料載入器修復測試失敗
    echo.
    echo 🔍 可能的問題:
    echo    1. 資料庫連接問題
    echo    2. 日期範圍內無可用資料
    echo    3. FeatureEngine初始化失敗
    echo.
    echo 📋 查看詳細報告: tmp_rovodev_comprehensive_dataloader_fix_20250115.md
    echo.
)

pause