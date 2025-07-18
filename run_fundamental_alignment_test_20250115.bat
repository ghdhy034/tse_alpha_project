@echo off
chcp 65001 > nul
echo 🧪 基本面資料智能對齊測試
echo ================================

REM 啟動虛擬環境
call C:\Users\user\Desktop\environment\stock\Scripts\activate

echo.
echo 🔍 測試基本面資料的智能時間對齊邏輯...
echo.

REM 執行基本面對齊測試
python tmp_rovodev_improved_fundamental_test_20250115.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 基本面對齊測試成功！
    echo.
    echo 💡 改善內容:
    echo    - 月營收資料: 智能找尋最近的過去資料
    echo    - 財報資料: 考慮季度發布延遲
    echo    - 時效性檢查: 過舊資料自動處理
    echo    - 覆蓋率統計: 監控資料完整性
    echo.
    echo 🚀 建議下一步:
    echo    1. 執行: run_quick_fix_test_20250115.bat
    echo    2. 重新測試: run_stage2_single_stock_20250115.bat
    echo.
) else (
    echo.
    echo ❌ 基本面對齊測試失敗，需要進一步調整
    echo.
)

pause