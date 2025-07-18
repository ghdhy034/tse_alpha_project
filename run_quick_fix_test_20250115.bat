@echo off
chcp 65001 > nul
echo 🔧 快速修復測試 - 驗證錯誤修復
echo ================================

REM 啟動虛擬環境
call C:\Users\user\Desktop\environment\stock\Scripts\activate

echo.
echo 🧪 執行快速修復驗證測試...
echo.

REM 執行快速修復測試
python tmp_rovodev_quick_fix_test_20250115.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 修復驗證成功！可以重新執行階段2測試
    echo.
    echo 🚀 建議下一步:
    echo    1. 執行: run_stage2_single_stock_20250115.bat
    echo    2. 或執行完整測試: run_complete_smoke_test_20250115.bat
    echo.
) else (
    echo.
    echo ❌ 修復驗證失敗，需要進一步調試
    echo.
)

pause