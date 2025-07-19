@echo off
chcp 65001 > nul
echo 🧪 TSE Alpha 生產級煙霧測試 - 完整執行
echo ================================================

REM 啟動虛擬環境
call C:\Users\user\Desktop\environment\stock\Scripts\activate

echo.
echo 🚀 開始生產級煙霧測試...
echo.

REM 階段1: 基礎驗證
echo ============================================
echo 🎯 階段1: 基礎驗證 (預計15分鐘)
echo ============================================
python tmp_rovodev_stage1_basic_verification_20250115.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 階段1失敗，停止測試
    pause
    exit /b 1
)

echo.
echo ✅ 階段1完成，繼續階段2...
echo.

REM 階段2: 單股票測試
echo ============================================
echo 🎯 階段2: 單股票測試 (預計30分鐘)
echo ============================================
python tmp_rovodev_stage2_single_stock_test_20250115.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 階段2失敗，停止測試
    pause
    exit /b 1
)

echo.
echo ✅ 階段2完成，繼續階段3...
echo.

REM 階段3: 小規模多股票測試
echo ============================================
echo 🎯 階段3: 小規模多股票測試 (預計45分鐘)
echo ============================================
python tmp_rovodev_stage3_multi_stock_test_20250115.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 階段3失敗，停止測試
    pause
    exit /b 1
)

echo.
echo ✅ 階段3完成，繼續階段4...
echo.

REM 階段4: 訓練流程驗證
echo ============================================
echo 🎯 階段4: 訓練流程驗證 (預計60分鐘)
echo ============================================
python tmp_rovodev_stage4_training_validation_20250115.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 階段4失敗，停止測試
    pause
    exit /b 1
)

echo.
echo ✅ 階段4完成，繼續階段5...
echo.

REM 階段5: 穩定性測試
echo ============================================
echo 🎯 階段5: 穩定性測試 (預計30分鐘)
echo ============================================
python tmp_rovodev_stage5_stability_test_20250115.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 階段5失敗
    pause
    exit /b 1
)

echo.
echo ================================================
echo 🎉 生產級煙霧測試全部完成！
echo ================================================
echo.
echo ✅ 所有5個階段測試通過
echo ✅ 66維特徵配置系統驗證完成
echo ✅ 系統準備進入大規模生產訓練
echo.
echo 📋 下一步建議:
echo    1. 執行完整資料載入測試 (180支股票)
echo    2. 開始模型訓練基準測試
echo    3. 準備生產環境部署
echo.
pause