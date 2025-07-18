@echo off
echo === Shioaji 分鐘線資料收集器 (更新版) ===
echo 收集5分鐘K線資料
echo 儲存到: candlesticks_min 資料表
echo 正式模式: 2020-03-02 ~ 2025-07-08
echo 測試模式: 2024-12-01 ~ 2024-12-31 (節省流量)

REM 啟動虛擬環境
call "C:\Users\user\Desktop\environment\stock\Scripts\activate"

REM 執行 Shioaji 分鐘線收集
python shioaji_minute_collector.py

echo.
echo === Shioaji 分鐘線收集完成 ===
pause