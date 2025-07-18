@echo off
echo === FinMind 歷史資料收集器 (修正版) ===
echo 修正內容:
echo • 自動建立資料表
echo • 修正 API 欄位對應 (max/min/Trading_Volume)
echo • 使用原始 data_fetcher 處理方式
echo 日期範圍: 2020-03-02 ~ 2025-07-08

REM 啟動虛擬環境
call "C:\Users\user\Desktop\environment\stock\Scripts\activate"

REM 執行修正版 FinMind 資料收集
python finmind_data_collector.py

echo.
echo === FinMind 資料收集完成 ===
pause