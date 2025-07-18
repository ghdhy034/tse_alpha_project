@echo off
echo 測試資料庫抽象層...
cd /d "%~dp0"

echo 啟動虛擬環境...
call "C:\Users\user\Desktop\environment\stock\Scripts\activate"

python test_db_simple.py
pause