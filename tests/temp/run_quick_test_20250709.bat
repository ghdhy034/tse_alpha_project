@echo off
echo === TSE Alpha Quick Test (Fixed Encoding) ===

REM Set UTF-8 encoding
chcp 65001 > nul

REM Activate virtual environment
call "C:\Users\user\Desktop\environment\stock\Scripts\activate"

REM Run quick test
python tmp_rovodev_quick_test.py

echo.
echo === Test Complete ===
pause