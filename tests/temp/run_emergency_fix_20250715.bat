@echo off
echo ========================================
echo RTX 4090 緊急記憶體修復執行
echo ========================================

echo.
echo 步驟0: 啟動虛擬環境
call "C:\Users\user\Desktop\environment\stock\Scripts\activate"
echo 虛擬環境已啟動

echo.
echo 步驟1: 檢查PyTorch版本
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

echo.
echo 步驟2: 強制清理GPU記憶體
echo 正在執行強制記憶體清理...
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect(); print('第一次清理完成')"
python -c "import torch; torch.cuda.synchronize(); torch.cuda.empty_cache(); print('同步清理完成')"
python -c "import torch; torch.cuda.reset_peak_memory_stats(); print('重置記憶體統計')"

echo.
echo 步驟3: 設置激進記憶體管理
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
echo 已設置: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32

echo.
echo 步驟4: 檢查記憶體狀態
python -c "import torch; print(f'GPU記憶體: 總計={torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB'); print(f'已分配={torch.cuda.memory_allocated(0)/1024**3:.2f}GB'); print(f'已保留={torch.cuda.memory_reserved(0)/1024**3:.2f}GB')" 2>nul

echo.
echo 步驟5: 執行緊急測試 (極小配置)
echo 使用5支股票，批次大小2，2個epoch
echo 預計需要2-3分鐘...
python step0_quick_validation.py --smoke --config rtx4090_emergency_config.yaml

echo.
echo 步驟6: 最終記憶體檢查
python -c "import torch; print(f'最終GPU記憶體: 已分配={torch.cuda.memory_allocated(0)/1024**3:.2f}GB, 已保留={torch.cuda.memory_reserved(0)/1024**3:.2f}GB, 峰值={torch.cuda.max_memory_allocated(0)/1024**3:.2f}GB')" 2>nul

echo.
echo ========================================
echo 緊急修復執行完成
echo ========================================
echo 如果此測試成功，我們可以逐步增加配置
pause