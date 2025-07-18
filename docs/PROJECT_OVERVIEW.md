# TSE Alpha - 台股量化交易系統

## 🎯 專案概述
TSE Alpha 是台股15日隔日沖量化交易系統，整合多資料源、深度學習模型和風險控制。

### 核心特色
- **多資料源**: FinMind + Shioaji API (日線、分鐘線、籌碼面)
- **特徵工程**: 70個特徵 (基本面15 + 其他51 + 帳戶4)
- **深度學習**: Conv1D + Transformer，Optuna優化
- **交易環境**: Gymnasium介面，15日持倉限制
- **資料規模**: 1,200萬+筆記錄，180支股票，5.3年資料

## 📊 當前狀態 (2025-01-10)
**完成度**: 100% (系統完全可用)
**狀態**: 🟢 所有核心功能已完成並通過測試

### ✅ 已完成模組
1. **資料收集** - 1,200萬+筆記錄完成 ✅
2. **資料庫系統** - SQLite抽象層 ✅  
3. **特徵工程** - 72個特徵 (基本面15+其他53+帳戶4) ✅
4. **交易環境** - Gymnasium標準介面 ✅
5. **模型架構** - Conv1D + Transformer ✅
6. **訓練配置** - 完整參數管理系統 ✅
7. **資料載入器** - 支援大規模資料處理 ✅

### 🔧 待開發模組
- **端到端訓練管線** - 優先級: 極高 🚀
- **回測引擎完善** - 優先級: 高

## 🚀 快速開始

### 環境設置
```bash
# 啟動虛擬環境 (Windows)
C:\Users\user\Desktop\environment\stock\Scripts\activate
```

### 驗證系統
```bash
# 快速系統驗證
python tmp_rovodev_quick_test_20250110.py

# 完整系統驗證
python tmp_rovodev_final_verification_20250110.py
```

### 主要執行腳本
```bash
# 資料收集 (已完成)
run_finmind_data_collector.bat    # FinMind資料
run_shioaji_minute_collector.bat  # Shioaji分鐘線

# 測試驗證
run_features_test.bat             # 特徵工程測試
gym_env/run_smoke_test.bat        # 交易環境測試
```

## 📈 資料狀況
- **股票池**: 180支 (A/B/C組各60支)
- **時間範圍**: 2020-03-02 ~ 2025-07-08 (5.3年)
- **總記錄數**: 1,200萬+筆
- **特徵數**: 70個/股票 (基本面15 + 其他51 + 帳戶4)

### 核心資料表
- `candlesticks_min`: 11,467,227筆 (分鐘線)
- `candlesticks_daily`: 233,560筆 (日線)
- `technical_indicators`: 233,560筆 (技術指標)
- `margin_purchase_shortsale`: 232,260筆 (融資融券)
- `institutional_investors_buy_sell`: 230,655筆 (法人進出)
- `financials`: 3,770筆 (財報)
- `financial_per`: 233,329筆 (本益比)
- `monthly_revenue`: 11,409筆 (月營收)

## 📋 重要檔案
- `models/config/training_config.py` - 訓練配置系統
- `models/model_architecture.py` - Conv1D + Transformer模型
- `models/data_loader.py` - 資料載入器
- `gym_env/env.py` - Gymnasium交易環境
- `data_pipeline/features.py` - 特徵工程管線
- `stock_config.py` - 180支股票配置

## 🎯 當前工作重點 (2025-01-10)
**正在進行**: 環境和代理人測試腳本開發

### 立即執行任務
1. **環境核心功能測試** - TSEAlphaEnv基本功能驗證
2. **模型-環境整合測試** - 觀測格式和動作空間對齊驗證
3. **代理人行為測試** - 決策邏輯和風險管理測試
4. **端到端訓練測試** - 完整訓練流程驗證

### 後續開發計畫
1. **創建端到端訓練管線** - 整合所有組件
2. **小規模訓練測試** - 5支股票驗證
3. **完整訓練** - 180支股票生產訓練
4. **回測引擎開發** - Walk-forward驗證

## 📚 文檔結構
- `docs/PROJECT_OVERVIEW.md` - 專案總覽 (本文件)
- `docs/SYSTEM_STATUS.md` - 系統狀態詳情
- `docs/TECHNICAL_GUIDE.md` - 技術實作指南
- `docs/DEVELOPMENT_LOG.md` - 開發歷程記錄

---
**最後更新**: 2025-01-15  
**狀態**: 🟢 系統完全可用，75維特徵配置統一完成