# TSE Alpha 開發交接文檔

> **交接時間**: 2025-01-15  
> **前任 AI**: Claude (Anthropic)  
> **專案狀態**: 訓練模組開發規劃完成，準備實作階段  

## 🎯 **專案概況**

### **專案定位**
TSE Alpha 是台股15日隔日沖量化交易系統，整合多資料源、深度學習模型和風險控制。

### **核心技術架構**
- **模型**: Conv1D + Transformer
- **環境**: Gymnasium 標準交易環境
- **資料**: 1,200萬+筆記錄，180支股票，5.3年資料
- **特徵**: 74個 (27價量 + 43基本面 + 4帳戶)

### **當前完成度**
- ✅ **系統架構**: 100% 完成，所有核心模組已實作
- ✅ **重複清理**: 100% 完成，統一實作架構
- ✅ **開發規劃**: 100% 完成，詳細執行計畫
- 🔄 **訓練模組**: 規劃完成，準備實作

---

## 📋 **最近完成的重要工作**

### **1. 重複實作清理** ✅
**問題**: 系統中存在多個重複的 TrainingConfig 和 DataLoader 實作  
**解決**: 
- 刪除 `models/trainer.py` 中重複的 TrainingConfig
- 刪除 `models/data/data_loader.py` 和 `models/data/dataset.py` 舊版本
- 更新 `docs/TECHNICAL_GUIDE.md` 使用統一實作
- 保留統一版本: `models/config/training_config.py`, `models/data_loader.py`

### **2. 基於 References.txt 的規劃優化** ✅
**重要發現**:
- 建議使用 Hydra + Lightning 替代自寫組件
- GTX 1660 Ti (6GB) 需要特殊的低配置策略
- RL Agent 需要提前到 Week 2 實作
- 測試覆蓋率目標需要調整為核心路徑優先

### **3. 雙硬體環境配置系統** ✅
**創建檔案**:
- `configs/hardware_configs.py` - 自動檢測硬體並選擇配置
- `scripts/smoke_test_gtx1660ti.py` - GTX 1660 Ti 煙霧測試
- `scripts/full_training_rtx4090.py` - RTX 4090 完整訓練
- `scripts/run_dual_hardware_test.py` - 自動雙硬體測試

**配置策略**:
- **GTX 1660 Ti**: batch=8, seq_len=32, 10檔股票 (開發/測試)
- **RTX 4090**: batch=128, seq_len=64, 180檔股票 (生產/訓練)

---

## 🚀 **下一步工作重點**

### **立即優先任務** (P0)
1. **雙硬體環境驗證** - 執行 `python scripts/run_dual_hardware_test.py`
2. **SSOT 相容性檢查** - 驗證現有實作與 `training_module_ssot.md` 一致性
3. **Lightning 遷移評估** - 評估現有 `ModelTrainer` 遷移到 PyTorch Lightning 的成本

### **短期目標** (1-2週)
1. **Hydra 配置系統整合** - 替代現有配置管理
2. **RL Agent 基礎實作** - SB3 + TSEAlphaEnv 整合
3. **端到端煙霧測試** - 完整流程快速驗證

### **中期目標** (2-4週)
1. **Optuna 超參數優化** - 大規模參數搜索
2. **生產級訓練管線** - 180檔股票完整訓練
3. **性能基準建立** - 建立評估標準

---

## 📚 **重要文檔指引**

### **核心規範文檔**
- **`training_module_ssot.md`** ⭐ - 訓練模組開發的唯一標準
- **`References.txt`** ⭐ - 重要實作指導和技術建議
- **`docs/TRAINING_MODULE_DEVELOPMENT_PLAN.md`** ⭐ - 完整開發規劃

### **配置文檔**
- **`stock_split_config.json`** - 股票分割配置 (126/27/27)
- **`db_structure.json`** - 資料庫結構定義
- **`stock_config.py`** - 統一股票配置

### **技術文檔**
- **`docs/TECHNICAL_GUIDE.md`** - 系統架構指南
- **`docs/SYSTEM_STATUS.md`** - 系統狀態報告
- **`docs/PROJECT_OVERVIEW.md`** - 專案總覽

---

## ⚠️ **重要注意事項**

### **技術約束**
1. **硬體限制**: GTX 1660 Ti 只有 6GB VRAM，必須使用低配置
2. **資料規模**: 1,200萬+筆記錄，需要 PyArrow + Memory Mapped 優化
3. **SSOT 遵循**: 所有實作必須嚴格遵循 `training_module_ssot.md` 規範

### **架構原則**
1. **統一實作**: 避免重複定義，使用 core 套件統一管理
2. **配置分離**: 開發/生產環境配置完全分離
3. **自動檢測**: 硬體配置自動檢測和切換

### **開發模式**
1. **一人團隊**: 高彈性迭代，2-3天 mini-sprints
2. **AI 協作**: 開發者負責架構決策，AI 協助實作細節
3. **雙環境**: GTX 1660 Ti 開發，RTX 4090 生產

---

## 🔧 **技術選型建議**

### **已確定的技術選型**
- **配置管理**: Hydra + OmegaConf (替代自寫 ConfigManager)
- **訓練框架**: PyTorch Lightning (替代自寫 Trainer)
- **資料處理**: PyArrow + Memory Mapped (優化 I/O)
- **RL 整合**: Stable-Baselines3 (PPO baseline)
- **超參數優化**: Optuna + Hydra Sweep

### **待評估的選型**
- **實驗追蹤**: Lightning Logger vs. MLflow
- **分散式訓練**: 是否需要 Ray/Horovod
- **模型服務**: 部署方案選擇

---

## 🐛 **已知問題和風險**

### **技術風險**
1. **VRAM 不足**: GTX 1660 Ti 可能無法運行完整模型
2. **Lightning 遷移**: 現有 Trainer 遷移可能比預期複雜
3. **RL 整合**: SB3 與 TSEAlphaEnv 介接需要仔細測試

### **進度風險**
1. **Optuna 搜索時間**: 大規模搜索可能需要數天
2. **資料載入性能**: 1,200萬筆記錄可能有 I/O 瓶頸
3. **測試覆蓋**: 85%/90% 目標可能過高

---

## 💬 **給繼任者的建議**

### **工作方式**
1. **先驗證再開發**: 務必先執行雙硬體測試確認環境可用
2. **小步快跑**: 使用 GTX 1660 Ti 快速迭代，RTX 4090 驗證
3. **文檔同步**: 每個功能完成後立即更新相關文檔

### **技術重點**
1. **嚴格遵循 SSOT**: `training_module_ssot.md` 是不可違背的標準
2. **重視 References.txt**: 包含重要的實作經驗和坑點
3. **配置優先**: 硬體配置系統是整個架構的基礎

### **溝通要點**
1. **開發者是一人團隊**: 需要高效協作，避免冗長討論
2. **實用主義**: 優先解決實際問題，避免過度設計
3. **彈性調整**: 根據實際情況隨時調整計畫

---

## 📞 **快速啟動指令**

### **環境檢查**
```bash
# 檢查硬體配置
python configs/hardware_configs.py

# 執行雙硬體測試
python scripts/run_dual_hardware_test.py
```

### **煙霧測試**
```bash
# GTX 1660 Ti 煙霧測試
python scripts/smoke_test_gtx1660ti.py

# RTX 4090 快速驗證
python scripts/full_training_rtx4090.py --mode supervised --epochs 1 --force
```

### **開發驗證**
```bash
# 檢查統一實作
python scripts/validate_cleanup.py

# SSOT 相容性檢查 (待實作)
python scripts/validate_ssot_compliance.py
```

---

## 🎯 **成功標準**

### **短期目標** (1週內)
- [ ] 雙硬體環境測試 100% 通過
- [ ] SSOT 相容性驗證完成
- [ ] Lightning 遷移方案確定

### **中期目標** (2週內)
- [ ] 端到端訓練管線可用
- [ ] RL Agent 基礎功能完成
- [ ] Optuna 整合測試通過

### **長期目標** (1個月內)
- [ ] 180檔股票完整訓練成功
- [ ] 大規模超參數優化完成
- [ ] 生產部署方案確定

---

**祝繼任者工作順利！記住：先驗證環境，再開始開發。有問題時優先查看 SSOT 和 References.txt。**

---

**交接完成時間**: 2025-01-15  
**下次更新**: 重大里程碑達成時