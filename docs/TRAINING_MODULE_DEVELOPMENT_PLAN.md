# TSE Alpha 訓練模組開發規劃

> **基準文檔**: `training_module_ssot.md` + `References.txt`  
> **創建日期**: 2025-01-15  
> **更新日期**: 2025-01-15 (基於 References.txt 優化)  
> **開發模式**: 👥 一人團隊 + AI 協作  
> **狀態**: 🚀 準備開發  

## 📋 **開發基準與約束**

### **SSOT 規範遵循**
- **股票分割**: 126 訓練 + 27 驗證 + 27 測試 (基於 `stock_split_config.json`)
- **特徵維度**: 75個 (18基本面 + 53其他 + 4帳戶)
- **序列配置**: seq_len=64, stride=32, forward_window=15
- **資料來源**: 8個核心資料表 (基於 `db_structure.json`)

### **優化後技術架構** (基於 References.txt)
```
技術選型優化:
├── 配置管理: Hydra + OmegaConf ✅ (替代自寫 ConfigManager)
├── 訓練框架: PyTorch Lightning ✅ (替代自寫 Trainer/Checkpoint)
├── 資料處理: PyArrow + Memory Mapped ✅ (優化 I/O 性能)
├── 實驗追蹤: Lightning Logger ✅ (TensorBoard + CSV + JSON)
├── RL 整合: Stable-Baselines3 ✅ (PPO baseline policy)
├── 超參數優化: Optuna + Hydra Sweep ✅ (內建整合)
└── 特徵管理: Singleton Registry ✅ (core 套件統一)
```

### **雙硬體環境配置策略**

#### **開發/測試環境** (GTX 1660 Ti 6GB)
```
低配置策略 (煙霧測試/初步驗證):
├── 批次大小: batch=8 (保守 VRAM 使用)
├── 序列長度: seq_len=32 (減半)
├── 股票數量: 10檔 (快速驗證)
├── 訓練輪數: epoch=2-5 (煙霧測試)
├── 混合精度: precision=16 (節省 VRAM)
├── 梯度累積: accumulate_grad_batches=8 (等效 batch=64)
└── 資料子集: 20% 資料 (快速迭代)
```

#### **生產/訓練環境** (RTX 4090 24GB)
```
高配置策略 (完整訓練):
├── 批次大小: batch=128 (充分利用 VRAM)
├── 序列長度: seq_len=64 (完整序列)
├── 股票數量: 180檔 (完整股票池)
├── 訓練輪數: epoch=150-300 (充分訓練)
├── 混合精度: precision=16 (性能優化)
├── 梯度累積: accumulate_grad_batches=1 (無需累積)
├── 完整資料集: 100% 資料 (1,200萬+筆)
└── 超參數搜索: 大規模 Optuna trials
```

### **硬體約束考量** (基於 References.txt)
- **GPU**: GTX 1660 Ti (6GB VRAM) - 需要謹慎的記憶體管理
- **批次大小**: 建議 batch=16, seq_len=64 (避免 OOM)
- **梯度累積**: 啟用 gradient accumulation 模擬大批次
- **資料載入**: 使用 Memory Mapped File + prefetch 優化

---

## 🎯 **開發階段規劃** (優化版)

> **協作模式**: 一人團隊 + AI 助手，高彈性時程調整  
> **技術策略**: 採用成熟框架減少重複造輪子 (基於 References.txt)

## **階段 1: 核心驗證與GPU資源確認** (彈性 3-5 天)

### **1.1 GPU 資源驗證 (優先級最高)**
```python
# 創建: scripts/gpu_memory_test.py
class GPUResourceValidator:
    """GTX 1660 Ti (6GB) 資源驗證"""
    
    def test_dataloader_memory(self):
        """DataLoader 記憶體峰值測試"""
        - batch_size=16, seq_len=64 測試
        - 監控 VRAM 使用峰值
        - 確定最佳 num_workers 設定
    
    def test_model_memory(self):
        """模型記憶體需求測試"""
        - Conv1D + Transformer 前向傳播
        - 梯度計算記憶體需求
        - 確定是否需要 gradient accumulation
    
    def optimize_batch_config(self):
        """優化批次配置"""
        - 找出最大可用 batch_size
        - 設定 gradient accumulation steps
        - 配置 mixed precision training
```

### **1.2 SSOT 相容性驗證器**
```python
# 創建: scripts/validate_ssot_compliance.py
class SSOTValidator:
    """驗證現有實作與 SSOT 規範的相容性"""
    
    def validate_feature_dimensions(self):
        """驗證特徵維度 (修正基於 References.txt)"""
        - 序列特徵: 75個 (18基本面 + 53其他 + 4帳戶)
        - 帳戶特徵: 4個 (作為 env info 單獨注入)
        - 明確區分 sequence features vs. portfolio features
    
    def validate_stock_splits(self):
        """驗證股票分割配置"""
        - 126/27/27 分割比例檢查
        - 無重複股票驗證
        - 資料庫完整性檢查
```

**交付物**:
- [ ] `scripts/gpu_memory_test.py` ⭐ (最高優先級)
- [ ] GPU 資源使用報告
- [ ] 優化的批次配置建議
- [ ] `scripts/validate_ssot_compliance.py`

### **1.3 特徵註冊系統 (Singleton 模式)**
```python
# 創建: core/features_registry.py (Singleton 模式)
class FeaturesRegistry:
    """統一特徵定義 (Singleton，避免重複定義)"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    # 序列特徵 (75個) - 明確分離
    SEQUENCE_FEATURES = {
        'price_volume': ['open', 'high', 'low', 'close', 'volume'],  # 5個
        'technical': [  # 22個技術指標
            'sma_5', 'sma_20', 'sma_60', 'ema_12', 'ema_26', 'ema_50', 
            'macd', 'macd_signal', 'macd_hist', 'rsi_14', 'stoch_k', 'stoch_d',
            'atr', 'adx', 'cci', 'obv', 'keltner_upper', 'keltner_middle', 
            'keltner_lower', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower'
        ],
        'fundamental': [  # 43個基本面特徵
            # 財務指標 (20個)
            'revenue', 'cost_of_goods_sold', 'gross_profit', 'operating_income', 'net_income',
            'total_assets', 'current_assets', 'total_liabilities', 'current_liabilities', 'equity',
            'cash_and_equivalents', 'inventory', 'accounts_receivable', 'accounts_payable', 'debt',
            'eps', 'book_value_per_share', 'dividend_per_share', 'roe', 'roa',
            # 財務比率 (15個)
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'debt_to_equity', 'current_ratio',
            'quick_ratio', 'gross_margin', 'operating_margin', 'net_margin', 'asset_turnover',
            'inventory_turnover', 'receivables_turnover', 'dividend_yield', 'payout_ratio', 'interest_coverage',
            # 成長率 (8個)
            'revenue_growth_yoy', 'eps_growth_yoy', 'asset_growth_yoy', 'equity_growth_yoy',
            'revenue_growth_qoq', 'eps_growth_qoq', 'asset_growth_qoq', 'equity_growth_qoq'
        ]
    }
    
    # 帳戶特徵 (4個) - 作為 env info 單獨注入
    PORTFOLIO_FEATURES = ['nav_change', 'position_ratio', 'unrealized_pnl', 'risk_buffer']
    
    @property
    def total_sequence_features(self):
        """總序列特徵數: 75個"""
        return len(self.SEQUENCE_FEATURES['price_volume']) + \
               len(self.SEQUENCE_FEATURES['technical']) + \
               len(self.SEQUENCE_FEATURES['fundamental'])
```

**交付物**:
- [ ] `core/features_registry.py` (Singleton 模式)
- [ ] 序列特徵 vs 帳戶特徵分離文檔
- [ ] 特徵維度驗證測試

### **1.3 採用成熟框架 (基於 References.txt 建議)**
```python
# 配置管理: 採用 omegaconf + hydra-core
# 安裝: pip install omegaconf hydra-core

# 創建: configs/training_config.yaml
defaults:
  - _self_
  - model: conv1d_transformer
  - data: tse_alpha

# 模型訓練: 採用 PyTorch Lightning
# 安裝: pip install pytorch-lightning

class TSEAlphaLightningModule(pl.LightningModule):
    """基於 Lightning 的訓練模組"""
    
    def __init__(self, config):
        super().__init__()
        self.model = TSEAlphaModel(config.model)
        self.config = config
    
    def training_step(self, batch, batch_idx):
        """自動處理 GPU、梯度累積、檢查點"""
        # Lightning 自動處理複雜的訓練邏輯
    
    def configure_optimizers(self):
        """優化器配置"""
        # 支援學習率調度、多優化器等
```

**交付物**:
- [ ] Hydra 配置系統設置
- [ ] Lightning 模組重構
- [ ] 記憶體優化的 DataLoader

---

## **階段 2: 輕量化訓練管線** (彈性 4-6 天)

### **2.1 Lightning 訓練管線 (簡化版)**
```python
# 創建: training/lightning_trainer.py
@hydra.main(config_path="configs", config_name="training_config")
def train(cfg: DictConfig):
    """Hydra + Lightning 簡化訓練管線"""
    
    # 資料模組 (Lightning DataModule)
    data_module = TSEDataModule(cfg.data)
    
    # 模型模組 (Lightning Module)  
    model = TSEAlphaLightningModule(cfg)
    
    # 訓練器 (Lightning Trainer)
    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16,  # 混合精度節省記憶體
        gradient_clip_val=cfg.training.gradient_clip_norm,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,  # 模擬大批次
        callbacks=[
            EarlyStopping(patience=cfg.training.early_stopping_patience),
            ModelCheckpoint(save_top_k=3),
            LearningRateMonitor()
        ]
    )
    
    # 執行訓練 (Lightning 自動處理所有複雜邏輯)
    trainer.fit(model, data_module)

# RL Agent 整合 (基於 References.txt 建議)
class TSEAlphaRLAgent:
    """RL Agent 與環境介接"""
    
    def __init__(self, model, env):
        self.model = model
        self.env = env
        
    def train_rl(self):
        """RL 訓練冒煙測試"""
        # 確保 Week 3 混合訓練就緒
```

**交付物**:
- [ ] Lightning 訓練管線
- [ ] Hydra 配置文件
- [ ] RL Agent 冒煙測試 ⭐ (為 Week 3 準備)

### **2.2 記憶體優化資料載入**
```python
# 優化: models/data_loader.py (基於 References.txt)
class TSEDataModuleOptimized(pl.LightningDataModule):
    """記憶體優化的 Lightning DataModule"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = 16  # GTX 1660 Ti 安全批次大小
    
    def setup(self, stage=None):
        """使用 Memory Mapped File + Arrow"""
        # pyarrow.dataset + Memory Mapped File
        self.train_dataset = ArrowDataset("data/train.arrow")
        self.val_dataset = ArrowDataset("data/val.arrow")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=2,  # 基於 GPU 測試結果調整
            prefetch_factor=2,  # 預取優化
            pin_memory=True
        )
```

**交付物**:
- [ ] 記憶體優化的 DataModule
- [ ] Arrow 資料格式轉換
- [ ] 批次大小自動調整

### **2.3 Lightning Callbacks (取代自寫檢查點)**
```python
# 創建: training/callbacks.py
class TSEAlphaCallbacks:
    """Lightning Callbacks (省去自寫檢查點系統)"""
    
    @staticmethod
    def get_callbacks(cfg: DictConfig):
        """獲取 Lightning Callbacks"""
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
        
        return [
            ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                filename='tse-alpha-{epoch:02d}-{val_loss:.2f}'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=cfg.training.early_stopping_patience,
                mode='min'
            ),
            LearningRateMonitor(logging_interval='epoch'),
            # 自定義 RL 微調 Callback
            RLFinetuningCallback(cfg.rl) if cfg.training.mode == 'hybrid' else None
        ]

class RLFinetuningCallback(pl.Callback):
    """RL 微調 Callback (支援混合訓練)"""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """監督學習結束後切換到 RL"""
        if trainer.current_epoch == pl_module.cfg.training.supervised_epochs:
            # 切換到 RL 模式
            self.switch_to_rl_mode(trainer, pl_module)
```

**交付物**:
- [ ] `training/callbacks.py` (Lightning Callbacks)
- [ ] RL 微調 Callback 實作
- [ ] Lightning Logger 配置 (TensorBoard + CSV)

---

## **階段 3: 智能超參數優化** (彈性 3-4 天)

### **3.1 Optuna 整合系統**
```python
# 創建: training/optuna_optimizer.py
class OptunaOptimizer:
    """基於 SSOT 的 Optuna 超參數優化"""
    
    def create_study(self):
        """創建 Optuna 研究"""
        - 定義搜索空間 (基於 SSOT)
        - 配置 Pruner
        - 設定存儲後端
    
    def define_search_space(self, trial):
        """定義搜索空間 (基於 References.txt 優化)"""
        # 階梯型搜索避免 VRAM 爆炸
        d_model = trial.suggest_categorical('d_model', [256, 384, 512])  # 降低上限
        seq_len = trial.suggest_categorical('seq_len', [32, 48, 64])     # 動態序列長度
        
        # 根據模型大小調整批次大小
        if d_model >= 512 or seq_len >= 64:
            batch_size = 8
        else:
            batch_size = 16
            
        return {
            'lr': trial.suggest_loguniform('lr', 1e-4, 8e-4),
            'd_model': d_model,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'n_layer': trial.suggest_int('n_layer', 3, 6),  # 降低上限
            'dropout': trial.suggest_uniform('dropout', 0.1, 0.4),
            'alpha_beta': trial.suggest_uniform('alpha_beta', 0.3, 0.7)
        }
    
    def objective(self, trial):
        """Optuna 目標函數 (基於 References.txt 優化)"""
        params = self.define_search_space(trial)
        
        # Mini-epoch 評估 (20% 資料) 節省時間
        trainer = pl.Trainer(
            max_epochs=5,  # 快速評估
            limit_train_batches=0.2,  # 只用 20% 資料
            limit_val_batches=0.5,    # 驗證用 50% 資料
            enable_checkpointing=False,  # 節省 I/O
            logger=False  # 減少日誌開銷
        )
        
        # 返回 proxy metric
        return trainer.callback_metrics['val_loss'].item()
```

**交付物**:
- [ ] `training/optuna_optimizer.py`
- [ ] Optuna 配置模板
- [ ] 超參數優化腳本

### **3.2 Arrow 分片 + Memory Mapped I/O**
```python
# 創建: data/arrow_dataset.py
class ArrowDataset:
    """Arrow 分片資料集 (解決 I/O 瓶頸)"""
    
    def __init__(self, arrow_path: str):
        import pyarrow.dataset as ds
        
        # Memory Mapped File 載入
        self.dataset = ds.dataset(arrow_path, format='arrow')
        self.table = self.dataset.to_table()
    
    def __getitem__(self, idx):
        """Memory Mapped 存取"""
        # 直接從記憶體映射讀取，避免重複載入
        return self.table.slice(idx, 1).to_pandas()
    
    def create_shards(self, train_stocks, val_stocks, test_stocks):
        """創建 Arrow 分片"""
        # 按股票分割創建分片
        train_table = self.filter_by_stocks(train_stocks)
        val_table = self.filter_by_stocks(val_stocks)
        test_table = self.filter_by_stocks(test_stocks)
        
        # 保存分片
        train_table.write('data/train.arrow')
        val_table.write('data/validation.arrow')
        test_table.write('data/test.arrow')
```

**交付物**:
- [ ] `training/experiment_manager.py`
- [ ] 實驗追蹤儀表板
- [ ] 結果比較工具

---

## **階段 4: 核心路徑測試** (彈性 2-3 天)

### **4.1 核心路徑 Smoke Tests (基於 References.txt)**
```python
# 創建: tests/core_tests/ (聚焦核心路徑)
├── test_ssot_validation.py      # SSOT 驗證核心路徑
├── test_dataloader_memory.py    # DataLoader 記憶體安全
├── test_trainer_loop.py         # Trainer Loop 基本功能
├── test_env_compatibility.py    # Env 相容性核心路徑
└── test_gpu_utilization.py      # GPU 資源使用驗證

def test_dataloader_memory():
    """DataLoader 記憶體安全測試"""
    # GTX 1660 Ti 特定測試
    dataloader = create_dataloader(batch_size=16, seq_len=64)
    
    initial_memory = torch.cuda.memory_allocated()
    for batch in dataloader:
        current_memory = torch.cuda.memory_allocated()
        assert current_memory < 5.5 * 1024**3  # < 5.5GB 安全線
        break
        
def test_trainer_loop():
    """Lightning Trainer 核心功能"""
    trainer = pl.Trainer(fast_dev_run=True)  # 快速測試模式
    result = trainer.fit(model, datamodule)
    assert result  # 無例外完成
```

**交付物**:
- [ ] 核心路徑測試套件
- [ ] GPU 記憶體監控
- [ ] Type Hint 靜態檢查

### **4.2 整合測試**
```python
# 創建: tests/integration/
├── test_full_pipeline.py        # 端到端流程測試
├── test_model_env_compat.py     # 模型與環境相容性
├── test_optuna_integration.py   # 超參數優化測試
├── test_checkpoint_recovery.py  # 檢查點恢復測試
└── test_multi_gpu_training.py   # 多GPU訓練測試

def test_full_pipeline():
    """完整管線測試"""
    - 資料載入 → 模型訓練 → 環境評估
    - 檢查點保存 → 模型載入 → 推理測試
    - 配置變更 → 重新訓練 → 結果比較
```

**交付物**:
- [ ] 完整的整合測試套件
- [ ] 性能基準測試
- [ ] 回歸測試腳本

### **4.3 性能和穩定性測試**
```python
# 創建: tests/performance/
├── test_memory_usage.py         # 記憶體使用測試
├── test_training_speed.py       # 訓練速度基準
├── test_data_loading_perf.py    # 資料載入性能
└── test_long_running.py         # 長時間運行穩定性

def test_memory_usage():
    """記憶體使用測試"""
    - 監控訓練過程記憶體使用
    - 檢查記憶體洩漏
    - 驗證 GPU 記憶體管理
```

**交付物**:
- [ ] 性能基準報告
- [ ] 穩定性測試結果
- [ ] 優化建議文檔

---

## **階段 5: 簡化部署** (彈性 1-2 天)

### **5.1 Hydra CLI (自動化)**
```python
# train.py (Hydra 自動 CLI)
@hydra.main(config_path="configs", config_name="training_config")
def train(cfg: DictConfig):
    """Hydra 自動生成 CLI"""
    # 自動支援:
    # python train.py model.d_model=512 training.batch_size=16
    # python train.py --multirun model.d_model=256,512 training.lr=1e-4,3e-4
    
    if cfg.optuna.enabled:
        run_optuna_optimization(cfg)
    else:
        run_single_training(cfg)

# 簡化的工具腳本
# scripts/
├── gpu_test.py              # GPU 記憶體測試
├── data_prep.py             # 資料預處理
├── model_export.py          # 模型匯出
└── docker_dev.py            # Docker 開發環境
```

**交付物**:
- [ ] Hydra CLI 自動化
- [ ] 核心工具腳本 (4個)
- [ ] `Dockerfile.dev` 開發環境

### **5.2 工具腳本集**
```python
# 創建: scripts/
├── make_dataset.py              # Arrow 分片生成
├── validate_setup.py            # 環境驗證
├── benchmark_performance.py     # 性能基準測試
├── export_model.py              # 模型匯出工具
├── analyze_results.py           # 結果分析工具
└── deploy_model.py              # 模型部署工具
```

**交付物**:
- [ ] 完整的工具腳本集
- [ ] 腳本使用文檔
- [ ] 自動化部署腳本

---

## 📅 **優化後開發時程** (基於 References.txt 指導)

> **協作模式**: 一人團隊 + AI 協作，高彈性迭代開發  
> **技術選型**: Hydra + Lightning + PyArrow 優化架構  

### **Sprint 1 (3-5天): 核心基礎 + GPU 驗證**
| 任務 | 優先級 | 預估時間 | 技術選型優化 |
|------|--------|----------|-------------|
| SSOT 相容性驗證器 | P0 | 1天 | 使用 Hydra config validation |
| GPU 資源驗證 (GTX 1660 Ti) | P0 | 0.5天 | batch=16, seq_len=64 smoke test |
| 特徵註冊系統 (Singleton) | P1 | 1天 | 避免重複定義，core 套件統一管理 |
| DataLoader Smoke Test | P0 | 0.5天 | PyArrow + Memory Mapped File |
| Lightning 架構評估 | P1 | 1天 | 替代自寫 Trainer/Checkpoint |

### **Sprint 2 (4-6天): 訓練管線 + RL Agent**
| 任務 | 優先級 | 預估時間 | References.txt 建議 |
|------|--------|----------|-------------------|
| Lightning Trainer 整合 | P0 | 2天 | 省去訓練迴圈重寫 |
| RL Agent 基礎實作 | P0 | 1.5天 | Week 2 必須完成，避免與 Optuna 衝突 |
| TSEAlphaEnv ↔ SB3 介接 | P1 | 1天 | PPO baseline policy |
| Hydra 配置系統 | P1 | 1天 | 替代自寫 ConfigManager |
| 端到端管線冒煙測試 | P0 | 0.5天 | 包含 RL Agent 測試 |

### **Sprint 3 (3-4天): Optuna 優化 + 實驗管理**
| 任務 | 優先級 | 預估時間 | 風險緩解 |
|------|--------|----------|---------|
| Optuna + Hydra 整合 | P0 | 1.5天 | 內建 Sweep 功能 |
| Mini-epoch 評估機制 | P0 | 1天 | 限制 Trial 至 20% 資料，避免 VRAM 爆炸 |
| 階梯型搜索空間 | P1 | 1天 | seq_len, d_model 同時調整批次 |
| Lightning Logger 整合 | P1 | 0.5天 | TensorBoard + CSV + JSON |

### **Sprint 4 (2-3天): 核心路徑測試**
| 任務 | 優先級 | 預估時間 | 測試策略優化 |
|------|--------|----------|-------------|
| 核心路徑測試 | P0 | 1.5天 | SSOT驗證 + DataLoader + Trainer + Env |
| Smoke Tests 套件 | P1 | 1天 | 快速驗證，避免長時間 CI |
| Type Hint 靜態檢查 | P1 | 0.5天 | 補強輔助模組測試覆蓋 |

### **Sprint 5 (1-2天): 部署準備**
| 任務 | 優先級 | 預估時間 | 部署策略 |
|------|--------|----------|---------|
| Dockerfile.dev | P1 | 0.5天 | GPU 驅動 + 依賴環境 |
| CLI 介面 (Hydra) | P0 | 1天 | 內建 override 功能 |
| 文檔整理 | P1 | 0.5天 | 使用指南 + API 文檔 |

---

## 🎯 **成功標準**

### **功能完整性**
- [ ] 支援 SSOT 規範的完整訓練流程
- [ ] 監督學習 + 強化學習 + 混合訓練模式
- [ ] Optuna 超參數優化整合
- [ ] 完整的檢查點和恢復機制

### **性能指標**
- [ ] 訓練速度: >100 樣本/秒
- [ ] 記憶體使用: <16GB (180檔股票)
- [ ] GPU 利用率: >90%
- [ ] 系統穩定性: >24小時連續運行

### **測試覆蓋率**
- [ ] 單元測試: >85%
- [ ] 整合測試: >90%
- [ ] Smoke Tests: 100% 通過
- [ ] 性能測試: 符合基準

### **文檔完整性**
- [ ] API 文檔: 100% 覆蓋
- [ ] 使用指南: 完整
- [ ] 配置說明: 詳細
- [ ] 故障排除: 全面

---

## 🚨 **風險管控** (基於 References.txt 指導)

### **雙環境風險管控**

#### **開發環境風險** (GTX 1660 Ti 6GB)
| 風險 | 緩解策略 | 實作方案 |
|------|----------|----------|
| **VRAM 不足** | 極低配置 + 資料子集 | batch=8, seq_len=32, 10檔股票 |
| **煙霧測試超時** | 快速驗證模式 | epoch=2, 20% 資料 |
| **開發效率低** | 自動配置切換 | 環境檢測 → 自動選擇配置 |

#### **生產環境風險** (RTX 4090 24GB)
| 風險 | 緩解策略 | 實作方案 |
|------|----------|----------|
| **資源浪費** | 最大化利用策略 | batch=128, 完整資料集 |
| **訓練時間過長** | 智能早停 + 檢查點 | 自動保存 + 斷點續訓 |
| **超參數搜索成本** | 分層搜索策略 | 粗搜索 → 精細搜索 |

### **技術整合風險**
| 風險 | 緩解策略 | 實作方案 |
|------|----------|----------|
| **Lightning 遷移複雜度** | 漸進式遷移 | 先評估，再逐步替換現有 Trainer |
| **Hydra 配置衝突** | 統一配置入口 | 單一 config.yaml，避免多處定義 |
| **RL Agent 整合延遲** | 提前到 Sprint 2 | 避免與 Optuna 併發衝突 |

### **協作效率風險**
| 風險 | 緩解策略 | 實作方案 |
|------|----------|----------|
| **Singleton 模組衝突** | 明確約定 | FeaturesRegistry 只能由 core 套件 import |
| **測試覆蓋率過高** | 聚焦核心路徑 | 85%/90% → 核心路徑 100% + 輔助模組 Type Hint |
| **CI 時間過長** | 分層測試策略 | Smoke Tests 快速驗證，Integration Tests nightly |

---

## 🎯 **一人團隊協作策略**

### **彈性開發模式**
```
迭代週期: 2-3天 mini-sprints
├── 每日同步: 30分鐘進度檢視 + 問題討論
├── 技術決策: 即時調整，無需冗長會議
├── 測試策略: 邊開發邊測試，快速驗證
└── 文檔更新: 實時更新，保持同步
```

### **AI 協作分工**
| 開發者負責 | AI 協助 | 協作方式 |
|------------|---------|----------|
| 架構設計決策 | 程式碼實作建議 | 討論 → 實作 → 驗證 |
| 業務邏輯驗證 | 技術實作細節 | 需求 → 程式碼 → 測試 |
| 系統整合測試 | 單元測試生成 | 手動 → 自動 → 驗證 |
| 性能調優 | 程式碼優化建議 | 分析 → 優化 → 基準 |

### **立即行動建議** (優先級排序)

#### **🚀 Phase 1: 快速驗證 (1-2天)**
1. **雙硬體環境驗證** - 確認兩種配置可用性
   ```bash
   # GTX 1660 Ti 煙霧測試
   python scripts/smoke_test_gtx1660ti.py
   
   # RTX 4090 環境檢查 (如果可用)
   python scripts/full_training_rtx4090.py --mode supervised --epochs 1 --force
   ```

2. **硬體配置系統測試** - 自動檢測與配置切換
   ```python
   # 測試硬體檢測
   python configs/hardware_configs.py
   
   # 測試配置切換
   python -c "from configs.hardware_configs import ConfigManager; print(ConfigManager.get_auto_config())"
   ```

3. **SSOT 相容性檢查** - 驗證現有實作
   ```python
   # 創建快速驗證腳本
   scripts/quick_ssot_check.py
   ```

#### **🔧 Phase 2: 技術選型驗證 (2-3天)**
1. **Lightning 遷移評估** - 評估現有 Trainer 遷移成本
2. **Hydra 配置整合** - 替換現有配置系統
3. **PyArrow 性能測試** - 驗證 I/O 優化效果

#### **⚡ Phase 3: 核心功能實作 (3-4天)**
1. **RL Agent 基礎實作** - SB3 + TSEAlphaEnv 整合
2. **Mini-epoch 機制** - Optuna Trial 加速
3. **端到端冒煙測試** - 完整流程驗證

---

## 📚 **參考文檔**

- **基準規範**: `training_module_ssot.md`
- **實作指導**: `References.txt` ⭐ (新增)
- **股票配置**: `stock_split_config.json`
- **資料結構**: `db_structure.json`
- **系統架構**: `docs/TECHNICAL_GUIDE.md`
- **開發歷程**: `docs/DEVELOPMENT_LOG.md`

---

## 🚀 **下一步行動決策**

### **立即可執行選項**

#### **選項 A: 雙硬體驗證路線** ⚡ (推薦)
- **時間**: 1-2天
- **目標**: 確認雙硬體環境可行性
- **GTX 1660 Ti**: 煙霧測試 + 低配置驗證
- **RTX 4090**: 高配置測試 + 性能基準
- **交付**: 硬體配置系統 + 自動切換機制

#### **選項 B: 開發環境優化路線** 🔧  
- **時間**: 2-3天
- **目標**: 完成開發工具鏈
- **重點**: Lightning遷移 + Hydra配置 + 煙霧測試套件
- **交付**: 開發效率工具 + 快速迭代能力

#### **選項 C: 生產訓練路線** 💪
- **時間**: 3-5天  
- **目標**: 完整訓練系統
- **重點**: 大規模訓練 + Optuna優化 + 檢查點管理
- **交付**: 生產級訓練管線 + 超參數優化

### **建議開始點**
**推薦選項 A** - 雙硬體驗證路線，確保兩種環境都能正常工作

---

## 📋 **協作檢查清單**

### **每日同步檢查**
- [ ] 當前任務進度確認
- [ ] 技術問題討論解決  
- [ ] 下一步計畫調整
- [ ] 風險點識別緩解

### **Sprint 完成檢查**
- [ ] 功能驗證通過
- [ ] 程式碼品質檢查
- [ ] 文檔同步更新
- [ ] 下個 Sprint 規劃

---

**文檔版本**: v2.0 (References.txt 優化版)  
**最後更新**: 2025-01-15  
**協作模式**: 一人團隊 + AI 高彈性迭代  
**下次審查**: 每個 mini-sprint 完成後