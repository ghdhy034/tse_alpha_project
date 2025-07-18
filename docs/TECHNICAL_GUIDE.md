# TSE Alpha 技術實作指南

## 🏗️ 系統架構

### 核心組件
```
TSE Alpha System
├── 資料層 (Data Layer)
│   ├── market_data_collector/    # 資料收集
│   ├── data_pipeline/           # 特徵工程
│   └── db_structure.json        # 資料庫結構
├── 模型層 (Model Layer)
│   ├── models/config/           # 配置管理
│   ├── models/model_architecture.py  # 模型定義
│   ├── models/data_loader.py    # 資料載入
│   └── models/trainer.py        # 訓練器
├── 環境層 (Environment Layer)
│   ├── gym_env/env.py          # 交易環境
│   ├── gym_env/reward.py       # 獎勵函數
│   └── backtest/               # 回測引擎
└── 配置層 (Configuration Layer)
    ├── stock_config.py         # 股票配置
    └── References.txt          # 實作指導
```

## 💾 資料架構

### 資料庫設計
基於 `db_structure.json` 的實際結構：

#### 核心資料表
1. **candlesticks_daily** (233,560筆) - 日線資料
2. **candlesticks_min** (11,467,227筆) - 分鐘線資料
3. **technical_indicators** (233,560筆) - 技術指標
4. **financials** (3,770筆) - 財報資料
5. **monthly_revenue** (11,409筆) - 月營收
6. **financial_per** (233,329筆) - 本益比資料
7. **margin_purchase_shortsale** (232,260筆) - 融資融券
8. **institutional_investors_buy_sell** (230,655筆) - 法人進出

### 特徵工程
```python
# 特徵維度配置 (models/config/training_config.py)
price_features: int = 22        # OHLCV(5) + 技術指標(17)
fundamental_features: int = 43  # 基本面特徵
account_features: int = 4       # 帳戶狀態

# 技術指標 (17個)
technical_indicators = [
    'sma_5', 'sma_20', 'sma_60',                    # 移動平均
    'ema_12', 'ema_26', 'ema_50',                   # 指數移動平均
    'macd', 'macd_signal', 'macd_hist',             # MACD
    'keltner_upper', 'keltner_middle', 'keltner_lower',  # Keltner通道
    'bollinger_upper', 'bollinger_middle', 'bollinger_lower',  # 布林通道
    'pct_b', 'bandwidth'                            # 布林指標
]

# 基本面特徵 (43個)
fundamental_features = [
    # financials表 (17個) - 移除重複的pe_ratio
    # monthly_revenue表 (1個)
    # financial_per表 (3個)
    # margin_purchase_shortsale表 (13個)
    # institutional_investors_buy_sell表 (10個)
]
```

## 🤖 模型架構

### Conv1D + Transformer 設計
```python
# models/model_architecture.py
class TSEAlphaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        # 價格框架編碼器 (Conv1D + Transformer)
        self.price_encoder = PriceFrameEncoder(config)
        
        # 基本面特徵編碼器
        self.fundamental_encoder = nn.Sequential(...)
        
        # 帳戶狀態編碼器
        self.account_encoder = nn.Sequential(...)
        
        # 跨股票注意力
        self.cross_stock_attention = nn.MultiheadAttention(...)
        
        # 決策頭
        self.stock_selector = nn.Sequential(...)
        self.position_sizer = nn.Sequential(...)
        self.value_head = nn.Sequential(...)
        self.risk_head = nn.Sequential(...)
```

### 模型配置
```python
# 小型配置 (快速測試)
ModelConfig(
    price_frame_shape=(5, 32, 22),
    fundamental_dim=43,
    account_dim=4,
    hidden_dim=128,
    num_heads=4,
    num_layers=2
)

# 生產配置 (完整訓練)
ModelConfig(
    price_frame_shape=(180, 64, 22),
    fundamental_dim=43,
    account_dim=4,
    hidden_dim=256,
    num_heads=8,
    num_layers=4
)
```

## 🎮 交易環境

### Gymnasium 介面
```python
# gym_env/env.py
class TSEAlphaEnv(gym.Env):
    def __init__(self, 
                 symbols: List[str],
                 start_date: str,
                 end_date: str,
                 initial_cash: float = 1000000.0,
                 max_holding_days: int = 15):
        
        # 動作空間: (股票索引, 交易數量)
        self.action_space = spaces.Tuple((
            spaces.Discrete(len(symbols)),
            spaces.Box(low=-300, high=300, shape=(1,), dtype=np.int16)
        ))
        
        # 觀測空間: 與模型輸入格式對應
        self.observation_space = spaces.Dict({
            'price_frame': spaces.Box(...),
            'fundamental': spaces.Box(...),
            'account': spaces.Box(...)
        })
```

### 風險控制
```python
# 15日持倉限制
def update_holding_days(self):
    for symbol in self.positions:
        self.positions[symbol]['days_held'] += 1

# 強制平倉超時持倉
timeout_positions = self.get_timeout_positions(15)
for symbol in timeout_positions:
    self.execute_trade(symbol, -pos['qty'], price)
```

## 🔧 配置管理

### 訓練配置
```python
# 使用統一的訓練配置
from models.config.training_config import TrainingConfig

# 創建配置實例
config = TrainingConfig()
# 所有參數已在 models/config/training_config.py 中統一定義
# 包含完整的資料配置、訓練參數、股票配置等
```

### 股票分割
```python
# stock_config.py
STOCK_SPLITS = {
    'train': 126支股票,      # 70%
    'validation': 27支股票,  # 15%
    'test': 27支股票        # 15%
}
```

## 📊 資料載入

### 批次處理
```python
# 使用統一的資料載入器
from models.data_loader import TSEDataLoader, DataConfig

# 創建資料載入器
config = DataConfig()
loader = TSEDataLoader(config)
train_loader, val_loader, test_loader = loader.get_dataloaders()
```

### 觀測格式
```python
# 與 Gym 環境相容的觀測格式
observation = {
    'price_frame': torch.tensor(shape=(batch, n_stocks, seq_len, 22)),
    'fundamental': torch.tensor(shape=(batch, 43)),
    'account': torch.tensor(shape=(batch, 4))
}
```

## 🧪 測試框架

### 測試層級
1. **單元測試** - 個別組件功能
2. **整合測試** - 組件間互動
3. **端到端測試** - 完整流程
4. **性能測試** - 壓力和穩定性

### 關鍵測試腳本
```bash
# 快速驗證
python tmp_rovodev_quick_test_20250110.py

# 完整驗證
python tmp_rovodev_final_verification_20250110.py

# 環境測試
gym_env/run_smoke_test.bat

# 特徵測試
data_pipeline/run_features_test.bat
```

## 🚀 部署指南

### 環境需求
```bash
# Python 環境
Python 3.8+
PyTorch 1.9+
Gymnasium 0.26+
Optuna 3.0+

# 硬體需求
RAM: 16GB+ (推薦32GB)
GPU: RTX 3060+ (訓練用)
Storage: 100GB+ SSD
```

### 快速啟動
```bash
# 1. 啟動環境
C:\Users\user\Desktop\environment\stock\Scripts\activate

# 2. 驗證系統
python tmp_rovodev_quick_test_20250110.py

# 3. 開始訓練 (待開發)
python train_pipeline.py
```

## 📋 開發規範

### 程式碼風格
- 遵循 PEP8 規範
- 使用型別註解
- 詳細文檔字符串
- 單元測試覆蓋

### 檔案命名
- 臨時測試檔案: `tmp_rovodev_*_YYYYMMDD.py`
- 配置檔案: `*_config.py`
- 測試檔案: `test_*.py`
- 批次檔案: `run_*.bat`

### Git 工作流
- 功能分支開發
- 程式碼審查
- 自動化測試
- 持續整合

---
**文件版本**: 1.1  
**最後更新**: 2025-01-15 (75維特徵配置統一)  
**維護者**: TSE Alpha 開發團隊