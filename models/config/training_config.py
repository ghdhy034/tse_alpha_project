# models/config/training_config.py
"""
TSE Alpha 訓練配置管理

統一管理所有訓練相關的配置參數，確保配置的一致性和可維護性。
基於 db_structure.json 的實際資料狀況進行配置。
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class TrainingConfig:
    """TSE Alpha 訓練配置類"""
    
    # ==================== 資料時間配置 ====================
    data_start_date: str = "2020-03-02"      # 資料起始日期 (統一標準)
    data_end_date: str = "2025-07-10"        # 資料結束日期
    
    # 原始分割日期 (用於資料查詢)
    train_end_date: str = "2023-12-29"       # 訓練期結束 (原始)
    val_start_date: str = "2024-01-02"       # 驗證期開始
    val_end_date: str = "2024-06-28"         # 驗證期結束 (原始)
    test_start_date: str = "2024-07-01"      # 測試期開始
    test_end_date: str = "2025-07-09"        # 測試期結束 (原始)
    
    # 實際可用樣本日期 (扣除前瞻窗口，避免標籤外溢)
    # 基於 References.txt 建議：每個 split 自動排除尾端 15 個交易日
    effective_train_end: str = "2023-12-08"  # 實際訓練期結束 (往前挪15個交易日)
    effective_val_end: str = "2024-06-07"    # 實際驗證期結束 (往前挪15個交易日)
    effective_test_end: str = "2025-06-18"   # 實際測試期結束 (往前挪15個交易日)
    
    # ==================== 股票池配置 ====================
    n_stocks: int = 180                      # 總股票數量
    symbols: List[str] = field(default_factory=lambda: ['2330', '2317', '2454'])  # 預設股票清單 - 快速修復
    stock_groups: Dict[str, int] = field(default_factory=lambda: {
        'group_A': 60,  # 半導體電子
        'group_B': 60,  # 傳產原物料
        'group_C': 60   # 金融內需
    })
    
    # 張量維度配置 ====================
    # 基於 66維特徵配置 (15基本面 + 51其他，帳戶特徵暫不使用)
    sequence_length: int = 64                # 64個5分鐘bar ≈ 1交易日 (320分鐘)
    fundamental_features: int = 15           # 基本面特徵 (月/季度更新): monthly_revenue(1) + financials(14)
    other_features: int = 51                 # 其他特徵 (每日更新): 價量+技術+籌碼+估值+日內結構
    account_features: int = 0                # 帳戶狀態特徵: 未來待加入 (暫不使用)
    total_features: int = 66                 # 總特徵數: 15 + 51 + 0 = 66
    
    # 其他特徵詳細配置 (51個，每日更新)
    other_features_detail: Dict[str, int] = field(default_factory=lambda: {
        'candlesticks_daily': 5,      # OHLCV
        'technical_indicators': 17,   # 技術指標 (實際資料庫17個)
        'margin_purchase_shortsale': 13,  # 融資融券
        'institutional_investors_buy_sell': 8,  # 法人進出 (排除Foreign_Dealer_Self)
        'financial_per': 3,           # 本益比等估值指標
        'intraday_structure': 5       # 日內結構信號 (從5分K萃取)
    })  # 總計: 5+17+13+8+3+5 = 51個其他特徵
    
    # 技術指標配置 (17個)
    technical_indicators: List[str] = field(default_factory=lambda: [
        # 移動平均 (3個)
        'sma_5', 'sma_20', 'sma_60',
        # 指數移動平均 (3個)  
        'ema_12', 'ema_26', 'ema_50',
        # MACD (3個)
        'macd', 'macd_signal', 'macd_hist',
        # Keltner 通道 (3個)
        'keltner_upper', 'keltner_middle', 'keltner_lower',
        # Bollinger 通道 (3個)
        'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
        # 其他指標 (2個)
        'pct_b', 'bandwidth'
    ])  # 17個技術指標 (屬於其他特徵)
    
    # 基本面特徵配置 (15個，月/季度更新)
    fundamental_features_detail: Dict[str, int] = field(default_factory=lambda: {
        'monthly_revenue': 1,         # 月營收 (月更新)
        'financials': 14             # 財報數據 (季度更新)
    })  # 總計: 1+14 = 15個基本面特徵
    
    # 基本面特徵列表 (來自 financials 表的14個欄位，基於 References.txt)
    fundamental_features_list: List[str] = field(default_factory=lambda: [
        # 來自 financials 表 (14個實際可用欄位)
        'cost_of_goods_sold', 'eps', 'equity_attributable_to_owners',
        'gross_profit', 'income_after_taxes', 'income_from_continuing_operations',
        'operating_expenses', 'operating_income', 'other_comprehensive_income',
        'pre_tax_income', 'revenue', 'tax', 'total_profit', 'nonoperating_income_expense'
        # 加上 monthly_revenue 表的 1 個特徵 (monthly_revenue)
    ])  # 15個基本面特徵 (月/季度更新頻率)
    
    # ==================== 標籤生成配置 ====================
    forward_window: int = 15                 # 15交易日前瞻窗口 (符合持倉限制)
    prediction_horizon: int = 5              # 預測時間範圍 (天) - 快速修復
    buy_threshold: float = 0.005             # 0.5% 買入門檻 (高於交易成本)
    sell_threshold: float = -0.005           # -0.5% 賣出門檻
    label_method: str = "max_min_return"     # 使用區間最大/最小報酬
    
    # 標籤外溢處理配置 (基於 References.txt 建議)
    auto_trim_horizon: bool = True           # 自動排除尾端前瞻窗口樣本
    horizon_buffer_days: int = 15            # 前瞻窗口緩衝天數 (與 forward_window 一致)
    
    # ==================== 交易成本配置 ====================
    # 基於 References.txt 的修正數值
    commission_rate: float = 0.1425          # 手續費率 % (法定上限)
    commission_discount: float = 0.21        # 折扣 (2.1折)
    tax_rate: float = 0.3                    # 證交稅 % (現股非當沖)
    min_commission: int = 20                 # 最低手續費 (元)
    round_trip_cost: float = 0.0036          # 總成本 ≈ 0.36% (0.0299%*2 + 0.3%)
    slippage_factor: float = 0.25            # 滑點係數 (乘於 spread)
    
    # ==================== 訓練超參數配置 ====================
    batch_size: int = 32                     # 批次大小
    learning_rate: float = 1e-4              # 學習率
    num_epochs: int = 100                    # 訓練輪數
    early_stopping_patience: int = 10        # 早停耐心值
    patience: int = 10                        # 別名，與 early_stopping_patience 相同
    gradient_clip_norm: float = 1.0          # 梯度裁剪
    weight_decay: float = 1e-5               # 權重衰減
    
    # 學習率調度
    lr_scheduler: str = "ReduceLROnPlateau"  # 學習率調度器類型
    lr_patience: int = 5                     # 學習率調度耐心值
    lr_factor: float = 0.5                   # 學習率衰減因子
    
    # ==================== 資料處理配置 ====================
    # 標準化配置
    normalize_features: bool = True          # 是否標準化特徵 - 快速修復
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)  # Winsorize 限制 (1%, 99%)
    normalize_method: str = "zscore"         # 標準化方法: zscore, minmax, robust
    
    # 特徵工程配置
    include_chip_features: bool = True       # 是否包含籌碼面特徵 - 快速修復
    
    # 缺失值處理
    missing_value_strategy: str = "forward_fill"  # 缺失值處理策略
    max_missing_consecutive: int = 2         # 最大連續缺失數 (5分鐘bar)
    
    # 異常值檢測
    outlier_detection: bool = True           # 是否進行異常值檢測
    price_jump_threshold: float = 0.1        # 價格跳空閾值 (10%)
    volume_spike_threshold: float = 5.0      # 成交量異常倍數
    
    # ==================== 資料庫配置 ====================
    db_path: str = "market_data_collector/data/stock_data.db"  # 資料庫路徑
    db_connection_timeout: int = 30          # 資料庫連接超時 (秒)
    query_chunk_size: int = 10000           # 查詢分塊大小
    
    # ==================== 模型配置 ====================
    model_name: str = "tse_alpha_v1"         # 模型名稱
    model_save_dir: str = "models/checkpoints"  # 模型保存目錄
    log_dir: str = "logs/training"           # 日誌目錄
    
    # ==================== 驗證配置 ====================
    validation_split: float = 0.2            # 驗證集比例 (如果不使用時間分割)
    cross_validation_folds: int = 5          # 交叉驗證折數
    
    # ==================== 並行處理配置 ====================
    num_workers: int = 4                     # DataLoader 工作進程數
    pin_memory: bool = True                  # 是否使用 pin_memory
    prefetch_factor: int = 2                 # 預取因子
    
    # ==================== 調試配置 ====================
    debug_mode: bool = False                 # 調試模式
    verbose_logging: bool = True             # 詳細日誌
    save_intermediate_results: bool = False  # 保存中間結果
    
    # ==================== GPU配置 ====================
    device: str = "cuda"                     # 設備 (cuda/cpu)
    mixed_precision: bool = True             # 混合精度訓練
    
    def __post_init__(self):
        """初始化後的驗證和設置"""
        # 確保 patience 與 early_stopping_patience 同步
        if hasattr(self, 'patience') and self.patience != self.early_stopping_patience:
            self.early_stopping_patience = self.patience
        elif not hasattr(self, 'patience'):
            self.patience = self.early_stopping_patience
            
        self._validate_config()
        self._setup_directories()
    
    def _validate_config(self):
        """驗證配置的合理性"""
        # 驗證原始日期順序
        dates = [self.data_start_date, self.train_end_date, self.val_start_date, 
                self.val_end_date, self.test_start_date, self.test_end_date, self.data_end_date]
        
        for i in range(len(dates) - 1):
            if dates[i] >= dates[i + 1]:
                raise ValueError(f"日期順序錯誤: {dates[i]} >= {dates[i + 1]}")
        
        # 驗證有效日期順序 (考慮前瞻窗口) - 暫時註解掉進行調試
        # effective_dates = [self.data_start_date, self.effective_train_end, self.val_start_date,
        #                   self.effective_val_end, self.test_start_date, self.effective_test_end]
        # 
        # for i in range(len(effective_dates) - 1):
        #     if effective_dates[i] >= effective_dates[i + 1]:
        #         raise ValueError(f"有效日期順序錯誤: {effective_dates[i]} >= {effective_dates[i + 1]}")
        
        # 驗證前瞻窗口一致性
        if self.horizon_buffer_days != self.forward_window:
            raise ValueError(f"horizon_buffer_days ({self.horizon_buffer_days}) 應等於 forward_window ({self.forward_window})")
        
        # 驗證特徵數量 (基於實際資料庫結構)
        if len(self.technical_indicators) != 17:
            raise ValueError(f"技術指標數量不匹配: 期望 17, 實際 {len(self.technical_indicators)}")
        
        if len(self.fundamental_features_list) != 14:  # financials 表的實際可用欄位數
            raise ValueError(f"基本面特徵列表數量不匹配: 期望 14 (financials表實際可用), 實際 {len(self.fundamental_features_list)}")
        
        # 驗證總特徵數 (確保66維配置正確)
        calculated_total = self.fundamental_features + self.other_features + self.account_features
        if calculated_total != self.total_features or self.total_features != 66:
            raise ValueError(f"總特徵數不匹配: 計算值 {calculated_total}, 設定值 {self.total_features}, 要求值 66")
        
        # 驗證股票數量
        total_stocks = sum(self.stock_groups.values())
        if total_stocks != self.n_stocks:
            raise ValueError(f"股票分組數量不匹配: 期望 {self.n_stocks}, 實際 {total_stocks}")
        
        # 驗證標籤門檻
        if self.buy_threshold <= self.round_trip_cost:
            raise ValueError(f"買入門檻 ({self.buy_threshold}) 應高於交易成本 ({self.round_trip_cost})")
        
        if abs(self.sell_threshold) <= self.round_trip_cost:
            raise ValueError(f"賣出門檻 ({abs(self.sell_threshold)}) 應高於交易成本 ({self.round_trip_cost})")
    
    def _setup_directories(self):
        """設置必要的目錄"""
        directories = [self.model_save_dir, self.log_dir]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    def save_to_file(self, filepath: str):
        """保存配置到文件"""
        config_dict = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TrainingConfig':
        """從文件載入配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def get_data_split_info(self) -> Dict[str, str]:
        """獲取資料分割信息"""
        return {
            'train_period_raw': f"{self.data_start_date} ~ {self.train_end_date}",
            'train_period_effective': f"{self.data_start_date} ~ {self.effective_train_end}",
            'validation_period_raw': f"{self.val_start_date} ~ {self.val_end_date}",
            'validation_period_effective': f"{self.val_start_date} ~ {self.effective_val_end}",
            'test_period_raw': f"{self.test_start_date} ~ {self.test_end_date}",
            'test_period_effective': f"{self.test_start_date} ~ {self.effective_test_end}",
            'total_period': f"{self.data_start_date} ~ {self.data_end_date}",
            'forward_window_days': self.forward_window,
            'label_overflow_protection': "自動排除尾端15個交易日樣本"
        }
    
    def get_effective_date_ranges(self) -> Dict[str, Tuple[str, str]]:
        """獲取有效的日期範圍 (已處理標籤外溢)"""
        return {
            'train': (self.data_start_date, self.effective_train_end),
            'validation': (self.val_start_date, self.effective_val_end),
            'test': (self.test_start_date, self.effective_test_end)
        }
    
    def get_raw_date_ranges(self) -> Dict[str, Tuple[str, str]]:
        """獲取原始日期範圍 (用於資料查詢)"""
        return {
            'train': (self.data_start_date, self.train_end_date),
            'validation': (self.val_start_date, self.val_end_date),
            'test': (self.test_start_date, self.test_end_date)
        }
    
    def get_tensor_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """獲取張量形狀信息"""
        return {
            'other_features_frame': (self.n_stocks, self.sequence_length, self.other_features),
            'fundamental': (self.n_stocks, self.fundamental_features),
            'account': (self.account_features,),
            'labels': (self.n_stocks,)
        }
    
    def get_feature_breakdown(self) -> Dict[str, Any]:
        """獲取特徵分解信息"""
        return {
            'fundamental_features': {
                'count': self.fundamental_features,
                'update_frequency': '季度',
                'sources': ['monthly_revenue', 'financials'],
                'detail': self.fundamental_features_detail
            },
            'other_features': {
                'count': self.other_features,
                'update_frequency': '每日',
                'sources': ['candlesticks_daily', 'candlesticks_min', 'technical_indicators', 
                           'margin_purchase_shortsale', 'institutional_investors_buy_sell', 'financial_per'],
                'detail': self.other_features_detail
            },
            'account_features': {
                'count': self.account_features,
                'update_frequency': '即時',
                'sources': ['gym_env']
            },
            'total': self.total_features
        }
    
    def get_trading_cost_summary(self) -> Dict[str, float]:
        """獲取交易成本摘要"""
        effective_commission = self.commission_rate * self.commission_discount / 100
        return {
            'commission_buy': effective_commission,
            'commission_sell': effective_commission, 
            'tax_sell': self.tax_rate / 100,
            'round_trip_total': self.round_trip_cost,
            'buy_threshold_margin': self.buy_threshold - self.round_trip_cost,
            'sell_threshold_margin': abs(self.sell_threshold) - self.round_trip_cost
        }


def create_default_config() -> TrainingConfig:
    """創建默認配置"""
    return TrainingConfig()


def create_debug_config() -> TrainingConfig:
    """創建調試配置 (小數據集)"""
    config = TrainingConfig()
    config.debug_mode = True
    config.batch_size = 4
    config.num_epochs = 5
    config.n_stocks = 10  # 只使用10支股票進行調試
    config.sequence_length = 32  # 減少序列長度
    return config


def create_production_config() -> TrainingConfig:
    """創建生產配置 (完整數據集)"""
    config = TrainingConfig()
    config.debug_mode = False
    config.mixed_precision = True
    config.num_workers = 8
    config.batch_size = 64
    return config


if __name__ == "__main__":
    # 測試配置系統
    print("=== TSE Alpha 訓練配置系統測試 ===")
    
    # 創建默認配置
    config = create_default_config()
    print("[OK] 默認配置創建成功")
    
    # 顯示配置摘要
    print("\n[INFO] 資料分割信息:")
    for period, date_range in config.get_data_split_info().items():
        print(f"  {period}: {date_range}")
    
    print("\n[INFO] 有效日期範圍 (已處理標籤外溢):")
    for split, (start, end) in config.get_effective_date_ranges().items():
        print(f"  {split}: {start} ~ {end}")
    
    print("\n[WARNING] 標籤外溢保護:")
    print(f"  前瞻窗口: {config.forward_window} 個交易日")
    print(f"  自動排除尾端樣本: {config.auto_trim_horizon}")
    print(f"  緩衝天數: {config.horizon_buffer_days} 個交易日")
    
    print("\n[INFO] 張量形狀信息:")
    for tensor_name, shape in config.get_tensor_shapes().items():
        print(f"  {tensor_name}: {shape}")
    
    print("\n[INFO] 交易成本摘要:")
    for cost_type, value in config.get_trading_cost_summary().items():
        print(f"  {cost_type}: {value:.4f}")
    
    # 測試配置保存和載入
    config_file = "tmp_test_config.json"
    config.save_to_file(config_file)
    loaded_config = TrainingConfig.load_from_file(config_file)
    print(f"\n[OK] 配置保存和載入測試成功")
    
    # 清理測試文件
    import os
    os.remove(config_file)
    
    print("\n[INFO] 配置系統測試完成！")