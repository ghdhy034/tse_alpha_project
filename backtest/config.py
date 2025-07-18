# backtest/config.py
"""
回測引擎配置模組 - 參考 References.txt 建議
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from datetime import date
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回測配置 - 整合 References.txt 建議"""
    
    # === 時間設定 ===
    train_window_months: int = 12      # 訓練窗口
    test_window_months: int = 3        # 測試窗口  
    step_size_months: int = 1          # 步進大小
    horizon_days: int = 15             # 預測視野 (用於剪尾)
    
    # === 股票設定 ===
    stock_universe: Optional[List[str]] = None   # 股票池
    use_stock_splits: bool = True                # 使用股票分割
    
    # === 環境設定 ===
    initial_cash: float = 1000000.0
    max_holding_days: int = 15
    max_position_per_stock: int = 300
    
    # === 風險設定 ===
    daily_max_loss_pct: float = 0.02
    rolling_max_dd_pct: float = 0.10
    position_size_limit: float = 0.1   # 單檔股票最大倉位
    
    # === 成本設定 (References.txt 建議) ===
    commission_rate: float = 0.001425
    commission_discount: float = 1.0    # 手續費折扣
    min_commission: float = 20.0        # 最低手續費
    tax_rate: float = 0.003
    slippage_bps: int = 5               # 滑價 (基點)
    k_slip: float = 0.0001              # 滑價係數
    min_tick: float = 0.01              # 最小跳動單位
    
    # === 資料管理 (References.txt 建議) ===
    cache_mode: Literal['arrow', 'duckdb', 'none'] = 'duckdb'
    preload_data: bool = True           # 預載入資料
    
    # === 並行設定 (References.txt 建議) ===
    backend: Literal['seq', 'mp', 'ray'] = 'seq'
    max_workers: int = 4
    
    # === 輸出設定 ===
    save_trades: bool = True
    save_positions: bool = True
    generate_report: bool = True
    report_format: str = "html"
    
    # === 測試設定 ===
    smoke_test_timeout: int = 60        # Smoke test 超時 (秒)
    
    def __post_init__(self):
        """配置驗證"""
        if self.stock_universe is None:
            # 使用預設股票池
            try:
                import stock_config
                self.stock_universe = stock_config.get_all_stocks()
                logger.info(f"使用預設股票池: {len(self.stock_universe)} 檔股票")
            except ImportError:
                self.stock_universe = ['2330', '2317', '2603']
                logger.warning("無法載入 stock_config，使用最小股票池")
        
        # 驗證參數合理性
        if self.train_window_months < 1:
            raise ValueError("訓練窗口至少需要 1 個月")
        
        if self.test_window_months < 1:
            raise ValueError("測試窗口至少需要 1 個月")
        
        if self.horizon_days < 1:
            raise ValueError("預測視野至少需要 1 天")
        
        if self.commission_discount <= 0 or self.commission_discount > 1:
            raise ValueError("手續費折扣必須在 (0, 1] 範圍內")
        
        if self.min_commission < 0:
            raise ValueError("最低手續費不能為負數")
        
        logger.info(f"回測配置初始化完成: {self.train_window_months}M訓練/{self.test_window_months}M測試")


@dataclass
class Period:
    """時間週期"""
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    period_id: str
    
    def __post_init__(self):
        """週期驗證"""
        if self.train_start >= self.train_end:
            raise ValueError(f"訓練開始日期 {self.train_start} 必須早於結束日期 {self.train_end}")
        
        if self.test_start >= self.test_end:
            raise ValueError(f"測試開始日期 {self.test_start} 必須早於結束日期 {self.test_end}")
        
        if self.train_end >= self.test_start:
            raise ValueError(f"訓練結束日期 {self.train_end} 必須早於測試開始日期 {self.test_start}")
    
    @property
    def train_days(self) -> int:
        """訓練天數"""
        return (self.train_end - self.train_start).days
    
    @property
    def test_days(self) -> int:
        """測試天數"""
        return (self.test_end - self.test_start).days
    
    def __str__(self) -> str:
        return f"Period({self.period_id}: Train[{self.train_start}~{self.train_end}] Test[{self.test_start}~{self.test_end}])"


@dataclass
class CostConfig:
    """成本計算配置 - 實作 References.txt 建議"""
    commission_rate: float = 0.001425
    commission_discount: float = 1.0
    min_commission: float = 20.0
    tax_rate: float = 0.003
    slippage_bps: int = 5
    k_slip: float = 0.0001
    min_tick: float = 0.01
    
    def calculate_commission(self, trade_value: float) -> float:
        """計算手續費"""
        commission = trade_value * self.commission_rate * self.commission_discount
        return max(commission, self.min_commission)
    
    def calculate_slippage(self, price: float, quantity: int, is_buy: bool) -> float:
        """計算滑價 - k_slip + min_tick 考量"""
        base_slippage = abs(quantity) * self.k_slip
        tick_slippage = self.min_tick * (1 if is_buy else -1)
        
        # 滑價方向：買入向上，賣出向下
        direction = 1 if is_buy else -1
        total_slippage = (base_slippage + abs(tick_slippage)) * direction
        
        return price + total_slippage
    
    def calculate_total_cost(self, price: float, quantity: int, is_buy: bool) -> Dict[str, float]:
        """計算總交易成本"""
        adjusted_price = self.calculate_slippage(price, quantity, is_buy)
        trade_value = abs(quantity) * adjusted_price
        
        commission = self.calculate_commission(trade_value)
        tax = trade_value * self.tax_rate if not is_buy else 0.0  # 只有賣出收稅
        
        total_cost = commission + tax
        net_value = trade_value - total_cost if not is_buy else trade_value + total_cost
        
        return {
            'original_price': price,
            'adjusted_price': adjusted_price,
            'trade_value': trade_value,
            'commission': commission,
            'tax': tax,
            'total_cost': total_cost,
            'net_value': net_value
        }


def create_default_config() -> BacktestConfig:
    """創建預設配置"""
    return BacktestConfig()


def create_smoke_test_config() -> BacktestConfig:
    """創建 Smoke Test 配置 - 快速測試用"""
    return BacktestConfig(
        train_window_months=3,      # 縮短訓練窗口
        test_window_months=1,       # 縮短測試窗口
        stock_universe=['2330', '2317', '2603'],  # 最小股票池
        cache_mode='none',          # 不使用快取
        save_trades=False,          # 不保存詳細記錄
        save_positions=False,
        generate_report=False,
        smoke_test_timeout=60
    )


def test_config():
    """測試配置模組"""
    print("=== 測試回測配置模組 ===")
    
    # 測試預設配置
    print("1. 測試預設配置...")
    try:
        config = create_default_config()
        print(f"   ✅ 預設配置創建成功")
        print(f"   股票數量: {len(config.stock_universe)}")
        print(f"   訓練窗口: {config.train_window_months} 月")
        print(f"   測試窗口: {config.test_window_months} 月")
    except Exception as e:
        print(f"   ❌ 預設配置創建失敗: {e}")
        return False
    
    # 測試 Smoke Test 配置
    print("2. 測試 Smoke Test 配置...")
    try:
        smoke_config = create_smoke_test_config()
        print(f"   ✅ Smoke Test 配置創建成功")
        print(f"   股票數量: {len(smoke_config.stock_universe)}")
        print(f"   超時設定: {smoke_config.smoke_test_timeout} 秒")
    except Exception as e:
        print(f"   ❌ Smoke Test 配置創建失敗: {e}")
        return False
    
    # 測試週期創建
    print("3. 測試週期創建...")
    try:
        from datetime import date
        period = Period(
            train_start=date(2024, 1, 1),
            train_end=date(2024, 12, 31),
            test_start=date(2025, 1, 1),
            test_end=date(2025, 3, 31),
            period_id="test_period"
        )
        print(f"   ✅ 週期創建成功: {period}")
        print(f"   訓練天數: {period.train_days}")
        print(f"   測試天數: {period.test_days}")
    except Exception as e:
        print(f"   ❌ 週期創建失敗: {e}")
        return False
    
    # 測試成本計算
    print("4. 測試成本計算...")
    try:
        cost_config = CostConfig()
        
        # 測試買入成本
        buy_cost = cost_config.calculate_total_cost(
            price=580.0, quantity=100, is_buy=True
        )
        print(f"   買入成本: {buy_cost}")
        
        # 測試賣出成本
        sell_cost = cost_config.calculate_total_cost(
            price=590.0, quantity=100, is_buy=False
        )
        print(f"   賣出成本: {sell_cost}")
        
        print("   ✅ 成本計算測試完成")
    except Exception as e:
        print(f"   ❌ 成本計算測試失敗: {e}")
        return False
    
    print("✅ 配置模組測試完成")
    return True


if __name__ == "__main__":
    test_config()