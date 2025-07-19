# gym_env/env.py
"""
TSE Alpha Trading Environment - 台股量化交易環境
實作 15 日持倉限制、T+1 結算、風險控制的交易環境
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import logging

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent / "market_data_collector"))

try:
    from market_data_collector.utils.db import query_df
    from market_data_collector.utils.config import STOCK_IDS
except ImportError as e:
    print(f"警告: 無法導入資料庫模組: {e}")
    STOCK_IDS = [
        # 完整180支股票清單 - 從 stock_split_config.json 載入
        "2330", "2317", "2454", "2303", "2408", "2412", "2382", "2357", "2379", "3034",
        "3008", "4938", "2449", "2383", "2356", "3006", "3661", "2324", "8046", "3017",
        "6121", "3037", "3014", "3035", "3062", "3030", "3529", "5443", "2337", "8150",
        "3293", "3596", "2344", "2428", "2345", "2338", "6202", "5347", "3673", "3105",
        "6231", "6669", "4961", "4967", "6668", "4960", "3528", "6147", "3526", "6547",
        "8047", "3227", "4968", "5274", "6415", "6414", "6770", "2331", "6290", "2342",
        "2603", "2609", "2615", "2610", "2618", "2637", "2606", "2002", "2014", "2027",
        "2201", "1201", "1216", "1301", "1303", "1326", "1710", "1717", "1722", "1723",
        "1402", "1409", "1434", "1476", "2006", "2049", "2105", "2106", "2107", "1605",
        "1609", "1608", "1612", "2308", "1727", "1730", "1101", "1102", "1108", "1210",
        "1215", "1802", "1806", "1810", "1104", "1313", "1314", "1310", "5608", "5607",
        "8105", "8940", "5534", "5609", "5603", "2023", "2028", "2114", "9933", "2501",
        "2880", "2881", "2882", "2883", "2884", "2885", "2886", "2887", "2888", "2890",
        "2891", "2892", "2812", "3665", "2834", "2850", "2801", "2836", "2845", "4807",
        "3702", "3706", "4560", "8478", "4142", "4133", "6525", "6548", "6843", "1513",
        "1514", "1516", "1521", "1522", "1524", "1533", "1708", "3019", "5904", "5906",
        "5902", "6505", "6806", "6510", "2207", "2204", "2231", "1736", "4105", "4108",
        "4162", "1909", "1702", "9917", "1217", "1218", "1737", "1783", "3708", "1795"
    ]  # 完整180支股票清單

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccountManager:
    """帳戶管理器 - 處理持倉、NAV、風險控制"""
    
    def __init__(self, initial_cash: float = 1000000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # {symbol: {'qty': int, 'avg_price': float, 'days_held': int}}
        self.nav_history = []
        self.trade_history = []
        
        # 風險控制參數
        self.daily_max_loss_pct = 0.02  # 2%
        self.rolling_max_dd_pct = 0.10  # 10%
        self.max_position_per_stock = 300
        
    def get_nav(self, current_prices: Dict[str, float]) -> float:
        """計算當前 NAV"""
        position_value = 0.0
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                position_value += pos['qty'] * current_prices[symbol]
        
        nav = self.cash + position_value
        return nav
    
    def get_position_ratio(self, current_prices: Dict[str, float]) -> float:
        """計算持倉比例"""
        nav = self.get_nav(current_prices)
        if nav <= 0:
            return 0.0
        
        position_value = nav - self.cash
        return position_value / nav
    
    def get_unrealized_pnl_pct(self, current_prices: Dict[str, float]) -> float:
        """計算未實現損益百分比"""
        if not self.positions:
            return 0.0
        
        total_cost = 0.0
        total_value = 0.0
        
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                cost = pos['qty'] * pos['avg_price']
                value = pos['qty'] * current_prices[symbol]
                total_cost += cost
                total_value += value
        
        if total_cost <= 0:
            return 0.0
        
        return (total_value - total_cost) / total_cost
    
    def check_risk_limits(self, current_prices: Dict[str, float]) -> Dict[str, bool]:
        """檢查風險限制"""
        nav = self.get_nav(current_prices)
        
        # 檢查每日最大損失
        if self.nav_history:
            daily_loss_pct = (nav - self.nav_history[-1]) / self.nav_history[-1]
            daily_max_loss_exceeded = daily_loss_pct < -self.daily_max_loss_pct
        else:
            daily_max_loss_exceeded = False
        
        # 檢查滾動最大回撤
        if len(self.nav_history) >= 30:
            recent_navs = self.nav_history[-30:] + [nav]
            peak = max(recent_navs)
            current_dd = (nav - peak) / peak
            rolling_max_dd_exceeded = current_dd < -self.rolling_max_dd_pct
        else:
            rolling_max_dd_exceeded = False
        
        return {
            'daily_max_loss_exceeded': daily_max_loss_exceeded,
            'rolling_max_dd_exceeded': rolling_max_dd_exceeded
        }
    
    def execute_trade(self, symbol: str, qty: int, price: float, 
                     trade_cost: float = 0.001425) -> Dict[str, Any]:
        """執行交易"""
        if symbol not in self.positions:
            self.positions[symbol] = {'qty': 0, 'avg_price': 0.0, 'days_held': 0}
        
        current_qty = self.positions[symbol]['qty']
        new_qty = current_qty + qty
        
        # 檢查持倉限制
        if abs(new_qty) > self.max_position_per_stock:
            return {'success': False, 'reason': 'position_limit_exceeded'}
        
        # 計算交易成本
        trade_value = abs(qty * price)
        cost = trade_value * trade_cost
        
        # 檢查現金是否足夠
        if qty > 0:  # 買入
            total_cost = trade_value + cost
            if self.cash < total_cost:
                return {'success': False, 'reason': 'insufficient_cash'}
            self.cash -= total_cost
        else:  # 賣出
            self.cash += trade_value - cost
        
        # 更新持倉
        if new_qty == 0:
            # 平倉
            del self.positions[symbol]
        else:
            # 更新平均成本
            if qty > 0:  # 買入，更新平均價格
                total_cost = current_qty * self.positions[symbol]['avg_price'] + qty * price
                self.positions[symbol]['avg_price'] = total_cost / new_qty
            
            self.positions[symbol]['qty'] = new_qty
        
        # 記錄交易
        trade_record = {
            'symbol': symbol,
            'qty': qty,
            'price': price,
            'cost': cost,
            'timestamp': datetime.now()
        }
        self.trade_history.append(trade_record)
        
        return {'success': True, 'cost': cost}
    
    def update_holding_days(self):
        """更新持倉天數"""
        for symbol in self.positions:
            self.positions[symbol]['days_held'] += 1
    
    def get_timeout_positions(self, max_days: int = 15) -> List[str]:
        """獲取超時持倉"""
        timeout_symbols = []
        for symbol, pos in self.positions.items():
            if pos['days_held'] >= max_days:
                timeout_symbols.append(symbol)
        return timeout_symbols


class DataProvider:
    """資料提供器 - 從資料庫載入價格和特徵資料"""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data_cache = {}
        self.current_date_idx = 0
        self.trading_dates = []
        
        self._load_data()
    
    def _load_data(self):
        """載入資料"""
        try:
            # 載入日線資料
            for symbol in self.symbols:
                query = """
                SELECT date, open, high, low, close, volume 
                FROM candlesticks_daily 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date ASC
                """
                df = query_df(query, (symbol, self.start_date, self.end_date))
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date']).dt.date
                    self.data_cache[symbol] = df
                    
                    # 建立交易日期清單
                    if not self.trading_dates:
                        self.trading_dates = df['date'].tolist()
            
            logger.info(f"載入 {len(self.symbols)} 檔股票，{len(self.trading_dates)} 個交易日")
            
        except Exception as e:
            logger.error(f"資料載入失敗: {e}")
            # 建立虛擬資料用於測試
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """建立虛擬資料用於測試"""
        logger.warning("使用虛擬資料進行測試")
        
        start = datetime.strptime(self.start_date, '%Y-%m-%d').date()
        end = datetime.strptime(self.end_date, '%Y-%m-%d').date()
        
        current_date = start
        while current_date <= end:
            # 跳過週末
            if current_date.weekday() < 5:
                self.trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        # 為每檔股票建立虛擬價格資料
        for symbol in self.symbols:
            base_price = 100.0 if symbol == '2330' else 50.0
            data = []
            
            for i, trade_date in enumerate(self.trading_dates):
                # 簡單的隨機遊走
                price_change = np.random.normal(0, 0.02)  # 2% 波動
                price = base_price * (1 + price_change * i * 0.1)
                
                data.append({
                    'date': trade_date,
                    'open': price * 0.99,
                    'high': price * 1.02,
                    'low': price * 0.98,
                    'close': price,
                    'volume': np.random.randint(100000, 1000000)
                })
            
            self.data_cache[symbol] = pd.DataFrame(data)
    
    def get_current_date(self) -> date:
        """獲取當前日期"""
        if self.current_date_idx < len(self.trading_dates):
            return self.trading_dates[self.current_date_idx]
        return None
    
    def get_current_prices(self) -> Dict[str, float]:
        """獲取當前價格"""
        current_date = self.get_current_date()
        if current_date is None:
            return {}
        
        prices = {}
        for symbol in self.symbols:
            if symbol in self.data_cache:
                df = self.data_cache[symbol]
                day_data = df[df['date'] == current_date]
                if not day_data.empty:
                    prices[symbol] = float(day_data.iloc[0]['close'])
        
        return prices
    
    def get_observation_data(self, lookback_days: int = 64) -> Dict[str, Any]:
        """獲取觀測資料"""
        current_date = self.get_current_date()
        if current_date is None:
            return {}
        
        # 計算開始日期索引
        start_idx = max(0, self.current_date_idx - lookback_days + 1)
        end_idx = self.current_date_idx + 1
        
        observation = {
            'price_frame': {},
            'fundamental': {},
            'current_date': current_date,
            'trading_dates': self.trading_dates[start_idx:end_idx]
        }
        
        # 載入價格框架資料
        for symbol in self.symbols:
            if symbol in self.data_cache:
                df = self.data_cache[symbol]
                period_data = df.iloc[start_idx:end_idx]
                
                if not period_data.empty:
                    # 標準化價格資料 (使用最後一天的收盤價作為基準)
                    last_close = period_data.iloc[-1]['close']
                    
                    price_features = np.array([
                        period_data['open'] / last_close,
                        period_data['high'] / last_close,
                        period_data['low'] / last_close,
                        period_data['close'] / last_close,
                        period_data['volume'] / period_data['volume'].mean()
                    ]).T  # Shape: (days, 5)
                    
                    observation['price_frame'][symbol] = price_features
        
        return observation
    
    def step(self):
        """前進一個交易日"""
        self.current_date_idx += 1
    
    def is_done(self) -> bool:
        """檢查是否結束"""
        return self.current_date_idx >= len(self.trading_dates)


class TSEAlphaEnv(gym.Env):
    """TSE Alpha 交易環境"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 symbols: Optional[List[str]] = None,
                 start_date: str = '2024-01-01',
                 end_date: str = '2024-12-31',
                 initial_cash: float = 1000000.0,
                 max_holding_days: int = 15,
                 max_position_per_stock: int = 300,
                 daily_max_loss_pct: float = 0.02,
                 rolling_max_dd_pct: float = 0.10):
        
        super().__init__()
        
        # 環境參數
        self.symbols = symbols or STOCK_IDS[:10]  # 預設使用前 10 檔股票
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.max_holding_days = max_holding_days
        self.max_position_per_stock = max_position_per_stock
        self.daily_max_loss_pct = daily_max_loss_pct
        self.rolling_max_dd_pct = rolling_max_dd_pct
        
        # 初始化組件
        self.account_manager = AccountManager(initial_cash)
        self.account_manager.max_position_per_stock = max_position_per_stock
        self.account_manager.daily_max_loss_pct = daily_max_loss_pct
        self.account_manager.rolling_max_dd_pct = rolling_max_dd_pct
        
        self.data_provider = None
        self.episode_step_count = 0
        
        # 定義動作空間: (股票索引, 交易數量)
        self.n_symbols = len(self.symbols)
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.n_symbols),  # 股票索引
            spaces.Box(low=-max_position_per_stock, high=max_position_per_stock, 
                      shape=(1,), dtype=np.int16)  # 交易數量
        ))
        
        # 定義觀測空間
        self._define_observation_space()
        
        logger.info(f"TSE Alpha 環境初始化完成，股票: {self.symbols}")
    
    def _define_observation_space(self):
        """定義觀測空間"""
        # 獲取實際配置的價格特徵數量
        try:
            from models.config.training_config import TrainingConfig
            config = TrainingConfig()
            price_features_count = config.other_features  # 51個其他特徵 (66維配置)
        except:
            price_features_count = 51  # 預設值 (66維配置)
        
        # price_frame: 每檔股票 64 天 × 51 個其他特徵 (66維配置)
        price_frame_shape = (len(self.symbols), 64, price_features_count)
        
        # fundamental: 基本面特徵 (15個，66維配置)
        fundamental_shape = (15,)
        
        # account: 帳戶狀態 (4 個特徵)
        account_shape = (4,)
        
        self.observation_space = spaces.Dict({
            'price_frame': spaces.Box(
                low=0.0, high=10.0, 
                shape=price_frame_shape, 
                dtype=np.float32
            ),
            'fundamental': spaces.Box(
                low=-5.0, high=5.0,
                shape=fundamental_shape,
                dtype=np.float32
            ),
            'account': spaces.Box(
                low=-1.0, high=1.0,
                shape=account_shape,
                dtype=np.float32
            )
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """重置環境"""
        super().reset(seed=seed)
        
        # 重置帳戶管理器
        self.account_manager = AccountManager(self.initial_cash)
        self.account_manager.max_position_per_stock = self.max_position_per_stock
        self.account_manager.daily_max_loss_pct = self.daily_max_loss_pct
        self.account_manager.rolling_max_dd_pct = self.rolling_max_dd_pct
        
        # 重置資料提供器
        self.data_provider = DataProvider(self.symbols, self.start_date, self.end_date)
        self.episode_step_count = 0
        
        # 獲取初始觀測
        observation = self._get_observation()
        info = self._get_info()
        
        logger.info("環境重置完成")
        return observation, info
    
    def step(self, action: Tuple[int, np.ndarray]) -> Tuple[Dict, float, bool, bool, Dict]:
        """執行一步動作"""
        self.episode_step_count += 1
        
        # 解析動作
        symbol_idx, qty_array = action
        qty = int(qty_array[0])
        
        if symbol_idx >= len(self.symbols):
            symbol_idx = 0  # 防止索引越界
        
        symbol = self.symbols[symbol_idx]
        
        # 獲取當前價格
        current_prices = self.data_provider.get_current_prices()
        
        # 執行交易
        reward = 0.0
        info = {'trade_executed': False, 'trade_result': None}
        
        if symbol in current_prices and qty != 0:
            price = current_prices[symbol]
            trade_result = self.account_manager.execute_trade(symbol, qty, price)
            info['trade_executed'] = True
            info['trade_result'] = trade_result
            
            if trade_result['success']:
                logger.debug(f"交易執行: {symbol} {qty} @ {price}")
        
        # 更新持倉天數
        self.account_manager.update_holding_days()
        
        # 檢查超時持倉並強制平倉
        timeout_positions = self.account_manager.get_timeout_positions(self.max_holding_days)
        timeout_penalty = 0.0
        
        for timeout_symbol in timeout_positions:
            if timeout_symbol in current_prices:
                pos = self.account_manager.positions[timeout_symbol]
                price = current_prices[timeout_symbol]
                
                # 強制平倉
                self.account_manager.execute_trade(timeout_symbol, -pos['qty'], price)
                
                # 超時罰款
                timeout_penalty -= 0.01  # 1% 罰款
                logger.warning(f"強制平倉超時持倉: {timeout_symbol}")
        
        # 計算獎勵
        nav = self.account_manager.get_nav(current_prices)
        self.account_manager.nav_history.append(nav)
        
        # 基本獎勵：NAV 變化
        if len(self.account_manager.nav_history) > 1:
            nav_change = nav - self.account_manager.nav_history[-2]
            reward = nav_change / self.initial_cash  # 標準化
        
        # 加入超時罰款
        reward += timeout_penalty
        
        # 檢查風險限制
        risk_status = self.account_manager.check_risk_limits(current_prices)
        
        # 前進到下一個交易日
        self.data_provider.step()
        
        # 檢查是否結束
        terminated = self.data_provider.is_done()
        truncated = (risk_status['daily_max_loss_exceeded'] or 
                    risk_status['rolling_max_dd_exceeded'])
        
        if truncated:
            reward -= 0.05  # 風險限制罰款
            logger.warning("觸發風險限制，提前結束")
        
        # 獲取新的觀測
        observation = self._get_observation()
        info.update(self._get_info())
        info.update(risk_status)
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """獲取當前觀測"""
        if self.data_provider is None:
            # 返回零觀測
            return {
                'price_frame': np.zeros(self.observation_space['price_frame'].shape, dtype=np.float32),
                'fundamental': np.zeros(self.observation_space['fundamental'].shape, dtype=np.float32),
                'account': np.zeros(self.observation_space['account'].shape, dtype=np.float32)
            }
        
        # 獲取價格框架資料
        obs_data = self.data_provider.get_observation_data()
        current_prices = self.data_provider.get_current_prices()
        
        # 獲取實際配置的價格特徵數量
        try:
            from models.config.training_config import TrainingConfig
            config = TrainingConfig()
            price_features_count = config.other_features  # 51個其他特徵 (66維配置)
        except:
            price_features_count = 51  # 預設值 (66維配置)
        
        # 構建價格框架 - 使用51個其他特徵 (66維配置)
        price_frame = np.zeros((len(self.symbols), 64, price_features_count), dtype=np.float32)
        for i, symbol in enumerate(self.symbols):
            if symbol in obs_data.get('price_frame', {}):
                symbol_data = obs_data['price_frame'][symbol]
                # 檢查是否需要擴展特徵維度
                if symbol_data.shape[1] == 5:
                    # 如果只有5個特徵，需要使用特徵引擎擴展到27個
                    # 這裡暫時用零填充，實際應該重新計算技術指標
                    expanded_data = np.zeros((symbol_data.shape[0], price_features_count))
                    expanded_data[:, :5] = symbol_data  # 保留原始OHLCV
                    symbol_data = expanded_data
                
                # 填充到固定大小 (64, 27)
                rows = min(symbol_data.shape[0], 64)
                cols = min(symbol_data.shape[1], price_features_count)
                price_frame[i, -rows:, :cols] = symbol_data[-rows:, :cols]
        
        # 基本面特徵 (15個，66維配置)
        fundamental = np.zeros(15, dtype=np.float32)
        
        # 帳戶狀態
        nav = self.account_manager.get_nav(current_prices)
        nav_norm = nav / self.initial_cash - 1.0  # 標準化 NAV
        pos_ratio = self.account_manager.get_position_ratio(current_prices)
        unreal_pnl_pct = self.account_manager.get_unrealized_pnl_pct(current_prices)
        
        # 風險緩衝 (距離風險限制的距離)
        risk_buffer = 1.0 - max(abs(nav_norm), pos_ratio)
        
        account = np.array([nav_norm, pos_ratio, unreal_pnl_pct, risk_buffer], dtype=np.float32)
        
        return {
            'price_frame': price_frame,
            'fundamental': fundamental,
            'account': account
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """獲取額外資訊"""
        current_prices = self.data_provider.get_current_prices() if self.data_provider else {}
        nav = self.account_manager.get_nav(current_prices)
        
        return {
            'nav': nav,
            'cash': self.account_manager.cash,
            'positions': dict(self.account_manager.positions),
            'current_date': self.data_provider.get_current_date() if self.data_provider else None,
            'episode_step': self.episode_step_count,
            'current_prices': current_prices
        }
    
    def get_account_state(self) -> Dict[str, Any]:
        """獲取帳戶狀態 (與_get_info相同，提供向後相容性)"""
        return self._get_info()
    
    def render(self, mode: str = 'human'):
        """渲染環境狀態"""
        if mode == 'human':
            info = self._get_info()
            print(f"=== TSE Alpha 環境狀態 ===")
            print(f"日期: {info['current_date']}")
            print(f"NAV: {info['nav']:,.2f}")
            print(f"現金: {info['cash']:,.2f}")
            print(f"持倉: {len(info['positions'])} 檔股票")
            print(f"步數: {info['episode_step']}")
            
            if info['positions']:
                print("持倉明細:")
                for symbol, pos in info['positions'].items():
                    current_price = info['current_prices'].get(symbol, 0)
                    market_value = pos['qty'] * current_price
                    print(f"  {symbol}: {pos['qty']} 股 @ {pos['avg_price']:.2f} "
                          f"(市值: {market_value:,.0f}, 持有: {pos['days_held']} 天)")
    
    def close(self):
        """關閉環境"""
        pass


# 測試函數
def test_env():
    """測試環境基本功能"""
    print("=== 測試 TSE Alpha 環境 ===")
    
    try:
        # 建立環境
        env = TSEAlphaEnv(
            symbols=['2330', '2317'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_cash=1000000.0
        )
        
        print("✅ 環境建立成功")
        
        # 重置環境
        obs, info = env.reset()
        print("✅ 環境重置成功")
        print(f"觀測空間: {env.observation_space}")
        print(f"動作空間: {env.action_space}")
        
        # 執行幾步
        for step in range(5):
            # 隨機動作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"步驟 {step + 1}: 獎勵 = {reward:.6f}, NAV = {info['nav']:,.2f}")
            
            if terminated or truncated:
                break
        
        print("✅ 環境測試完成")
        env.close()
        
    except Exception as e:
        print(f"❌ 環境測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_env()