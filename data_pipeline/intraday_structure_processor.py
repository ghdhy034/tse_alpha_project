"""
日內結構特徵處理器
從 candlesticks_min 5分K資料萃取日內結構信號
基於 References.txt 中的資料格式設計
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class IntradayStructureProcessor:
    """
    日內結構特徵處理器
    
    從 5分K 資料萃取 5 個日內結構信號：
    1. volatility: 日內波動度
    2. vwap_deviation: VWAP偏離度  
    3. volume_rhythm: 成交量節奏
    4. shadow_ratio: 上下影比
    5. noise_ratio: 噪音比
    """
    
    def __init__(self):
        self.feature_names = [
            'volatility',      # 日內波動度
            'vwap_deviation',  # VWAP偏離度
            'volume_rhythm',   # 成交量節奏
            'shadow_ratio',    # 上下影比
            'noise_ratio'      # 噪音比
        ]
        
        # 預期的 5分K 欄位 (基於 References.txt)
        self.expected_columns = ['open', 'high', 'low', 'close', 'volume']
    
    def process_daily_bars(self, minute_bars: pd.DataFrame) -> Dict[str, float]:
        """
        處理單日的5分K資料，萃取日內結構信號
        
        Args:
            minute_bars: 單日的5分K資料 (約64根K線)
                        包含欄位: open, high, low, close, volume
            
        Returns:
            Dict: 5個日內結構特徵值
        """
        # 驗證輸入資料
        if not self._validate_input(minute_bars):
            return self._get_default_features()
        
        try:
            # 1. 日內波動度
            volatility = self._calculate_volatility(minute_bars)
            
            # 2. VWAP偏離度
            vwap_deviation = self._calculate_vwap_deviation(minute_bars)
            
            # 3. 成交量節奏
            volume_rhythm = self._calculate_volume_rhythm(minute_bars)
            
            # 4. 上下影比
            shadow_ratio = self._calculate_shadow_ratio(minute_bars)
            
            # 5. 噪音比
            noise_ratio = self._calculate_noise_ratio(minute_bars)
            
            features = {
                'volatility': volatility,
                'vwap_deviation': vwap_deviation,
                'volume_rhythm': volume_rhythm,
                'shadow_ratio': shadow_ratio,
                'noise_ratio': noise_ratio
            }
            
            # 檢查並處理異常值
            features = self._handle_outliers(features)
            
            return features
            
        except Exception as e:
            logger.warning(f"日內結構特徵計算失敗: {e}")
            return self._get_default_features()
    
    def _validate_input(self, bars: pd.DataFrame) -> bool:
        """驗證輸入資料"""
        if bars is None or len(bars) == 0:
            return False
        
        # 檢查必要欄位
        missing_cols = [col for col in self.expected_columns if col not in bars.columns]
        if missing_cols:
            logger.warning(f"缺少必要欄位: {missing_cols}")
            return False
        
        # 檢查資料有效性
        if bars[self.expected_columns].isnull().all().any():
            logger.warning("存在全空的欄位")
            return False
        
        return True
    
    def _get_default_features(self) -> Dict[str, float]:
        """獲取預設特徵值"""
        return {name: 0.0 for name in self.feature_names}
    
    def _calculate_volatility(self, bars: pd.DataFrame) -> float:
        """
        計算日內波動度
        
        方法: (high - low) / close 的日內標準差
        反映價格在日內的波動程度
        """
        if len(bars) == 0:
            return 0.0
        
        # 避免除零錯誤
        close_prices = bars['close'].replace(0, np.nan)
        if close_prices.isnull().all():
            return 0.0
        
        # 計算每根K線的波動率
        bar_volatility = (bars['high'] - bars['low']) / close_prices
        
        # 移除異常值
        bar_volatility = bar_volatility.replace([np.inf, -np.inf], np.nan)
        bar_volatility = bar_volatility.dropna()
        
        if len(bar_volatility) == 0:
            return 0.0
        
        return float(bar_volatility.std())
    
    def _calculate_vwap_deviation(self, bars: pd.DataFrame) -> float:
        """
        計算VWAP偏離度
        
        方法: (收盤價 - VWAP) / VWAP
        反映價格相對於成交量加權平均價格的偏離程度
        """
        if len(bars) == 0:
            return 0.0
        
        # 計算VWAP
        total_volume = bars['volume'].sum()
        if total_volume == 0:
            return 0.0
        
        # 使用典型價格計算VWAP
        typical_price = (bars['high'] + bars['low'] + bars['close']) / 3
        vwap = (typical_price * bars['volume']).sum() / total_volume
        
        # 計算偏離度
        final_close = bars['close'].iloc[-1]
        if vwap == 0:
            return 0.0
        
        deviation = (final_close - vwap) / vwap
        
        # 限制偏離度範圍
        deviation = np.clip(deviation, -1.0, 1.0)
        
        return float(deviation)
    
    def _calculate_volume_rhythm(self, bars: pd.DataFrame) -> float:
        """
        計算成交量節奏
        
        方法: 成交量分佈的變異係數
        反映成交量在日內的分佈均勻程度
        """
        if len(bars) == 0:
            return 0.0
        
        volumes = bars['volume']
        mean_volume = volumes.mean()
        
        if mean_volume == 0:
            return 0.0
        
        # 變異係數 = 標準差 / 平均值
        volume_cv = volumes.std() / mean_volume
        
        # 限制變異係數範圍
        volume_cv = np.clip(volume_cv, 0, 10.0)
        
        return float(volume_cv)
    
    def _calculate_shadow_ratio(self, bars: pd.DataFrame) -> float:
        """
        計算上下影比
        
        方法: (上影線 - 下影線) / 實體 的平均值
        反映買賣壓力的相對強度
        """
        if len(bars) == 0:
            return 0.0
        
        # 計算上影線和下影線
        upper_shadow = bars['high'] - np.maximum(bars['open'], bars['close'])
        lower_shadow = np.minimum(bars['open'], bars['close']) - bars['low']
        
        # 計算實體大小
        body_size = np.abs(bars['close'] - bars['open'])
        
        # 避免除零錯誤，為實體大小添加小的常數
        body_size_safe = body_size + 1e-8
        
        # 計算上下影比
        shadow_ratio = (upper_shadow - lower_shadow) / body_size_safe
        
        # 移除異常值
        shadow_ratio = shadow_ratio.replace([np.inf, -np.inf], np.nan)
        shadow_ratio = shadow_ratio.dropna()
        
        if len(shadow_ratio) == 0:
            return 0.0
        
        # 限制範圍
        mean_ratio = shadow_ratio.mean()
        mean_ratio = np.clip(mean_ratio, -10.0, 10.0)
        
        return float(mean_ratio)
    
    def _calculate_noise_ratio(self, bars: pd.DataFrame) -> float:
        """
        計算噪音比
        
        方法: 總價格變動 / 淨價格變動
        反映價格變動中噪音的比例
        """
        if len(bars) <= 1:
            return 0.0
        
        # 計算價格變動
        price_changes = bars['close'].diff().abs()
        total_movement = price_changes.sum()
        
        # 計算淨變動
        net_movement = abs(bars['close'].iloc[-1] - bars['close'].iloc[0])
        
        if net_movement == 0:
            # 如果沒有淨變動但有總變動，說明全是噪音
            return 10.0 if total_movement > 0 else 0.0
        
        noise_ratio = total_movement / net_movement
        
        # 限制噪音比範圍
        noise_ratio = np.clip(noise_ratio, 1.0, 20.0)
        
        return float(noise_ratio)
    
    def _handle_outliers(self, features: Dict[str, float]) -> Dict[str, float]:
        """處理異常值"""
        # 定義各特徵的合理範圍
        ranges = {
            'volatility': (0.0, 1.0),
            'vwap_deviation': (-1.0, 1.0),
            'volume_rhythm': (0.0, 10.0),
            'shadow_ratio': (-10.0, 10.0),
            'noise_ratio': (1.0, 20.0)
        }
        
        cleaned_features = {}
        for name, value in features.items():
            if name in ranges:
                min_val, max_val = ranges[name]
                cleaned_value = np.clip(value, min_val, max_val)
                
                # 檢查是否為有效數值
                if np.isnan(cleaned_value) or np.isinf(cleaned_value):
                    cleaned_value = 0.0
                
                cleaned_features[name] = float(cleaned_value)
            else:
                cleaned_features[name] = float(value) if not (np.isnan(value) or np.isinf(value)) else 0.0
        
        return cleaned_features
    
    def process_symbol_data(self, minute_data: pd.DataFrame) -> pd.DataFrame:
        """
        處理多股票多日的5分K資料，按 symbol + date 分組
        
        Args:
            minute_data: 包含 symbol, timestamp, OHLCV 的5分K資料
                        timestamp 格式: '2020-03-02 09:00:00' (基於 References.txt)
            
        Returns:
            pd.DataFrame: 每個 (symbol, date) 組合的日內結構特徵
        """
        if minute_data.empty:
            return pd.DataFrame()
        
        # 驗證必要欄位
        required_columns = ['symbol', 'timestamp'] + self.expected_columns
        missing_cols = [col for col in required_columns if col not in minute_data.columns]
        if missing_cols:
            logger.error(f"缺少必要欄位: {missing_cols}")
            return pd.DataFrame()
        
        # 從 timestamp 提取日期
        minute_data = minute_data.copy()
        minute_data['date'] = pd.to_datetime(minute_data['timestamp']).dt.date
        
        # 按 symbol + date 分組處理
        results = []
        
        for (symbol, date), group_data in minute_data.groupby(['symbol', 'date']):
            # 按時間排序確保正確的時間序列
            group_data = group_data.sort_values('timestamp')
            
            # 萃取日內結構特徵
            features = self.process_daily_bars(group_data)
            
            # 添加識別信息
            features['symbol'] = symbol
            features['date'] = date
            features['bar_count'] = len(group_data)  # 記錄實際K線數量
            
            results.append(features)
        
        if not results:
            return pd.DataFrame()
        
        # 轉換為DataFrame
        result_df = pd.DataFrame(results)
        
        # 重新排列欄位順序
        columns = ['symbol', 'date', 'bar_count'] + self.feature_names
        result_df = result_df[columns]
        
        return result_df
    
    def process_single_symbol_multiple_days(self, symbol_data: pd.DataFrame, 
                                          symbol: str) -> pd.DataFrame:
        """
        處理單一股票多日的5分K資料
        
        Args:
            symbol_data: 單一股票的5分K資料
            symbol: 股票代碼
            
        Returns:
            pd.DataFrame: 該股票每日的日內結構特徵
        """
        if symbol_data.empty:
            return pd.DataFrame()
        
        # 添加 symbol 欄位如果不存在
        if 'symbol' not in symbol_data.columns:
            symbol_data = symbol_data.copy()
            symbol_data['symbol'] = symbol
        
        return self.process_symbol_data(symbol_data)


def test_intraday_processor():
    """測試日內結構特徵處理器"""
    print("=== 測試日內結構特徵處理器 ===")
    
    # 創建測試資料 (模擬 References.txt 的格式)
    np.random.seed(42)
    
    # 模擬多股票多日的5分K資料
    symbols = ['2330', '2317']
    dates = ['2020-03-02', '2020-03-03']
    test_data = []
    
    for symbol in symbols:
        for date in dates:
            # 每日約64根5分K線 (09:00-13:30)
            base_price = 100.0 if symbol == '2330' else 50.0
            
            # 生成一天的時間戳
            start_time = pd.Timestamp(f'{date} 09:00:00')
            times = [start_time + pd.Timedelta(minutes=5*i) for i in range(64)]
            
            for i, timestamp in enumerate(times):
                # 模擬價格隨機遊走
                price_change = np.random.normal(0, 0.5)
                base_price += price_change
                
                # 生成OHLCV
                open_price = base_price
                high_price = open_price + abs(np.random.normal(0, 0.3))
                low_price = open_price - abs(np.random.normal(0, 0.3))
                close_price = open_price + np.random.normal(0, 0.2)
                volume = max(100, np.random.normal(1000, 300))
                
                test_data.append({
                    'symbol': symbol,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
    
    test_df = pd.DataFrame(test_data)
    
    print(f"📊 測試資料概況:")
    print(f"   總記錄數: {len(test_df)}")
    print(f"   股票數: {test_df['symbol'].nunique()}")
    print(f"   日期範圍: {test_df['timestamp'].min()} ~ {test_df['timestamp'].max()}")
    
    # 測試處理器
    processor = IntradayStructureProcessor()
    
    # 1. 測試單日處理
    print(f"\n1️⃣ 測試單日處理 (2330, 2020-03-02):")
    single_day_data = test_df[(test_df['symbol'] == '2330') & 
                             (test_df['timestamp'].str.startswith('2020-03-02'))]
    features = processor.process_daily_bars(single_day_data)
    
    for name, value in features.items():
        print(f"   {name}: {value:.6f}")
    
    # 2. 測試多股票多日處理
    print(f"\n2️⃣ 測試多股票多日處理:")
    result_df = processor.process_symbol_data(test_df)
    
    print(f"   處理結果:")
    print(f"   - 總組合數: {len(result_df)}")
    print(f"   - 欄位: {list(result_df.columns)}")
    
    print(f"\n   詳細結果:")
    for _, row in result_df.iterrows():
        print(f"   {row['symbol']} {row['date']}: {row['bar_count']}根K線")
        print(f"     volatility={row['volatility']:.4f}, vwap_deviation={row['vwap_deviation']:.4f}")
        print(f"     volume_rhythm={row['volume_rhythm']:.4f}, shadow_ratio={row['shadow_ratio']:.4f}")
        print(f"     noise_ratio={row['noise_ratio']:.4f}")
    
    # 3. 測試單股票處理
    print(f"\n3️⃣ 測試單股票處理 (2330):")
    symbol_2330_data = test_df[test_df['symbol'] == '2330']
    single_symbol_result = processor.process_single_symbol_multiple_days(symbol_2330_data, '2330')
    
    print(f"   2330 處理結果: {len(single_symbol_result)} 個交易日")
    for _, row in single_symbol_result.iterrows():
        print(f"     {row['date']}: volatility={row['volatility']:.4f}")
    
    print("\n✅ 日內結構特徵處理器測試完成")
    
    return result_df


if __name__ == "__main__":
    test_intraday_processor()