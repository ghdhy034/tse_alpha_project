# data_pipeline/test_features.py
"""
特徵工程模組測試
測試技術指標、籌碼面、基本面特徵的計算正確性
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import unittest
from datetime import datetime, timedelta

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent / "market_data_collector"))
sys.path.append(str(Path(__file__).parent))

try:
    from features import FeatureEngine
    from market_data_collector.utils.db import query_df, get_conn
except ImportError as e:
    print(f"導入錯誤: {e}")
    sys.exit(1)


class TestFeatureEngine(unittest.TestCase):
    """特徵工程測試類"""
    
    def setUp(self):
        """測試設置"""
        self.engine = FeatureEngine()
        
        # 創建測試資料
        self.test_data = self._create_test_data()
    
    def _create_test_data(self):
        """創建測試用的價格資料"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # 模擬價格資料 - 簡單的上升趨勢加上噪音
        base_price = 100.0
        price_trend = np.linspace(0, 20, 100)  # 20% 上升趨勢
        noise = np.random.normal(0, 2, 100)    # 2% 隨機噪音
        
        closes = base_price + price_trend + noise
        
        # 確保價格合理性
        closes = np.maximum(closes, base_price * 0.8)  # 最低不低於 80%
        
        # 生成 OHLC 資料
        data = []
        for i, (date, close) in enumerate(zip(dates, closes)):
            # 模擬合理的 OHLC 關係
            volatility = abs(noise[i]) / 100 + 0.01  # 1-3% 日內波動
            
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = close * (1 + np.random.normal(0, 0.005))  # 開盤價接近收盤價
            
            volume = np.random.randint(100000, 1000000)
            
            data.append({
                'symbol': '2330',
                'date': date.date(),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_technical_indicators(self):
        """測試技術指標計算"""
        print("測試技術指標...")
        
        # 測試移動平均線
        ma_features = self.engine.calculate_ma_features(self.test_data)
        
        self.assertIn('ma_5', ma_features.columns)
        self.assertIn('ma_20', ma_features.columns)
        self.assertIn('ma_60', ma_features.columns)
        
        # 檢查移動平均線的合理性
        self.assertTrue(ma_features['ma_5'].iloc[-1] > 0)
        self.assertTrue(ma_features['ma_20'].iloc[-1] > 0)
        
        # 測試 MACD
        macd_features = self.engine.calculate_macd_features(self.test_data)
        
        self.assertIn('macd', macd_features.columns)
        self.assertIn('macd_signal', macd_features.columns)
        self.assertIn('macd_histogram', macd_features.columns)
        
        # 測試 RSI
        rsi_features = self.engine.calculate_rsi_features(self.test_data)
        
        self.assertIn('rsi_14', rsi_features.columns)
        self.assertTrue(0 <= rsi_features['rsi_14'].iloc[-1] <= 100)
        
        # 測試布林通道
        bb_features = self.engine.calculate_bollinger_features(self.test_data)
        
        self.assertIn('bb_upper', bb_features.columns)
        self.assertIn('bb_lower', bb_features.columns)
        self.assertIn('bb_width', bb_features.columns)
        
        print("✅ 技術指標測試通過")
    
    def test_volume_indicators(self):
        """測試成交量指標"""
        print("測試成交量指標...")
        
        # 測試 OBV
        obv_features = self.engine.calculate_obv_features(self.test_data)
        self.assertIn('obv', obv_features.columns)
        
        # 測試 VWAP
        vwap_features = self.engine.calculate_vwap_features(self.test_data)
        self.assertIn('vwap_20', vwap_features.columns)
        
        # 測試成交量比率
        vol_features = self.engine.calculate_volume_features(self.test_data)
        self.assertIn('volume_ratio_5', vol_features.columns)
        self.assertIn('volume_ratio_20', vol_features.columns)
        
        print("✅ 成交量指標測試通過")
    
    def test_volatility_indicators(self):
        """測試波動率指標"""
        print("測試波動率指標...")
        
        # 測試 ATR
        atr_features = self.engine.calculate_atr_features(self.test_data)
        self.assertIn('atr_14', atr_features.columns)
        self.assertTrue(atr_features['atr_14'].iloc[-1] > 0)
        
        # 測試歷史波動率
        vol_features = self.engine.calculate_volatility_features(self.test_data)
        self.assertIn('volatility_20', vol_features.columns)
        self.assertTrue(vol_features['volatility_20'].iloc[-1] > 0)
        
        print("✅ 波動率指標測試通過")
    
    def test_feature_standardization(self):
        """測試特徵標準化"""
        print("測試特徵標準化...")
        
        # 創建測試特徵
        features = pd.DataFrame({
            'feature1': np.random.normal(100, 20, 100),
            'feature2': np.random.normal(0.5, 0.1, 100),
            'feature3': np.random.normal(-10, 5, 100)
        })
        
        # 測試 Z-score 標準化
        standardized = self.engine.standardize_features(features, method='zscore')
        
        # 檢查標準化後的統計特性
        for col in standardized.columns:
            if col.endswith('_zscore'):
                mean_val = standardized[col].mean()
                std_val = standardized[col].std()
                self.assertAlmostEqual(mean_val, 0, places=1)
                self.assertAlmostEqual(std_val, 1, places=1)
        
        # 測試滾動標準化
        rolling_std = self.engine.standardize_features(features, method='rolling', window=20)
        self.assertTrue(len(rolling_std) == len(features))
        
        print("✅ 特徵標準化測試通過")
    
    def test_feature_pipeline(self):
        """測試完整特徵管線"""
        print("測試完整特徵管線...")
        
        # 測試單一股票特徵計算
        features = self.engine.calculate_stock_features('2330', '2024-01-01', '2024-03-31')
        
        if features is not None and not features.empty:
            # 檢查特徵數量
            self.assertGreater(len(features.columns), 10)
            
            # 檢查是否有 NaN 值
            nan_count = features.isnull().sum().sum()
            print(f"NaN 值數量: {nan_count}")
            
            # 檢查是否有無限值
            inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
            print(f"無限值數量: {inf_count}")
            
            print(f"特徵維度: {features.shape}")
            print(f"特徵列表: {list(features.columns)}")
        else:
            print("⚠️  無法從資料庫載入資料，使用測試資料")
            
            # 使用測試資料計算特徵
            all_features = self.engine.calculate_all_features(self.test_data)
            self.assertGreater(len(all_features.columns), 5)
        
        print("✅ 特徵管線測試通過")
    
    def test_performance(self):
        """測試效能"""
        print("測試效能...")
        
        start_time = datetime.now()
        
        # 計算多檔股票的特徵
        symbols = ['2330', '2317', '2603']
        
        for symbol in symbols:
            features = self.engine.calculate_stock_features(
                symbol, '2024-01-01', '2024-01-31'
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"計算 {len(symbols)} 檔股票特徵耗時: {duration:.2f} 秒")
        
        # 效能要求: 每檔股票 < 10 秒
        self.assertLess(duration / len(symbols), 10.0)
        
        print("✅ 效能測試通過")


def run_smoke_test():
    """快速煙霧測試"""
    print("=== 特徵工程煙霧測試 ===")
    
    try:
        # 基本功能測試
        engine = FeatureEngine()
        print("✅ 特徵引擎初始化成功")
        
        # 測試資料創建
        test_data = pd.DataFrame({
            'symbol': ['2330'] * 50,
            'date': pd.date_range('2024-01-01', periods=50),
            'open': np.random.uniform(95, 105, 50),
            'high': np.random.uniform(100, 110, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(95, 105, 50),
            'volume': np.random.randint(100000, 1000000, 50)
        })
        
        # 計算基本特徵
        ma_features = engine.calculate_ma_features(test_data)
        print(f"✅ 移動平均特徵計算成功: {ma_features.shape}")
        
        macd_features = engine.calculate_macd_features(test_data)
        print(f"✅ MACD 特徵計算成功: {macd_features.shape}")
        
        rsi_features = engine.calculate_rsi_features(test_data)
        print(f"✅ RSI 特徵計算成功: {rsi_features.shape}")
        
        # 測試完整特徵計算
        all_features = engine.calculate_all_features(test_data)
        print(f"✅ 完整特徵計算成功: {all_features.shape}")
        
        print("🎉 特徵工程煙霧測試全部通過！")
        return True
        
    except Exception as e:
        print(f"❌ 煙霧測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主測試函數"""
    print("=== 特徵工程模組測試 ===")
    
    # 先執行煙霧測試
    if not run_smoke_test():
        print("煙霧測試失敗，跳過完整測試")
        return
    
    print("\n=== 執行完整測試套件 ===")
    
    # 執行完整測試
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    main()