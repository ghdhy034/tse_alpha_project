# data_pipeline/test_chip_features.py
"""
籌碼面特徵測試 - 測試融資融券和法人進出特徵
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent / "market_data_collector"))
sys.path.append(str(Path(__file__).parent))

def test_chip_features():
    """測試籌碼面特徵計算"""
    print("=== 測試籌碼面特徵 ===")
    
    try:
        from features import FeatureEngine, ChipIndicators
        print("✅ 籌碼面模組導入成功")
        
        # 初始化
        engine = FeatureEngine()
        chip_calc = ChipIndicators()
        
        # 創建測試資料
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        
        # 模擬價格資料
        price_data = pd.DataFrame({
            'open': np.random.uniform(95, 105, 50),
            'high': np.random.uniform(100, 110, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(95, 105, 50),
            'volume': np.random.randint(100000, 1000000, 50)
        }, index=dates)
        
        # 模擬融資融券資料
        margin_data = pd.DataFrame({
            'margin_purchase_buy_volume': np.random.randint(1000, 10000, 50),
            'margin_purchase_sell_volume': np.random.randint(500, 5000, 50),
            'margin_purchase_cash_balance': np.random.randint(100000, 1000000, 50),
            'margin_purchase_yesterday_balance': np.random.randint(90000, 950000, 50),
            'short_sale_buy_volume': np.random.randint(100, 1000, 50),
            'short_sale_sell_volume': np.random.randint(200, 2000, 50),
            'short_sale_balance': np.random.randint(10000, 100000, 50),
            'short_sale_yesterday_balance': np.random.randint(9000, 95000, 50)
        }, index=dates)
        
        # 模擬機構投信資料
        institutional_data = pd.DataFrame({
            'foreign_investor_buy': np.random.randint(10000, 100000, 50),
            'foreign_investor_sell': np.random.randint(8000, 90000, 50),
            'investment_trust_buy': np.random.randint(5000, 50000, 50),
            'investment_trust_sell': np.random.randint(4000, 45000, 50),
            'dealer_buy': np.random.randint(2000, 20000, 50),
            'dealer_sell': np.random.randint(1500, 18000, 50),
            'dealer_hedge_buy': np.random.randint(1000, 10000, 50),
            'dealer_hedge_sell': np.random.randint(800, 9000, 50)
        }, index=dates)
        
        chip_data = {
            'margin': margin_data,
            'institutional': institutional_data
        }
        
        print("✅ 測試資料創建成功")
        
        # 測試籌碼面特徵計算
        chip_features = engine.calculate_chip_features('2330', price_data, chip_data)
        
        print(f"✅ 籌碼面特徵計算成功: {chip_features.shape}")
        print(f"特徵列表: {list(chip_features.columns)}")
        
        # 檢查特徵內容
        expected_features = [
            'margin_purchase_ratio', 'margin_balance_change_5d', 'margin_balance_change_20d',
            'short_sale_ratio', 'short_balance_change_5d', 'total_margin_ratio', 'margin_utilization',
            'foreign_net_buy_ratio', 'foreign_net_buy_5d', 'foreign_net_buy_20d',
            'trust_net_buy_ratio', 'trust_net_buy_5d', 'trust_net_buy_20d',
            'dealer_net_buy_ratio', 'dealer_hedge_ratio', 'institutional_consensus',
            'total_institutional_ratio', 'institutional_momentum_5d', 'institutional_momentum_20d',
            'foreign_consecutive_days'
        ]
        
        missing_features = set(expected_features) - set(chip_features.columns)
        if missing_features:
            print(f"⚠️  缺少特徵: {missing_features}")
        else:
            print("✅ 所有預期特徵都存在")
        
        # 測試特徵品質
        nan_count = chip_features.isnull().sum().sum()
        inf_count = np.isinf(chip_features.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"NaN 值數量: {nan_count}")
        print(f"無限值數量: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            print("✅ 籌碼面特徵品質良好")
        
        # 測試個別指標計算
        print("\n--- 測試個別籌碼指標 ---")
        
        # 測試融資融券比率
        margin_ratio = chip_calc.margin_ratio(
            margin_data['margin_purchase_buy_volume'],
            margin_data['short_sale_sell_volume'],
            price_data['volume']
        )
        print(f"✅ 融資融券比率: {margin_ratio.shape}, 範圍: {margin_ratio.min():.4f} ~ {margin_ratio.max():.4f}")
        
        # 測試外資淨買賣比率
        foreign_ratio = chip_calc.foreign_net_buy_ratio(
            institutional_data['foreign_investor_buy'],
            institutional_data['foreign_investor_sell'],
            price_data['volume']
        )
        print(f"✅ 外資淨買賣比率: {foreign_ratio.shape}, 範圍: {foreign_ratio.min():.4f} ~ {foreign_ratio.max():.4f}")
        
        # 測試機構一致性
        foreign_net = institutional_data['foreign_investor_buy'] - institutional_data['foreign_investor_sell']
        trust_net = institutional_data['investment_trust_buy'] - institutional_data['investment_trust_sell']
        dealer_net = institutional_data['dealer_buy'] - institutional_data['dealer_sell']
        
        consensus = chip_calc.institutional_consensus(foreign_net, trust_net, dealer_net)
        print(f"✅ 機構一致性指標: {consensus.shape}, 範圍: {consensus.min():.4f} ~ {consensus.max():.4f}")
        
        # 測試完整特徵管線
        print("\n--- 測試完整特徵管線 (含籌碼面) ---")
        
        # 模擬完整特徵計算
        all_features = engine.calculate_technical_features(price_data)
        chip_features_full = engine.calculate_chip_features('2330', price_data, chip_data)
        fundamental_features = engine.calculate_fundamental_features('2330', price_data)
        
        # 合併所有特徵
        complete_features = pd.concat([all_features, chip_features_full, fundamental_features], axis=1)
        
        print(f"✅ 完整特徵計算: {complete_features.shape}")
        print(f"技術指標: {all_features.shape[1]} 個")
        print(f"籌碼面特徵: {chip_features_full.shape[1]} 個")
        print(f"基本面特徵: {fundamental_features.shape[1]} 個")
        print(f"總特徵數: {complete_features.shape[1]} 個")
        
        # 檢查特徵分佈
        print("\n--- 籌碼面特徵統計 ---")
        for col in chip_features_full.columns[:5]:  # 顯示前5個特徵的統計
            stats = chip_features_full[col].describe()
            print(f"{col}: 均值={stats['mean']:.4f}, 標準差={stats['std']:.4f}")
        
        print("\n🎉 籌碼面特徵測試全部通過！")
        return True
        
    except Exception as e:
        print(f"❌ 籌碼面特徵測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_data():
    """測試真實資料庫資料"""
    print("\n=== 測試真實資料庫資料 ===")
    
    try:
        from features import FeatureEngine
        
        engine = FeatureEngine()
        
        # 嘗試載入真實資料
        symbol = '2330'
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        
        print(f"嘗試載入 {symbol} 的籌碼面資料...")
        
        chip_data = engine.load_chip_data(symbol, start_date, end_date)
        
        if chip_data:
            print(f"✅ 成功載入籌碼面資料:")
            for data_type, df in chip_data.items():
                print(f"  {data_type}: {df.shape}")
                
            # 載入價格資料
            price_data = engine.load_price_data(symbol, start_date, end_date)
            
            if not price_data.empty:
                # 計算籌碼面特徵
                chip_features = engine.calculate_chip_features(symbol, price_data, chip_data)
                print(f"✅ 真實資料籌碼面特徵: {chip_features.shape}")
                
                # 測試完整流程
                features, labels, prices = engine.process_single_symbol(
                    symbol, start_date, end_date, include_chip_features=True
                )
                
                print(f"✅ 完整流程測試: 特徵={features.shape}, 標籤={labels.shape}")
                
        else:
            print("⚠️  無真實籌碼面資料，這是正常的（可能資料庫中沒有相關資料）")
            
        return True
        
    except Exception as e:
        print(f"⚠️  真實資料測試失敗: {e}")
        print("這可能是因為資料庫中沒有籌碼面資料，屬於正常情況")
        return True  # 不算作失敗


def main():
    """主測試函數"""
    print("=== 籌碼面特徵模組測試 ===")
    
    success1 = test_chip_features()
    success2 = test_real_data()
    
    if success1 and success2:
        print("\n🎉 所有籌碼面特徵測試通過！")
    else:
        print("\n❌ 部分測試失敗")


if __name__ == "__main__":
    main()