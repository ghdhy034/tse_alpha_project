#!/usr/bin/env python3
"""
測試籌碼面特徵完整功能
"""
import sys
import os
from pathlib import Path

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "market_data_collector"))

def test_chip_features():
    """測試籌碼面特徵功能"""
    print("🔧 測試籌碼面特徵功能")
    print("=" * 50)
    
    try:
        # 測試模組導入
        print("✅ 測試模組導入...")
        import market_data_collector
        from market_data_collector.utils import config
        from market_data_collector.utils import db
        from data_pipeline import features
        
        print("✅ 所有模組導入成功")
        
        # 測試特徵引擎初始化
        print("✅ 測試特徵引擎初始化...")
        engine = features.FeatureEngine(['2330', '2317'])
        chip_indicators = features.ChipIndicators()
        
        print("✅ 特徵引擎初始化成功")
        
        # 測試資料庫連接
        print("✅ 測試資料庫連接...")
        conn = db.get_conn()
        print("✅ 資料庫連接成功")
        
        # 檢查籌碼面資料表
        print("✅ 檢查籌碼面資料表...")
        
        # 檢查融資融券資料
        margin_query = "SELECT COUNT(*) as count FROM margin_purchase_shortsale"
        try:
            margin_result = db.query_df(margin_query)
            margin_count = margin_result.iloc[0]['count'] if not margin_result.empty else 0
            print(f"   融資融券資料: {margin_count} 筆")
        except Exception as e:
            print(f"   融資融券資料表不存在或無資料: {e}")
            margin_count = 0
        
        # 檢查法人進出資料
        institutional_query = "SELECT COUNT(*) as count FROM institutional_investors_buy_sell"
        try:
            inst_result = db.query_df(institutional_query)
            inst_count = inst_result.iloc[0]['count'] if not inst_result.empty else 0
            print(f"   法人進出資料: {inst_count} 筆")
        except Exception as e:
            print(f"   法人進出資料表不存在或無資料: {e}")
            inst_count = 0
        
        # 測試特徵計算
        print("✅ 測試特徵計算...")
        test_symbol = '2330'
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        
        # 載入價格資料
        price_data = engine.load_price_data(test_symbol, start_date, end_date)
        print(f"   價格資料: {len(price_data)} 筆")
        
        # 載入籌碼面資料
        chip_data = engine.load_chip_data(test_symbol, start_date, end_date)
        print(f"   籌碼面資料源: {list(chip_data.keys())}")
        
        # 計算技術特徵
        tech_features = engine.calculate_technical_features(price_data)
        print(f"   技術特徵: {tech_features.shape[1]} 個")
        
        # 計算籌碼面特徵
        chip_features = engine.calculate_chip_features(test_symbol, price_data, chip_data)
        print(f"   籌碼面特徵: {chip_features.shape[1]} 個")
        
        # 計算基本面特徵
        fundamental_features = engine.calculate_fundamental_features(test_symbol, price_data)
        print(f"   基本面特徵: {fundamental_features.shape[1]} 個")
        
        print("\n" + "=" * 50)
        print("🎉 籌碼面特徵功能測試完成！")
        print(f"💡 總資料量: 融資融券 {margin_count} 筆, 法人進出 {inst_count} 筆")
        print(f"💡 特徵維度: 技術 {tech_features.shape[1]}, 籌碼 {chip_features.shape[1]}, 基本面 {fundamental_features.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_chip_features()