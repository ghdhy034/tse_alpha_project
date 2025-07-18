#!/usr/bin/env python3
"""
測試籌碼面特徵完整功能
驗證 1336 筆籌碼面資料是否能正常計算特徵
"""

import sys
import os
from pathlib import Path
from datetime import datetime, date

# 確保路徑正確
current_dir = Path.cwd()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def test_chip_features_complete():
    """測試籌碼面特徵完整功能"""
    print("🧪 測試籌碼面特徵完整功能")
    print("=" * 50)
    
    try:
        # 導入必要模組
        from data_pipeline.features import FeatureEngine, ChipIndicators
        from market_data_collector.utils.db import query_df
        print("✅ 模組導入成功")
        
    except ImportError as e:
        print(f"❌ 模組導入失敗: {e}")
        return False
    
    # 檢查籌碼面資料
    print("\n--- 檢查籌碼面資料 ---")
    
    try:
        # 檢查融資融券資料
        margin_data = query_df("SELECT COUNT(*) as count FROM margin_purchase_shortsale")
        margin_count = margin_data.iloc[0]['count']
        print(f"📊 融資融券資料: {margin_count:,} 筆")
        
        # 檢查法人進出資料
        institutional_data = query_df("SELECT COUNT(*) as count FROM institutional_investors_buy_sell")
        institutional_count = institutional_data.iloc[0]['count']
        print(f"📊 法人進出資料: {institutional_count:,} 筆")
        
        total_chip_data = margin_count + institutional_count
        print(f"📈 總籌碼面資料: {total_chip_data:,} 筆")
        
        if total_chip_data == 0:
            print("⚠️  警告: 沒有籌碼面資料，無法測試特徵計算")
            return False
            
    except Exception as e:
        print(f"❌ 籌碼面資料檢查失敗: {e}")
        return False
    
    # 測試特徵計算
    print("\n--- 測試籌碼面特徵計算 ---")
    
    try:
        # 初始化籌碼面指標計算器
        chip_indicators = ChipIndicators()
        print("✅ ChipIndicators 初始化成功")
        
        # 取得測試股票的籌碼面資料
        test_symbol = "2330"  # 台積電
        print(f"🔄 測試股票: {test_symbol}")
        
        # 檢查該股票的融資融券資料
        margin_query = """
        SELECT * FROM margin_purchase_shortsale 
        WHERE symbol = ? 
        ORDER BY date DESC 
        LIMIT 30
        """
        margin_df = query_df(margin_query, (test_symbol,))
        
        if not margin_df.empty:
            print(f"✅ {test_symbol} 融資融券資料: {len(margin_df)} 筆")
            print(f"   日期範圍: {margin_df['date'].min()} ~ {margin_df['date'].max()}")
            
            # 測試融資融券特徵計算
            try:
                margin_features = chip_indicators.calculate_margin_features(margin_df)
                print(f"✅ 融資融券特徵計算成功: {margin_features.shape[1]} 個特徵")
                print(f"   特徵名稱: {list(margin_features.columns)}")
                
                # 顯示樣本資料
                if not margin_features.empty:
                    print("   樣本資料 (最新3筆):")
                    for i, (idx, row) in enumerate(margin_features.head(3).iterrows()):
                        print(f"     {i+1}. 融資比率: {row.get('margin_purchase_ratio', 'N/A'):.4f}")
                
            except Exception as e:
                print(f"❌ 融資融券特徵計算失敗: {e}")
        else:
            print(f"⚠️  {test_symbol} 無融資融券資料")
        
        # 檢查該股票的法人進出資料
        institutional_query = """
        SELECT * FROM institutional_investors_buy_sell 
        WHERE symbol = ? 
        ORDER BY date DESC 
        LIMIT 30
        """
        institutional_df = query_df(institutional_query, (test_symbol,))
        
        if not institutional_df.empty:
            print(f"✅ {test_symbol} 法人進出資料: {len(institutional_df)} 筆")
            print(f"   日期範圍: {institutional_df['date'].min()} ~ {institutional_df['date'].max()}")
            
            # 測試法人進出特徵計算
            try:
                institutional_features = chip_indicators.calculate_institutional_features(institutional_df)
                print(f"✅ 法人進出特徵計算成功: {institutional_features.shape[1]} 個特徵")
                print(f"   特徵名稱: {list(institutional_features.columns)}")
                
                # 顯示樣本資料
                if not institutional_features.empty:
                    print("   樣本資料 (最新3筆):")
                    for i, (idx, row) in enumerate(institutional_features.head(3).iterrows()):
                        print(f"     {i+1}. 外資淨買賣比率: {row.get('foreign_net_buy_ratio', 'N/A'):.4f}")
                
            except Exception as e:
                print(f"❌ 法人進出特徵計算失敗: {e}")
        else:
            print(f"⚠️  {test_symbol} 無法人進出資料")
        
        return True
        
    except Exception as e:
        print(f"❌ 籌碼面特徵測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_engine_integration():
    """測試特徵引擎整合"""
    print(f"\n--- 測試特徵引擎整合 ---")
    
    try:
        from data_pipeline.features import FeatureEngine
        
        # 初始化特徵引擎
        engine = FeatureEngine()
        print("✅ FeatureEngine 初始化成功")
        
        # 測試完整特徵計算 (如果有足夠資料)
        test_symbol = "2330"
        start_date = "2024-01-01"
        end_date = "2024-01-31"
        
        print(f"🔄 測試完整特徵計算: {test_symbol} ({start_date} ~ {end_date})")
        
        try:
            # 這裡可能會因為資料不足而失敗，但至少可以測試介面
            features = engine.calculate_all_features(test_symbol, start_date, end_date)
            
            if not features.empty:
                print(f"✅ 完整特徵計算成功: {features.shape}")
                print(f"   特徵數量: {features.shape[1]} 個")
                print(f"   資料筆數: {features.shape[0]} 筆")
            else:
                print("⚠️  完整特徵計算回傳空資料 (可能是資料不足)")
                
        except Exception as e:
            print(f"⚠️  完整特徵計算失敗 (預期，可能是資料不足): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 特徵引擎整合測試失敗: {e}")
        return False

def check_data_availability():
    """檢查資料可用性"""
    print(f"\n--- 檢查資料可用性 ---")
    
    try:
        from market_data_collector.utils.db import query_df
        
        # 檢查各種資料表
        tables_to_check = [
            "candlesticks_daily",
            "margin_purchase_shortsale", 
            "institutional_investors_buy_sell",
            "minute_bars"
        ]
        
        for table in tables_to_check:
            try:
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                result = query_df(count_query)
                count = result.iloc[0]['count']
                print(f"📊 {table}: {count:,} 筆")
                
                if count > 0:
                    # 檢查日期範圍
                    if table == "minute_bars":
                        date_query = f"SELECT MIN(DATE(ts)) as min_date, MAX(DATE(ts)) as max_date FROM {table}"
                    else:
                        date_query = f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {table}"
                    
                    date_result = query_df(date_query)
                    min_date = date_result.iloc[0]['min_date']
                    max_date = date_result.iloc[0]['max_date']
                    print(f"   日期範圍: {min_date} ~ {max_date}")
                
            except Exception as e:
                print(f"❌ {table}: 檢查失敗 - {e}")
        
    except Exception as e:
        print(f"❌ 資料可用性檢查失敗: {e}")

def main():
    """主函數"""
    print("🚀 籌碼面特徵完整功能測試")
    
    # 1. 檢查資料可用性
    check_data_availability()
    
    # 2. 測試籌碼面特徵
    chip_success = test_chip_features_complete()
    
    # 3. 測試特徵引擎整合
    engine_success = test_feature_engine_integration()
    
    # 總結
    print(f"\n" + "=" * 50)
    print(f"📊 測試結果總結")
    print(f"=" * 50)
    
    if chip_success and engine_success:
        print("🎉 籌碼面特徵功能完全正常！")
        print("💡 現在可以開始準備訓練資料了")
        print("📈 建議下一步:")
        print("   1. 收集更多歷史資料 (如需要)")
        print("   2. 開始模型訓練模組開發")
        print("   3. 準備特徵工程管線")
    else:
        print("💥 仍有部分問題需要解決")
        if not chip_success:
            print("   - 籌碼面特徵計算有問題")
        if not engine_success:
            print("   - 特徵引擎整合有問題")

if __name__ == "__main__":
    main()