#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
調試 Shioaji 收集器的資料格式化問題
"""
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "market_data_collector"))

def debug_data_formatting():
    """調試資料格式化過程"""
    print("🔍 調試 Shioaji 資料格式化...")
    
    # 建立模擬的5分鐘資料
    test_data = pd.DataFrame({
        'ts': [
            datetime(2024, 12, 1, 9, 0),
            datetime(2024, 12, 1, 9, 5),
            datetime(2024, 12, 1, 9, 10)
        ],
        'Open': [100.0, 101.0, 102.0],
        'High': [102.0, 103.0, 104.0],
        'Low': [99.0, 100.0, 101.0],
        'Close': [101.0, 102.0, 103.0],
        'Volume': [1000, 1500, 2000]
    })
    
    print("📊 測試資料:")
    print(test_data)
    print(f"📋 欄位類型: {test_data.dtypes}")
    
    # 測試格式化函數
    try:
        from tmp_rovodev_shioaji_collector_updated import ShioajiDataCollector
        
        collector = ShioajiDataCollector()
        formatted_data = collector.format_for_candlesticks_min(test_data, "2330")
        
        print("\n📊 格式化後資料:")
        if not formatted_data.empty:
            print(formatted_data)
            print(f"📋 欄位類型: {formatted_data.dtypes}")
            
            # 檢查關鍵欄位
            print("\n🔍 關鍵欄位檢查:")
            print(f"market: {formatted_data['market'].tolist()}")
            print(f"symbol: {formatted_data['symbol'].tolist()}")
            print(f"timestamp: {formatted_data['timestamp'].tolist()}")
            print(f"open: {formatted_data['open'].tolist()}")
            
            return True
        else:
            print("❌ 格式化後資料為空")
            return False
            
    except Exception as e:
        print(f"❌ 格式化測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_database_data():
    """檢查資料庫中的資料"""
    print("\n🗄️ 檢查資料庫中的資料...")
    
    try:
        from market_data_collector.utils.db import query_df
        
        # 查詢 candlesticks_min 資料表
        query = "SELECT * FROM candlesticks_min LIMIT 5"
        df = query_df(query)
        
        if not df.empty:
            print("📊 資料庫中的資料:")
            print(df)
            print(f"📋 欄位類型: {df.dtypes}")
            
            # 檢查問題欄位
            print("\n🔍 問題檢查:")
            print(f"market 唯一值: {df['market'].unique()}")
            print(f"symbol 唯一值: {df['symbol'].unique()}")
            
            if (df['market'] == 0).any():
                print("❌ 發現 market 欄位為 0 的問題")
            if (df['symbol'] == 0).any():
                print("❌ 發現 symbol 欄位為 0 的問題")
                
        else:
            print("⚠️  資料庫中無資料")
            
    except Exception as e:
        print(f"❌ 檢查資料庫失敗: {e}")

def test_insert_process():
    """測試完整的插入過程"""
    print("\n🧪 測試完整插入過程...")
    
    try:
        from market_data_collector.utils.db import insert_df
        
        # 建立正確的測試資料
        test_data = pd.DataFrame({
            'market': ['TW', 'TW'],
            'symbol': ['TEST', 'TEST'],
            'timestamp': ['2024-12-01 09:00:00', '2024-12-01 09:05:00'],
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [99.0, 100.0],
            'close': [101.0, 102.0],
            'volume': [1000, 1500],
            'interval': ['5min', '5min']
        })
        
        print("📊 測試插入資料:")
        print(test_data)
        
        # 嘗試插入
        insert_df('candlesticks_min', test_data, if_exists='append')
        print("✅ 測試插入成功")
        
        # 驗證插入結果
        from market_data_collector.utils.db import query_df
        verify_query = "SELECT * FROM candlesticks_min WHERE symbol = 'TEST'"
        verify_df = query_df(verify_query)
        
        print("📊 驗證插入結果:")
        print(verify_df)
        
        return True
        
    except Exception as e:
        print(f"❌ 測試插入失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    print("=" * 60)
    print("🔍 Shioaji 收集器資料格式化調試")
    print("=" * 60)
    
    results = []
    
    # 測試 1: 資料格式化
    results.append(debug_data_formatting())
    
    # 測試 2: 檢查資料庫
    check_database_data()
    
    # 測試 3: 測試插入過程
    results.append(test_insert_process())
    
    # 結果統計
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("📊 調試結果")
    print("=" * 60)
    print(f"通過: {passed}/{total} 項測試")
    
    if passed == total:
        print("✅ 格式化功能正常")
    else:
        print("❌ 發現格式化問題，需要進一步修正")

if __name__ == "__main__":
    main()