#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試修正後的 Shioaji 收集器
"""
import sys
from pathlib import Path

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_shioaji_collector_fixed():
    """測試修正後的 Shioaji 收集器"""
    print("🧪 測試修正後的 Shioaji 收集器...")
    
    try:
        from tmp_rovodev_shioaji_collector_updated import ShioajiDataCollector, SimpleFlowMonitor
        
        # 測試 SimpleFlowMonitor
        print("📊 測試簡單流量監控器...")
        monitor = SimpleFlowMonitor()
        monitor.add_usage(1024 * 1024)  # 1MB
        monitor.show_status()
        print("✅ 簡單流量監控器正常")
        
        # 測試收集器初始化
        print("🔧 測試收集器初始化...")
        collector = ShioajiDataCollector()
        print("✅ 收集器初始化成功")
        
        # 測試股票清單
        stock_list = collector.get_stock_list()
        print(f"📊 股票清單: {len(stock_list)} 支")
        print(f"   前5支: {stock_list[:5]}")
        
        # 測試格式化函數
        print("🔧 測試格式化函數...")
        import pandas as pd
        from datetime import datetime
        
        # 建立測試資料
        test_data = pd.DataFrame({
            'ts': [datetime(2024, 12, 1, 9, 0), datetime(2024, 12, 1, 9, 5)],
            'Open': [100.0, 101.0],
            'High': [102.0, 103.0],
            'Low': [99.0, 100.0],
            'Close': [101.0, 102.0],
            'Volume': [1000, 1500]
        })
        
        formatted_data = collector.format_for_candlesticks_min(test_data, "2330")
        
        if not formatted_data.empty:
            print("✅ 格式化函數正常")
            print(f"   格式化後欄位: {list(formatted_data.columns)}")
            print(f"   資料筆數: {len(formatted_data)}")
        else:
            print("❌ 格式化函數失敗")
            return False
        
        print("✅ 所有測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_table():
    """測試資料表是否正確建立"""
    print("\n🗄️ 測試資料表...")
    
    try:
        from market_data_collector.utils.data_fetcher import create_db_and_table
        from market_data_collector.utils.db import query_df
        
        # 建立資料表
        create_db_and_table()
        print("✅ 資料表建立完成")
        
        # 檢查 candlesticks_min 資料表結構
        try:
            schema_query = "PRAGMA table_info(candlesticks_min)"
            schema_df = query_df(schema_query)
            
            if not schema_df.empty:
                print("✅ candlesticks_min 資料表存在")
                print("📋 資料表結構:")
                for _, row in schema_df.iterrows():
                    print(f"   {row['name']}: {row['type']} {'(NOT NULL)' if row['notnull'] else ''}")
            else:
                print("❌ candlesticks_min 資料表不存在")
                return False
                
        except Exception as e:
            print(f"❌ 檢查資料表結構失敗: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 資料表測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("=" * 60)
    print("🧪 Shioaji 收集器修正測試")
    print("=" * 60)
    print("測試項目:")
    print("1. 修正後的收集器功能")
    print("2. 簡單流量監控器")
    print("3. 格式化函數")
    print("4. 資料表結構")
    print("=" * 60)
    
    results = []
    
    # 測試 1: 收集器功能
    results.append(test_shioaji_collector_fixed())
    
    # 測試 2: 資料表
    results.append(test_database_table())
    
    # 結果統計
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("📊 測試結果統計")
    print("=" * 60)
    print(f"通過: {passed}/{total} 項測試")
    
    if passed == total:
        print("🎉 所有測試通過！修正版收集器已準備就緒")
        print("\n📋 下一步建議:")
        print("1. 執行 run_shioaji_data_collector_updated.bat")
        print("2. 選擇測試模式驗證修正效果")
    else:
        print("❌ 部分測試失敗，請檢查問題")
    
    print("=" * 60)

if __name__ == "__main__":
    main()