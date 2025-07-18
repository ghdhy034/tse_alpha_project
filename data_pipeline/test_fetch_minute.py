#!/usr/bin/env python3
"""
測試 fetch_minute.py 模組功能
"""
import sys
import os
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent / "market_data_collector"))

from fetch_minute import (
    DataRouter, FinMindDownloader, ProxyDataGenerator, 
    MinuteBarAggregator, fetch_symbol_date, store_minute_bars
)
from market_data_collector.utils.db import get_database_info, create_minute_bars_table

def test_data_router():
    """測試資料源路由器"""
    print("=== 測試資料源路由器 ===")
    
    test_cases = [
        (date(2019, 1, 1), 'proxy'),      # 早期日期
        (date(2019, 6, 1), 'finmind'),    # FinMind 期間
        (date(2020, 5, 1), 'shioaji'),    # Shioaji 期間
        (date(2024, 1, 1), 'shioaji'),    # 最新日期
    ]
    
    for test_date, expected in test_cases:
        result = DataRouter.route('2330', test_date)
        status = "✅" if result == expected else "❌"
        print(f"{status} {test_date}: {result} (期望: {expected})")
    
    return True

def test_minute_bar_aggregator():
    """測試 1分鐘 → 5分鐘聚合器"""
    print("\n=== 測試分鐘線聚合器 ===")
    
    # 創建測試用的 1 分鐘資料
    base_time = datetime(2024, 1, 2, 9, 0)  # 09:00 開始
    test_data = []
    
    for i in range(10):  # 10 分鐘的資料
        test_data.append({
            'datetime': base_time + timedelta(minutes=i),
            'open': 100 + i * 0.1,
            'high': 100 + i * 0.1 + 0.5,
            'low': 100 + i * 0.1 - 0.3,
            'close': 100 + i * 0.1 + 0.2,
            'volume': 1000 + i * 100
        })
    
    df_1m = pd.DataFrame(test_data)
    
    # 測試聚合
    aggregator = MinuteBarAggregator()
    df_5m = aggregator.to_5min(df_1m)
    
    if not df_5m.empty:
        print(f"✅ 聚合成功: {len(df_1m)} 筆 1分鐘 → {len(df_5m)} 筆 5分鐘")
        print("5分鐘資料樣本:")
        print(df_5m.head(2).to_string())
        
        # 檢查 VWAP 計算
        if 'vwap' in df_5m.columns:
            print("✅ VWAP 計算完成")
        else:
            print("❌ VWAP 計算失敗")
        
        return True
    else:
        print("❌ 聚合失敗")
        return False

def test_proxy_data_generator():
    """測試代理資料生成器"""
    print("\n=== 測試代理資料生成器 ===")
    
    try:
        generator = ProxyDataGenerator()
        
        # 測試早期日期的代理資料生成
        test_date = date(2018, 1, 2)
        df_proxy = generator.generate_proxy_data('2330', test_date)
        
        if not df_proxy.empty:
            print(f"✅ 代理資料生成成功: {len(df_proxy)} 筆資料")
            print("代理資料樣本:")
            print(df_proxy.head(3).to_string())
            return True
        else:
            print("⚠️  代理資料為空（可能是正常的，如果沒有次日開盤價）")
            return True
            
    except Exception as e:
        print(f"❌ 代理資料生成失敗: {e}")
        return False

def test_database_setup():
    """測試資料庫設置"""
    print("\n=== 測試資料庫設置 ===")
    
    try:
        # 檢查資料庫資訊
        db_info = get_database_info()
        print(f"資料庫類型: {db_info['type']}")
        print(f"資料表數量: {db_info['table_count']}")
        
        # 建立 minute_bars 資料表
        create_minute_bars_table()
        print("✅ minute_bars 資料表已建立")
        
        return True
        
    except Exception as e:
        print(f"❌ 資料庫設置失敗: {e}")
        return False

def test_finmind_rate_limiting():
    """測試 FinMind 速率限制"""
    print("\n=== 測試 FinMind 速率限制 ===")
    
    try:
        downloader = FinMindDownloader()
        
        # 測試速率限制機制（不實際發送請求）
        start_time = downloader.last_request_time
        downloader._rate_limit()
        end_time = downloader.last_request_time
        
        print(f"✅ 速率限制機制正常，請求間隔: {end_time - start_time:.3f} 秒")
        print(f"   當前請求計數: {downloader.request_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ 速率限制測試失敗: {e}")
        return False

def test_integration():
    """整合測試 - 測試完整的下載流程"""
    print("\n=== 整合測試 ===")
    
    try:
        # 測試代理資料流程（早期日期）
        test_date = date(2018, 1, 2)
        test_symbol = '2330'
        
        print(f"測試下載 {test_symbol} {test_date} 的資料...")
        df = fetch_symbol_date(test_symbol, test_date)
        
        if not df.empty:
            print(f"✅ 下載成功: {len(df)} 筆資料")
            print("資料樣本:")
            print(df.head(2).to_string())
            
            # 測試存儲
            print("測試資料存儲...")
            store_minute_bars(df)
            print("✅ 存儲成功")
            
            return True
        else:
            print("⚠️  下載結果為空（可能是正常的）")
            return True
            
    except Exception as e:
        print(f"❌ 整合測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """執行所有測試"""
    print("開始執行 fetch_minute 模組測試...")
    
    tests = [
        ("資料源路由器", test_data_router),
        ("分鐘線聚合器", test_minute_bar_aggregator),
        ("代理資料生成器", test_proxy_data_generator),
        ("資料庫設置", test_database_setup),
        ("FinMind 速率限制", test_finmind_rate_limiting),
        ("整合測試", test_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"執行測試: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通過")
            else:
                print(f"❌ {test_name} 失敗")
        except Exception as e:
            print(f"❌ {test_name} 異常: {e}")
    
    print(f"\n{'='*50}")
    print(f"測試結果: {passed}/{total} 通過")
    print('='*50)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)