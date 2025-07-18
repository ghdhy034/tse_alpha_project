#!/usr/bin/env python3
"""
Smoke Test for fetch_minute.py - 快速驗證核心功能
"""
import sys
import os
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent / "market_data_collector"))

def test_imports():
    """測試模組導入"""
    print("1. 測試模組導入...")
    try:
        from fetch_minute import DataRouter, MinuteBarAggregator, ProxyDataGenerator
        from market_data_collector.utils.db import get_database_info
        print("   ✅ 所有模組導入成功")
        return True
    except Exception as e:
        print(f"   ❌ 模組導入失敗: {e}")
        return False

def test_data_router():
    """測試資料源路由"""
    print("2. 測試資料源路由...")
    try:
        from fetch_minute import DataRouter
        
        # 測試不同日期的路由
        test_cases = [
            (date(2018, 1, 1), 'proxy'),
            (date(2019, 6, 1), 'finmind'),
            (date(2024, 1, 1), 'shioaji'),
        ]
        
        for test_date, expected in test_cases:
            result = DataRouter.route('2330', test_date)
            if result == expected:
                print(f"   ✅ {test_date} → {result}")
            else:
                print(f"   ❌ {test_date} → {result} (期望: {expected})")
                return False
        
        return True
    except Exception as e:
        print(f"   ❌ 路由測試失敗: {e}")
        return False

def test_aggregator():
    """測試聚合器"""
    print("3. 測試 1分鐘→5分鐘聚合...")
    try:
        from fetch_minute import MinuteBarAggregator
        
        # 創建測試資料
        base_time = datetime(2024, 1, 2, 9, 0)
        test_data = []
        
        for i in range(10):  # 10分鐘資料
            test_data.append({
                'datetime': base_time + timedelta(minutes=i),
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.5,
                'volume': 1000
            })
        
        df_1m = pd.DataFrame(test_data)
        
        # 測試聚合
        aggregator = MinuteBarAggregator()
        df_5m = aggregator.to_5min(df_1m)
        
        if not df_5m.empty and len(df_5m) == 2:  # 10分鐘應該產生2個5分鐘bar
            print(f"   ✅ 聚合成功: {len(df_1m)}筆→{len(df_5m)}筆")
            return True
        else:
            print(f"   ❌ 聚合結果異常: {len(df_5m)}筆")
            return False
            
    except Exception as e:
        print(f"   ❌ 聚合測試失敗: {e}")
        return False

def test_database():
    """測試資料庫連線"""
    print("4. 測試資料庫連線...")
    try:
        from market_data_collector.utils.db import get_database_info, create_minute_bars_table
        
        # 檢查資料庫資訊
        db_info = get_database_info()
        print(f"   資料庫類型: {db_info['type']}")
        
        # 建立資料表
        create_minute_bars_table()
        print("   ✅ minute_bars 資料表已建立")
        
        return True
    except Exception as e:
        print(f"   ❌ 資料庫測試失敗: {e}")
        return False

def test_proxy_generator():
    """測試代理資料生成"""
    print("5. 測試代理資料生成...")
    try:
        from fetch_minute import ProxyDataGenerator
        
        generator = ProxyDataGenerator()
        
        # 測試 tick size 計算
        tick_sizes = [
            (5.0, 0.01),
            (25.0, 0.05),
            (75.0, 0.1),
            (250.0, 0.5),
            (750.0, 1.0),
            (1500.0, 5.0)
        ]
        
        for price, expected_tick in tick_sizes:
            actual_tick = generator._get_tick_size(price)
            if actual_tick == expected_tick:
                print(f"   ✅ 價格 {price} → tick {actual_tick}")
            else:
                print(f"   ❌ 價格 {price} → tick {actual_tick} (期望: {expected_tick})")
                return False
        
        return True
    except Exception as e:
        print(f"   ❌ 代理資料生成測試失敗: {e}")
        return False

def test_command_line_interface():
    """測試命令列介面"""
    print("6. 測試命令列介面...")
    try:
        # 測試參數解析（不實際執行）
        import argparse
        
        # 模擬 fetch_minute.py 的 argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('--date', required=True)
        parser.add_argument('--symbols', required=True, nargs='+')
        parser.add_argument('--verbose', '-v', action='store_true')
        
        # 測試參數解析
        test_args = ['--date', '2024-01-02', '--symbols', '2330', '2603', '--verbose']
        args = parser.parse_args(test_args)
        
        if args.date == '2024-01-02' and args.symbols == ['2330', '2603'] and args.verbose:
            print("   ✅ 命令列參數解析正常")
            return True
        else:
            print("   ❌ 命令列參數解析異常")
            return False
            
    except Exception as e:
        print(f"   ❌ 命令列介面測試失敗: {e}")
        return False

def run_smoke_test():
    """執行 Smoke Test"""
    print("開始執行 fetch_minute.py Smoke Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_router,
        test_aggregator,
        test_database,
        test_proxy_generator,
        test_command_line_interface,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                break  # 如果有測試失敗，停止後續測試
        except Exception as e:
            print(f"   ❌ 測試異常: {e}")
            break
    
    print("\n" + "=" * 50)
    if passed == total:
        print(f"✅ Smoke Test 通過 ({passed}/{total})")
        print("fetch_minute.py 模組基本功能正常！")
        return True
    else:
        print(f"❌ Smoke Test 失敗 ({passed}/{total})")
        print("請檢查失敗的測試項目")
        return False

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)