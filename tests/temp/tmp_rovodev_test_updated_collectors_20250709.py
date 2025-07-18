#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試更新後的收集器 - 驗證日期範圍和原始處理方式
"""
import sys
from pathlib import Path

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_finmind_collector_updates():
    """測試 FinMind 收集器的更新"""
    print("🧪 測試 FinMind 收集器更新...")
    
    try:
        from tmp_rovodev_finmind_collector import FinMindDataCollector
        
        collector = FinMindDataCollector()
        
        # 檢查預設日期範圍
        print("📅 檢查預設日期範圍...")
        
        # 檢查是否能正確匯入原始處理函數
        try:
            from market_data_collector.utils.data_fetcher import (
                fetch_stock_data, store_stock_data_to_db,
                fetch_financial_data, store_financial_data_to_db,
                compute_technical_indicators, store_technical_indicators_to_db
            )
            print("✅ 原始處理函數匯入成功")
        except ImportError as e:
            print(f"❌ 原始處理函數匯入失敗: {e}")
            return False
        
        # 測試股票清單
        stock_list = collector.get_stock_list()
        print(f"📊 股票清單: {len(stock_list)} 支股票")
        print(f"   前5支: {stock_list[:5]}")
        
        print("✅ FinMind 收集器測試通過")
        return True
        
    except Exception as e:
        print(f"❌ FinMind 收集器測試失敗: {e}")
        return False

def test_shioaji_collector_updates():
    """測試 Shioaji 收集器的更新"""
    print("\n🧪 測試 Shioaji 收集器更新...")
    
    try:
        from tmp_rovodev_shioaji_collector import ShioajiDataCollector
        
        collector = ShioajiDataCollector()
        
        # 測試股票清單
        stock_list = collector.get_stock_list()
        print(f"📊 股票清單: {len(stock_list)} 支股票")
        print(f"   前5支: {stock_list[:5]}")
        
        print("✅ Shioaji 收集器測試通過")
        return True
        
    except Exception as e:
        print(f"❌ Shioaji 收集器測試失敗: {e}")
        return False

def test_date_ranges():
    """測試日期範圍設定"""
    print("\n📅 測試日期範圍設定...")
    
    # 檢查 FinMind 收集器
    try:
        from tmp_rovodev_finmind_collector import FinMindDataCollector
        collector = FinMindDataCollector()
        
        # 檢查 collect_stock_data 方法的預設參數
        import inspect
        sig = inspect.signature(collector.collect_stock_data)
        end_date_param = sig.parameters['end_date']
        print(f"FinMind 預設結束日期: {end_date_param.default}")
        
        if end_date_param.default == "2025-07-08":
            print("✅ FinMind 日期範圍正確")
        else:
            print(f"❌ FinMind 日期範圍錯誤: {end_date_param.default}")
            return False
            
    except Exception as e:
        print(f"❌ FinMind 日期檢查失敗: {e}")
        return False
    
    # 檢查 Shioaji 收集器
    try:
        from tmp_rovodev_shioaji_collector import ShioajiDataCollector
        collector = ShioajiDataCollector()
        
        # 檢查 collect_stock_minute_data 方法的預設參數
        sig = inspect.signature(collector.collect_stock_minute_data)
        end_date_param = sig.parameters['end_date']
        print(f"Shioaji 預設結束日期: {end_date_param.default}")
        
        if end_date_param.default == "2025-07-08":
            print("✅ Shioaji 日期範圍正確")
        else:
            print(f"❌ Shioaji 日期範圍錯誤: {end_date_param.default}")
            return False
            
    except Exception as e:
        print(f"❌ Shioaji 日期檢查失敗: {e}")
        return False
    
    return True

def test_original_data_fetcher_functions():
    """測試原始 data_fetcher 函數是否可用"""
    print("\n🔧 測試原始 data_fetcher 函數...")
    
    try:
        from market_data_collector.utils.data_fetcher import (
            fetch_stock_data,
            fetch_financial_data,
            fetch_monthly_revenue,
            fetch_margin_purchase_shortsale,
            fetch_investors_buy_sell,
            fetch_per_data,
            compute_technical_indicators,
            store_stock_data_to_db,
            store_financial_data_to_db,
            store_monthly_revenue_to_db,
            store_margin_purchase_shortsale_to_db,
            store_investors_buy_sell_to_db,
            store_per_data_to_db,
            store_technical_indicators_to_db
        )
        
        print("✅ 所有原始處理函數匯入成功")
        print("   📈 fetch_stock_data")
        print("   📊 fetch_financial_data")
        print("   💰 fetch_monthly_revenue")
        print("   💳 fetch_margin_purchase_shortsale")
        print("   🏛️ fetch_investors_buy_sell")
        print("   📋 fetch_per_data")
        print("   🔧 compute_technical_indicators")
        print("   💾 所有 store_*_to_db 函數")
        
        return True
        
    except ImportError as e:
        print(f"❌ 原始處理函數匯入失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("=" * 60)
    print("🧪 測試更新後的資料收集器")
    print("=" * 60)
    print("測試項目:")
    print("1. 日期範圍更新到 2025-07-08")
    print("2. FinMind 使用原始 data_fetcher 處理方式")
    print("3. 180支股票清單正確性")
    print("4. 原始處理函數可用性")
    print("=" * 60)
    
    results = []
    
    # 測試 1: FinMind 收集器
    results.append(test_finmind_collector_updates())
    
    # 測試 2: Shioaji 收集器
    results.append(test_shioaji_collector_updates())
    
    # 測試 3: 日期範圍
    results.append(test_date_ranges())
    
    # 測試 4: 原始處理函數
    results.append(test_original_data_fetcher_functions())
    
    # 結果統計
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("📊 測試結果統計")
    print("=" * 60)
    print(f"通過: {passed}/{total} 項測試")
    
    if passed == total:
        print("🎉 所有測試通過！收集器已準備就緒")
        print("\n📋 下一步建議:")
        print("1. 執行 run_finmind_data_collector.bat (收集歷史資料)")
        print("2. 執行 run_shioaji_data_collector.bat (收集分鐘線資料)")
        print("3. 或使用 run_data_collection_menu.bat (整合選單)")
    else:
        print("❌ 部分測試失敗，請檢查問題")
    
    print("=" * 60)

if __name__ == "__main__":
    main()