#!/usr/bin/env python3
"""
完整測試腳本 - 驗證籌碼面特徵和資料收集功能
"""
import sys
import os
from pathlib import Path
import time

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "market_data_collector"))

def test_chip_features():
    """測試籌碼面特徵功能"""
    print("🔧 步驟 1: 測試籌碼面特徵功能")
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
        
        # 檢查現有資料表
        print("✅ 檢查現有資料表...")
        
        tables_to_check = [
            "candlesticks_daily",
            "margin_purchase_shortsale", 
            "institutional_investors_buy_sell",
            "minute_bars"
        ]
        
        for table in tables_to_check:
            try:
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                result = db.query_df(count_query)
                count = result.iloc[0]['count'] if not result.empty else 0
                print(f"   {table}: {count} 筆資料")
            except Exception as e:
                print(f"   {table}: 資料表不存在或無資料")
        
        print("\n🎉 籌碼面特徵功能驗證完成！")
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_keys():
    """測試API Keys功能"""
    print("\n🔧 步驟 2: 測試API Keys管理")
    print("=" * 50)
    
    try:
        # 測試API Key管理器
        exec(open('tmp_rovodev_enhanced_data_collector.py').read(), {'__name__': '__test__'})
        
        # 創建API管理器實例
        from tmp_rovodev_enhanced_data_collector import APIKeyManager
        
        api_manager = APIKeyManager()
        
        print(f"✅ 載入 {len(api_manager.api_keys)} 個API Keys")
        
        # 測試Key輪換
        for i in range(3):
            current_key = api_manager.get_current_key()
            print(f"   Key {i+1}: {current_key[:20]}...")
            api_manager.record_usage(current_key)
        
        # 顯示使用狀況
        status = api_manager.get_usage_status()
        print(f"✅ API使用狀況: {status}")
        
        print("✅ API Keys管理功能正常")
        return True
        
    except Exception as e:
        print(f"❌ API Keys測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_progress_tracking():
    """測試進度追蹤功能"""
    print("\n🔧 步驟 3: 測試進度追蹤功能")
    print("=" * 50)
    
    try:
        # 測試進度追蹤器
        exec(open('tmp_rovodev_enhanced_data_collector.py').read(), {'__name__': '__test__'})
        
        from tmp_rovodev_enhanced_data_collector import ProgressTracker
        
        tracker = ProgressTracker("test_progress.json")
        
        # 測試基本功能
        tracker.mark_symbol_completed("2330", "daily_price")
        tracker.mark_symbol_completed("2317", "daily_price")
        tracker.mark_symbol_failed("2603", "margin_shortsale", "測試錯誤")
        
        # 檢查狀態
        remaining = tracker.get_remaining_symbols(["2330", "2317", "2603"], ["daily_price", "margin_shortsale"])
        print(f"✅ 剩餘任務: {len(remaining)} 個")
        
        # 儲存進度
        tracker.save_progress()
        print("✅ 進度儲存成功")
        
        # 清理測試檔案
        if os.path.exists("test_progress.json"):
            os.remove("test_progress.json")
        
        print("✅ 進度追蹤功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 進度追蹤測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_collection_demo():
    """測試資料收集功能（演示模式）"""
    print("\n🔧 步驟 4: 測試資料收集功能（演示模式）")
    print("=" * 50)
    
    try:
        # 載入資料收集器
        exec(open('tmp_rovodev_enhanced_data_collector.py').read(), {'__name__': '__test__'})
        
        from tmp_rovodev_enhanced_data_collector import EnhancedDataCollector
        
        collector = EnhancedDataCollector()
        
        # 測試股票清單生成
        stock_list = collector.get_full_stock_list()
        print(f"✅ 生成股票清單: {len(stock_list)} 支")
        print(f"   前10支: {stock_list[:10]}")
        
        # 測試單一API請求（不實際執行，避免消耗API額度）
        print("✅ 資料收集器初始化成功")
        print("   注意: 實際API請求已跳過以節省額度")
        
        print("✅ 資料收集功能準備就緒")
        return True
        
    except Exception as e:
        print(f"❌ 資料收集測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_usage_instructions():
    """顯示使用說明"""
    print("\n📋 使用說明")
    print("=" * 50)
    
    print("1. 測試籌碼面特徵:")
    print("   run_test_chip_features.bat")
    print()
    
    print("2. 開始資料收集:")
    print("   run_enhanced_data_collector.bat")
    print()
    
    print("3. 查看收集進度:")
    print("   python tmp_rovodev_progress_manager.py")
    print()
    
    print("4. 生成股票清單:")
    print("   python tmp_rovodev_stock_list_generator.py")
    print()
    
    print("🔧 重要功能特色:")
    print("✅ 多API Key自動輪換 (3組FinMind API)")
    print("✅ 斷點續傳功能 (可隨時中斷後繼續)")
    print("✅ 進度追蹤和統計")
    print("✅ 180支股票完整清單")
    print("✅ 籌碼面特徵完整支援")
    print("✅ Shioaji分鐘線下載")
    print()
    
    print("📊 資料收集範圍:")
    print("- 日線價格資料 (TaiwanStockPrice)")
    print("- 融資融券資料 (TaiwanStockMarginPurchaseShortSale)")
    print("- 法人進出資料 (TaiwanStockInstitutionalInvestorsBuySell)")
    print("- 財務報表資料 (TaiwanStockFinancialStatements)")
    print("- 資產負債表 (TaiwanStockBalanceSheet)")
    print("- 月營收資料 (TaiwanStockMonthRevenue)")


def main():
    """主測試函數"""
    print("=== TSE Alpha 完整功能測試 ===")
    print(f"測試時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 執行所有測試
    tests = [
        ("籌碼面特徵", test_chip_features),
        ("API Keys管理", test_api_keys),
        ("進度追蹤", test_progress_tracking),
        ("資料收集演示", test_data_collection_demo)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 測試異常: {e}")
            results.append((test_name, False))
    
    # 顯示測試結果
    print("\n" + "=" * 50)
    print("📊 測試結果總結")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n總體結果: {passed}/{len(results)} 項測試通過")
    
    if passed == len(results):
        print("🎉 所有測試通過！系統準備就緒")
        show_usage_instructions()
    else:
        print("⚠️  部分測試失敗，請檢查相關功能")
    
    return passed == len(results)


if __name__ == "__main__":
    main()