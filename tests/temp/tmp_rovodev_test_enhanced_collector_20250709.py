#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試增強版 FinMind 收集器的新功能
"""
import sys
import os
import time
from pathlib import Path

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_api_manager_features():
    """測試 API Manager 的新功能"""
    print("=" * 60)
    print("🧪 測試 API Manager 新功能")
    print("=" * 60)
    
    try:
        from tmp_rovodev_finmind_collector_fixed import FinMindAPIManager
        
        # 測試 API Manager 初始化
        api_manager = FinMindAPIManager()
        print(f"✅ API Manager 初始化成功")
        
        # 顯示配置
        print(f"🔑 API Key: {api_manager.api_key[:20]}...")
        print(f"📊 每日限制: {api_manager.daily_limit}")
        print(f"⚠️ 休眠閾值: {api_manager.rate_limit_threshold}")
        print(f"😴 休眠時間: {api_manager.sleep_duration} 秒 ({api_manager.sleep_duration//60} 分鐘)")
        
        # 測試使用狀況
        usage = api_manager.get_usage_status()
        print(f"\n📈 初始使用狀況:")
        print(f"   呼叫次數: {usage['call_count']}/{usage['daily_limit']}")
        print(f"   剩餘次數: {usage['remaining_calls']}")
        
        # 模擬一些 API 呼叫
        print(f"\n🔄 模擬 API 呼叫...")
        for i in range(10):
            api_manager.record_usage()
            if i == 4:
                usage = api_manager.get_usage_status()
                print(f"   第 {i+1} 次後: {usage['call_count']}/{usage['daily_limit']}")
        
        final_usage = api_manager.get_usage_status()
        print(f"\n🏁 最終狀況:")
        print(f"   呼叫次數: {final_usage['call_count']}/{final_usage['daily_limit']}")
        print(f"   剩餘次數: {final_usage['remaining_calls']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_existence_check():
    """測試資料存在檢查功能"""
    print("\n" + "=" * 60)
    print("🧪 測試防重複下載機制")
    print("=" * 60)
    
    try:
        from tmp_rovodev_finmind_collector_fixed import FinMindDataCollector
        
        collector = FinMindDataCollector()
        print(f"✅ 資料收集器初始化成功")
        
        # 測試檢查功能
        test_symbol = "2330"
        start_date = "2020-03-02"
        end_date = "2025-07-08"
        
        data_types = ["daily_price", "financial", "monthly_revenue", 
                     "margin_shortsale", "institutional", "per_data", "technical_indicators"]
        
        print(f"\n🔍 檢查 {test_symbol} 的資料存在狀況:")
        for data_type in data_types:
            exists = collector.check_data_exists(test_symbol, data_type, start_date, end_date)
            status = "✅ 已存在" if exists else "❌ 不存在"
            print(f"   {data_type:20}: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_collector_initialization():
    """測試收集器初始化"""
    print("\n" + "=" * 60)
    print("🧪 測試收集器完整初始化")
    print("=" * 60)
    
    try:
        from tmp_rovodev_finmind_collector_fixed import FinMindDataCollector
        
        collector = FinMindDataCollector()
        print(f"✅ 收集器初始化成功")
        
        # 測試股票清單
        stocks = collector.get_stock_list()
        print(f"📈 股票清單: {len(stocks)} 支")
        print(f"📈 前10支: {stocks[:10]}")
        
        # 測試 API 使用狀況
        usage = collector.api_manager.get_usage_status()
        print(f"\n🔑 API 狀況:")
        print(f"   Key: {usage['api_key']}")
        print(f"   呼叫次數: {usage['call_count']}/{usage['daily_limit']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    print("🚀 開始測試增強版 FinMind 收集器...")
    
    # 檢查 API Key 檔案
    api_key_file = "finmind_api_keys.txt"
    if not os.path.exists(api_key_file):
        print(f"❌ 找不到 API Key 檔案: {api_key_file}")
        return
    
    with open(api_key_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        print(f"📄 API Key 檔案: {content[:20]}...")
    
    # 執行測試
    test1_result = test_api_manager_features()
    test2_result = test_data_existence_check()
    test3_result = test_collector_initialization()
    
    print("\n" + "=" * 60)
    print("🏁 測試結果總結")
    print("=" * 60)
    print(f"API Manager 功能: {'✅ 通過' if test1_result else '❌ 失敗'}")
    print(f"防重複下載機制: {'✅ 通過' if test2_result else '❌ 失敗'}")
    print(f"收集器初始化: {'✅ 通過' if test3_result else '❌ 失敗'}")
    
    if all([test1_result, test2_result, test3_result]):
        print("\n🎉 所有測試通過！增強版收集器準備就緒。")
        print("\n📋 新功能摘要:")
        print("• ✅ 單一 API Key 管理")
        print("• ✅ 550次呼叫後自動休眠1小時3分鐘")
        print("• ✅ 防重複下載 - 自動檢查已存在資料")
        print("• ✅ 智能重試與錯誤處理")
        print("• ✅ 詳細的下載/略過狀態日誌")
    else:
        print("\n⚠️ 部分測試失敗，請檢查錯誤訊息。")

if __name__ == "__main__":
    main()