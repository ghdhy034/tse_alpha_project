#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試修正後的程式：API查詢、high/low欄位、volume單位轉換
"""
import sys
import logging
from pathlib import Path

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "market_data_collector"))

# 設定日誌
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from tmp_rovodev_finmind_collector_fixed import FinMindDataCollector
    
    print("🧪 測試修正後的程式...")
    print("=" * 50)
    
    # 1. 測試初始化和 API 查詢
    print("1️⃣ 測試初始化和 API 使用狀況查詢...")
    collector = FinMindDataCollector()
    
    # 檢查 DataLoader 的 api_usage_limit 屬性
    print(f"✅ DataLoader 實例: {collector.dl}")
    if hasattr(collector.dl, 'api_usage_limit'):
        print(f"✅ DataLoader.api_usage_limit: {collector.dl.api_usage_limit}")
    else:
        print("⚠️ DataLoader 沒有 api_usage_limit 屬性")
    
    # 檢查 API 使用狀況
    usage = collector.api_manager.get_usage_status()
    print(f"✅ API Key: {usage['api_key']}")
    print(f"✅ 當前使用量: {usage['current_usage']}/{usage['usage_limit']}")
    print(f"✅ 剩餘次數: {usage['remaining_calls']}")
    
    # 直接測試 get_api_usage_status
    current_usage, usage_limit = collector.api_manager.get_api_usage_status()
    print(f"✅ 直接查詢結果: {current_usage}/{usage_limit}")
    
    # 2. 測試單一股票資料下載
    print("\n2️⃣ 測試單一股票資料下載 (台積電 2330)...")
    test_symbol = "2330"
    
    # 下載一小段時間的資料進行測試
    df_daily = collector.fetch_stock_data_sdk(
        collector.dl, 
        test_symbol, 
        start_date="2024-01-01", 
        end_date="2024-01-05"
    )
    
    if not df_daily.empty:
        print(f"✅ 成功下載 {len(df_daily)} 筆日線資料")
        print(f"✅ 資料欄位: {list(df_daily.columns)}")
        
        # 檢查 high/low 欄位
        if "high" in df_daily.columns and "low" in df_daily.columns:
            print(f"✅ high/low 欄位存在")
            print(f"✅ 第一筆資料 high: {df_daily.iloc[0]['high']}")
            print(f"✅ 第一筆資料 low: {df_daily.iloc[0]['low']}")
        else:
            print("❌ high/low 欄位缺失")
        
        # 檢查 volume 單位轉換
        if "Trading_Volume" in df_daily.columns:
            volume_sample = df_daily.iloc[0]["Trading_Volume"]
            print(f"✅ Volume (張): {volume_sample}")
            if volume_sample < 1000000:  # 轉換後應該小於原始值
                print("✅ Volume 單位轉換正確 (已轉換為張)")
            else:
                print("⚠️ Volume 可能未正確轉換")
        
        # 顯示第一筆完整資料
        print(f"✅ 第一筆完整資料:")
        print(df_daily.iloc[0].to_dict())
    else:
        print("❌ 未能下載到資料")
    
    # 3. 測試 API 使用量變化
    print("\n3️⃣ 測試 API 使用量變化...")
    usage_before = collector.api_manager.get_usage_status()
    print(f"下載前使用量: {usage_before['current_usage']}")
    
    # 再下載一次
    df_daily2 = collector.fetch_stock_data_sdk(
        collector.dl, 
        "2317", 
        start_date="2024-01-01", 
        end_date="2024-01-02"
    )
    
    usage_after = collector.api_manager.get_usage_status()
    print(f"下載後使用量: {usage_after['current_usage']}")
    
    if usage_after['current_usage'] > usage_before['current_usage']:
        print("✅ API 使用量正確增加")
    else:
        print("⚠️ API 使用量未增加（可能使用本地計數器）")
    
    print("\n✅ 所有測試完成")
    
except Exception as e:
    print(f"❌ 測試失敗: {e}")
    import traceback
    traceback.print_exc()