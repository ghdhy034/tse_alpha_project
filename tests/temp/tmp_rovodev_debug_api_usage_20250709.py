#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
專門調試 DataLoader 的 api_usage_limit 屬性
"""
import sys
from pathlib import Path

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "market_data_collector"))

try:
    from FinMind.data import DataLoader
    from market_data_collector.utils.config import FINMIND_USER, FINMIND_PASS
    
    print("🔍 調試 DataLoader 的 api_usage_limit 屬性")
    print("=" * 50)
    
    # 1. 創建 DataLoader 實例
    print("1️⃣ 創建 DataLoader 實例...")
    dl = DataLoader()
    print(f"✅ DataLoader 創建成功: {dl}")
    
    # 2. 檢查登入前的屬性
    print("\n2️⃣ 檢查登入前的屬性...")
    print(f"DataLoader 所有屬性: {dir(dl)}")
    
    if hasattr(dl, 'api_usage_limit'):
        print(f"登入前 api_usage_limit: {dl.api_usage_limit}")
        print(f"登入前 api_usage_limit 類型: {type(dl.api_usage_limit)}")
    else:
        print("登入前沒有 api_usage_limit 屬性")
    
    # 3. 執行登入
    print(f"\n3️⃣ 執行登入...")
    print(f"使用帳號: {FINMIND_USER}")
    
    login_result = dl.login(user_id=FINMIND_USER, password=FINMIND_PASS)
    print(f"登入結果: {login_result}")
    print(f"登入結果類型: {type(login_result)}")
    
    # 4. 檢查登入後的屬性 (根據 References.txt)
    print("\n4️⃣ 檢查登入後的屬性 (根據 References.txt)...")
    
    # 檢查 api_usage (已使用次數)
    if hasattr(dl, 'api_usage'):
        print(f"✅ 找到 api_usage 屬性")
        print(f"api_usage 值: {dl.api_usage}")
        print(f"api_usage 類型: {type(dl.api_usage)}")
    else:
        print("❌ 沒有 api_usage 屬性")
    
    # 檢查 api_usage_limit (上限)
    if hasattr(dl, 'api_usage_limit'):
        print(f"✅ 找到 api_usage_limit 屬性")
        print(f"api_usage_limit 值: {dl.api_usage_limit}")
        print(f"api_usage_limit 類型: {type(dl.api_usage_limit)}")
    else:
        print("❌ 沒有 api_usage_limit 屬性")
    
    # 根據 References.txt 的範例
    if hasattr(dl, 'api_usage') and hasattr(dl, 'api_usage_limit'):
        used = dl.api_usage
        limit = dl.api_usage_limit
        print(f"📊 References.txt 範例結果: {used} / {limit}")
    else:
        print("❌ 無法使用 References.txt 的範例方式")
    
    # 5. 檢查其他可能的使用量屬性
    print("\n5️⃣ 檢查其他可能的使用量相關屬性...")
    possible_attrs = ['usage', 'limit', 'count', 'api_count', 'user_count', 'request_limit']
    
    for attr in possible_attrs:
        if hasattr(dl, attr):
            value = getattr(dl, attr)
            print(f"✅ 找到屬性 {attr}: {value} (類型: {type(value)})")
    
    # 6. 測試一次 API 調用看看是否會更新
    print("\n6️⃣ 測試 API 調用後的變化...")
    try:
        # 調用一個簡單的 API
        test_df = dl.taiwan_stock_daily(stock_id="2330", start_date="2024-01-01", end_date="2024-01-01")
        print(f"✅ API 調用成功，返回 {len(test_df)} 筆資料")
        
        # 再次檢查使用量 (根據 References.txt)
        if hasattr(dl, 'api_usage') and hasattr(dl, 'api_usage_limit'):
            used = dl.api_usage
            limit = dl.api_usage_limit
            print(f"API 調用後使用量: {used} / {limit}")
        else:
            print("❌ API 調用後仍無法取得使用量")
        
    except Exception as e:
        print(f"❌ API 調用失敗: {e}")
    
    print("\n✅ 調試完成")
    
except Exception as e:
    print(f"❌ 調試失敗: {e}")
    import traceback
    traceback.print_exc()