#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 FinMind API 回傳資料格式
"""
import requests
import pandas as pd
import json

def test_finmind_api_response():
    """測試 FinMind API 回傳的資料格式"""
    
    # 從 config 取得 API 設定
    try:
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir / "market_data_collector"))
        
        from market_data_collector.utils.config import API_ENDPOINT, TOKEN
    except ImportError as e:
        print(f"❌ 無法匯入設定: {e}")
        return
    
    # 測試股票價格資料
    print("🧪 測試 TaiwanStockPrice API 回傳格式...")
    
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": "2330",
        "start_date": "2024-12-01",
        "end_date": "2024-12-05",
        "token": TOKEN
    }
    
    try:
        response = requests.get(API_ENDPOINT, params=params, timeout=30)
        
        if response.status_code == 200:
            json_data = response.json()
            
            if json_data.get("status") == 200 and "data" in json_data:
                data = json_data["data"]
                
                if data:
                    df = pd.DataFrame(data)
                    print("✅ API 回傳成功")
                    print(f"📊 資料筆數: {len(df)}")
                    print(f"📋 欄位名稱: {list(df.columns)}")
                    print("\n📝 前3筆資料:")
                    print(df.head(3).to_string())
                    
                    print("\n🔍 欄位對應檢查:")
                    expected_mapping = {
                        "open": "open",
                        "high": "max",  # FinMind 使用 max
                        "low": "min",   # FinMind 使用 min
                        "close": "close",
                        "volume": "Trading_Volume"  # FinMind 使用 Trading_Volume
                    }
                    
                    for db_col, api_col in expected_mapping.items():
                        if api_col in df.columns:
                            print(f"✅ {db_col} ← {api_col}")
                        else:
                            print(f"❌ {db_col} ← {api_col} (缺失)")
                    
                    return df
                else:
                    print("❌ API 回傳空資料")
            else:
                print(f"❌ API 回傳錯誤: {json_data}")
        else:
            print(f"❌ HTTP 錯誤: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def test_other_datasets():
    """測試其他資料集的格式"""
    
    try:
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir / "market_data_collector"))
        
        from market_data_collector.utils.config import API_ENDPOINT, TOKEN
    except ImportError as e:
        print(f"❌ 無法匯入設定: {e}")
        return
    
    datasets = {
        "融資融券": "TaiwanStockMarginPurchaseShortSale",
        "法人進出": "TaiwanStockInstitutionalInvestorsBuySell",
        "月營收": "TaiwanStockMonthRevenue",
        "本益比": "TaiwanStockPER"
    }
    
    for name, dataset in datasets.items():
        print(f"\n🧪 測試 {name} ({dataset})...")
        
        params = {
            "dataset": dataset,
            "data_id": "2330",
            "start_date": "2024-12-01",
            "end_date": "2024-12-05",
            "token": TOKEN
        }
        
        try:
            response = requests.get(API_ENDPOINT, params=params, timeout=30)
            
            if response.status_code == 200:
                json_data = response.json()
                
                if json_data.get("status") == 200 and "data" in json_data:
                    data = json_data["data"]
                    
                    if data:
                        df = pd.DataFrame(data)
                        print(f"✅ {name}: {len(df)} 筆資料")
                        print(f"📋 欄位: {list(df.columns)}")
                    else:
                        print(f"⚠️  {name}: 無資料")
                else:
                    print(f"❌ {name}: API 錯誤 - {json_data}")
            else:
                print(f"❌ {name}: HTTP 錯誤 {response.status_code}")
                
        except Exception as e:
            print(f"❌ {name} 測試失敗: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 FinMind API 回傳資料格式測試")
    print("=" * 60)
    
    # 測試主要價格資料
    df = test_finmind_api_response()
    
    # 測試其他資料集
    test_other_datasets()
    
    print("\n" + "=" * 60)
    print("測試完成")
    print("=" * 60)