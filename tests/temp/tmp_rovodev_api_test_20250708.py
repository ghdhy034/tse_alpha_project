#!/usr/bin/env python3
"""
API 連接測試腳本 - 測試 FinMind 和 Shioaji API 連接
"""
import sys
import os
from pathlib import Path
import requests
import time
from datetime import datetime, date

# 添加路徑
sys.path.append(str(Path(__file__).parent / "market_data_collector"))

def test_finmind_api():
    """測試 FinMind API 連接"""
    print("=== 測試 FinMind API ===")
    
    try:
        from market_data_collector.utils.config import TOKEN, API_ENDPOINT
        
        # 測試基本連接
        params = {
            "dataset": "TaiwanStockPrice",
            "data_id": "2330",
            "start_date": "2024-12-01",
            "end_date": "2024-12-01",
            "token": TOKEN
        }
        
        print(f"API Endpoint: {API_ENDPOINT}")
        print(f"Token (前10字元): {TOKEN[:10]}...")
        print("發送測試請求...")
        
        response = requests.get(API_ENDPOINT, params=params, timeout=10)
        
        print(f"HTTP 狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            json_data = response.json()
            print(f"API 回應狀態: {json_data.get('status')}")
            print(f"資料筆數: {len(json_data.get('data', []))}")
            
            if json_data.get('status') == 200:
                print("✅ FinMind API 連接成功！")
                return True
            else:
                print(f"❌ FinMind API 錯誤: {json_data}")
                return False
        else:
            print(f"❌ HTTP 錯誤: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ FinMind API 測試失敗: {e}")
        return False


def test_finmind_datasets():
    """測試 FinMind 各種資料集"""
    print("\n=== 測試 FinMind 資料集 ===")
    
    try:
        from market_data_collector.utils.config import TOKEN, API_ENDPOINT
        
        datasets_to_test = [
            ("TaiwanStockPrice", "日線資料"),
            ("TaiwanStockMinuteData", "分鐘線資料"),
            ("TaiwanStockMarginPurchaseShortSale", "融資融券"),
            ("TaiwanStockInstitutionalInvestorsBuySell", "法人進出")
        ]
        
        results = {}
        
        for dataset, description in datasets_to_test:
            print(f"\n測試 {description} ({dataset})...")
            
            params = {
                "dataset": dataset,
                "data_id": "2330",
                "start_date": "2024-12-01",
                "end_date": "2024-12-01",
                "token": TOKEN
            }
            
            try:
                response = requests.get(API_ENDPOINT, params=params, timeout=10)
                
                if response.status_code == 200:
                    json_data = response.json()
                    if json_data.get('status') == 200:
                        data_count = len(json_data.get('data', []))
                        print(f"✅ {description}: {data_count} 筆資料")
                        results[dataset] = True
                    else:
                        print(f"❌ {description}: API 錯誤 {json_data}")
                        results[dataset] = False
                else:
                    print(f"❌ {description}: HTTP {response.status_code}")
                    results[dataset] = False
                    
                # 避免 API 限流
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ {description}: {e}")
                results[dataset] = False
        
        success_count = sum(results.values())
        total_count = len(results)
        print(f"\n📊 FinMind 資料集測試結果: {success_count}/{total_count} 成功")
        
        return results
        
    except Exception as e:
        print(f"❌ FinMind 資料集測試失敗: {e}")
        return {}


def test_shioaji_api():
    """測試 Shioaji API 連接"""
    print("\n=== 測試 Shioaji API ===")
    
    try:
        # 檢查 Shioaji 是否安裝
        try:
            import shioaji as sj
            print("✅ Shioaji 套件已安裝")
        except ImportError:
            print("❌ Shioaji 套件未安裝")
            print("請執行: pip install shioaji")
            return False
        
        from market_data_collector.utils.config import (
            SHIOAJI_USER, SHIOAJI_PASS, SHIOAJI_CA_PATH, SHIOAJI_CA_PASS
        )
        
        print(f"API Key (前10字元): {SHIOAJI_USER[:10]}...")
        print(f"憑證路徑: {SHIOAJI_CA_PATH}")
        
        # 檢查憑證檔案是否存在
        if not os.path.exists(SHIOAJI_CA_PATH):
            print(f"❌ 憑證檔案不存在: {SHIOAJI_CA_PATH}")
            return False
        else:
            print("✅ 憑證檔案存在")
        
        # 嘗試登入
        print("嘗試登入 Shioaji...")
        api = sj.Shioaji()
        
        try:
            accounts = api.login(
                api_key=SHIOAJI_USER,
                secret_key=SHIOAJI_PASS,
                contracts_cb=lambda security_type: None
            )
            
            print("✅ Shioaji 登入成功！")
            print(f"帳戶數量: {len(accounts) if accounts else 0}")
            
            # 測試合約查詢
            try:
                contract = api.Contracts.Stocks['2330']
                print(f"✅ 合約查詢成功: {contract.code} - {contract.name}")
            except Exception as e:
                print(f"⚠️  合約查詢失敗: {e}")
            
            # 登出
            api.logout()
            print("✅ Shioaji 登出成功")
            
            return True
            
        except Exception as e:
            print(f"❌ Shioaji 登入失敗: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Shioaji API 測試失敗: {e}")
        return False


def test_database_connection():
    """測試資料庫連接"""
    print("\n=== 測試資料庫連接 ===")
    
    try:
        from market_data_collector.utils.db import get_conn, query_df
        
        # 測試連接
        conn = get_conn()
        print("✅ 資料庫連接成功")
        
        # 測試查詢
        tables = query_df("SELECT name FROM sqlite_master WHERE type='table'")
        print(f"✅ 資料表數量: {len(tables)}")
        print(f"資料表列表: {list(tables['name']) if not tables.empty else '無'}")
        
        # 檢查關鍵資料表
        key_tables = [
            'candlesticks_daily',
            'minute_bars', 
            'margin_purchase_shortsale',
            'institutional_investors_buy_sell'
        ]
        
        for table in key_tables:
            try:
                count = query_df(f"SELECT COUNT(*) as count FROM {table}")
                if not count.empty:
                    print(f"✅ {table}: {count.iloc[0]['count']} 筆資料")
                else:
                    print(f"⚠️  {table}: 資料表存在但無資料")
            except:
                print(f"❌ {table}: 資料表不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 資料庫連接測試失敗: {e}")
        return False


def main():
    """主測試函數"""
    print("🔧 TSE Alpha API 連接測試")
    print("=" * 50)
    
    results = {}
    
    # 測試資料庫
    results['database'] = test_database_connection()
    
    # 測試 FinMind API
    results['finmind_basic'] = test_finmind_api()
    
    if results['finmind_basic']:
        finmind_datasets = test_finmind_datasets()
        results['finmind_datasets'] = finmind_datasets
    
    # 測試 Shioaji API
    results['shioaji'] = test_shioaji_api()
    
    # 總結
    print("\n" + "=" * 50)
    print("📊 測試結果總結")
    print("=" * 50)
    
    print(f"資料庫連接: {'✅ 成功' if results['database'] else '❌ 失敗'}")
    print(f"FinMind API: {'✅ 成功' if results['finmind_basic'] else '❌ 失敗'}")
    print(f"Shioaji API: {'✅ 成功' if results['shioaji'] else '❌ 失敗'}")
    
    if 'finmind_datasets' in results:
        finmind_success = sum(results['finmind_datasets'].values())
        finmind_total = len(results['finmind_datasets'])
        print(f"FinMind 資料集: {finmind_success}/{finmind_total} 成功")
    
    # 建議
    print("\n💡 建議:")
    if not results['finmind_basic']:
        print("- 檢查 FinMind Token 是否有效")
        print("- 確認網路連接正常")
    
    if not results['shioaji']:
        print("- 檢查 Shioaji 憑證檔案路徑")
        print("- 確認 API Key 和密碼正確")
        print("- 安裝 Shioaji: pip install shioaji")
    
    if not results['database']:
        print("- 檢查資料庫檔案路徑")
        print("- 執行資料庫初始化")


if __name__ == "__main__":
    main()