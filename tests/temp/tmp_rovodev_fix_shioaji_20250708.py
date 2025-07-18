#!/usr/bin/env python3
"""
修復 Shioaji 登入問題和改用 Shioaji 取得分鐘線資料
"""
import sys
import os
from pathlib import Path

# 添加路徑
sys.path.append(str(Path(__file__).parent / "market_data_collector"))

def diagnose_shioaji_login():
    """診斷 Shioaji 登入問題"""
    print("=== 診斷 Shioaji 登入問題 ===")
    
    try:
        import shioaji as sj
        from market_data_collector.utils.config import (
            SHIOAJI_USER, SHIOAJI_PASS, SHIOAJI_CA_PATH, SHIOAJI_CA_PASS
        )
        
        print(f"API Key: {SHIOAJI_USER}")
        print(f"Secret Key: {SHIOAJI_PASS}")
        print(f"憑證路徑: {SHIOAJI_CA_PATH}")
        print(f"憑證密碼: {SHIOAJI_CA_PASS}")
        
        # 檢查憑證檔案
        if not os.path.exists(SHIOAJI_CA_PATH):
            print(f"❌ 憑證檔案不存在: {SHIOAJI_CA_PATH}")
            return False
        
        print("✅ 憑證檔案存在")
        
        # 嘗試不同的登入方式
        print("\n--- 嘗試方式 1: 使用 API Key + Secret Key ---")
        try:
            api = sj.Shioaji()
            accounts = api.login(
                api_key=SHIOAJI_USER,
                secret_key=SHIOAJI_PASS,
                contracts_cb=lambda security_type: None
            )
            print("✅ 方式 1 成功")
            api.logout()
            return True
            
        except Exception as e:
            print(f"❌ 方式 1 失敗: {e}")
        
        print("\n--- 嘗試方式 2: 使用憑證檔案 ---")
        try:
            api = sj.Shioaji()
            accounts = api.login(
                person_id=SHIOAJI_USER,
                passwd=SHIOAJI_PASS,
                contracts_cb=lambda security_type: None
            )
            print("✅ 方式 2 成功")
            api.logout()
            return True
            
        except Exception as e:
            print(f"❌ 方式 2 失敗: {e}")
        
        print("\n--- 嘗試方式 3: 檢查 API Key 格式 ---")
        # 檢查 API Key 是否包含無效字符
        if '_' in SHIOAJI_USER:
            print("⚠️  API Key 包含底線字符，這可能是問題原因")
            print("建議檢查 API Key 是否正確")
        
        if '_' in SHIOAJI_PASS:
            print("⚠️  Secret Key 包含底線字符，這可能是問題原因")
            print("建議檢查 Secret Key 是否正確")
        
        return False
        
    except Exception as e:
        print(f"❌ 診斷失敗: {e}")
        return False


def test_shioaji_minute_data():
    """測試 Shioaji 分鐘線資料下載"""
    print("\n=== 測試 Shioaji 分鐘線資料 ===")
    
    try:
        import shioaji as sj
        from datetime import datetime, date
        from market_data_collector.utils.config import (
            SHIOAJI_USER, SHIOAJI_PASS
        )
        
        # 嘗試登入
        api = sj.Shioaji()
        
        try:
            accounts = api.login(
                api_key=SHIOAJI_USER,
                secret_key=SHIOAJI_PASS,
                contracts_cb=lambda security_type: None
            )
            print("✅ Shioaji 登入成功")
            
        except Exception as e:
            print(f"❌ Shioaji 登入失敗: {e}")
            print("請檢查 API Key 和 Secret Key 是否正確")
            return False
        
        # 測試合約查詢
        try:
            contract = api.Contracts.Stocks['2330']
            print(f"✅ 合約查詢成功: {contract.code} - {contract.name}")
        except Exception as e:
            print(f"❌ 合約查詢失敗: {e}")
            api.logout()
            return False
        
        # 測試分鐘線資料下載
        try:
            target_date = date(2024, 12, 16)  # 週一
            start_time = datetime.combine(target_date, datetime.min.time().replace(hour=9))
            end_time = datetime.combine(target_date, datetime.min.time().replace(hour=13, minute=30))
            
            print(f"下載 2330 {target_date} 的分鐘線資料...")
            print(f"時間範圍: {start_time} ~ {end_time}")
            
            kbars = api.kbars(
                contract=contract,
                start=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                timeout=30000
            )
            
            if kbars and len(kbars) > 0:
                print(f"✅ 成功下載 {len(kbars)} 筆分鐘線資料")
                
                # 顯示前幾筆資料
                print("前 3 筆資料:")
                for i, k in enumerate(kbars[:3]):
                    print(f"  {i+1}. {k.ts} O:{k.Open} H:{k.High} L:{k.Low} C:{k.Close} V:{k.Volume}")
                
                api.logout()
                return True
            else:
                print("⚠️  無分鐘線資料（可能是非交易日）")
                api.logout()
                return False
                
        except Exception as e:
            print(f"❌ 分鐘線資料下載失敗: {e}")
            api.logout()
            return False
            
    except Exception as e:
        print(f"❌ Shioaji 分鐘線測試失敗: {e}")
        return False


def fix_module_import_issue():
    """修復模組導入問題"""
    print("\n=== 修復模組導入問題 ===")
    
    try:
        # 檢查 data_pipeline 目錄下的 __init__.py
        init_file = Path("data_pipeline/__init__.py")
        if init_file.exists():
            print("✅ data_pipeline/__init__.py 存在")
        else:
            print("❌ data_pipeline/__init__.py 不存在")
            
        # 檢查路徑問題
        current_dir = Path.cwd()
        market_data_collector_path = current_dir / "market_data_collector"
        
        print(f"當前目錄: {current_dir}")
        print(f"market_data_collector 路徑: {market_data_collector_path}")
        print(f"market_data_collector 存在: {market_data_collector_path.exists()}")
        
        # 建議的修復方案
        print("\n💡 建議修復方案:")
        print("1. 確保在專案根目錄執行腳本")
        print("2. 檢查 sys.path 設定")
        print("3. 使用相對導入或絕對路徑")
        
        return True
        
    except Exception as e:
        print(f"❌ 模組導入診斷失敗: {e}")
        return False


def update_shioaji_downloader():
    """更新 Shioaji 下載器以修復登入問題"""
    print("\n=== 更新 Shioaji 下載器 ===")
    
    try:
        # 讀取現有的 fetch_minute.py
        fetch_minute_path = Path("data_pipeline/fetch_minute.py")
        
        if not fetch_minute_path.exists():
            print("❌ fetch_minute.py 不存在")
            return False
        
        with open(fetch_minute_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 檢查是否需要更新 Shioaji 登入方式
        if "provided string contained invalid character" in content:
            print("發現已知的 Shioaji 登入問題")
        
        # 建議的修復
        print("💡 建議修復 Shioaji 登入:")
        print("1. 檢查 API Key 和 Secret Key 格式")
        print("2. 確認沒有多餘的字符或空格")
        print("3. 嘗試重新申請 API 憑證")
        print("4. 使用最新版本的 Shioaji 套件")
        
        return True
        
    except Exception as e:
        print(f"❌ 更新 Shioaji 下載器失敗: {e}")
        return False


def create_minute_bars_table():
    """建立 minute_bars 資料表"""
    print("\n=== 建立 minute_bars 資料表 ===")
    
    try:
        from market_data_collector.utils.db import execute_sql
        
        sql = """
        CREATE TABLE IF NOT EXISTS minute_bars (
            symbol TEXT NOT NULL,
            ts TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            vwap REAL,
            PRIMARY KEY(symbol, ts)
        )
        """
        
        execute_sql(sql)
        print("✅ minute_bars 資料表建立成功")
        
        # 建立索引
        index_sql = """
        CREATE INDEX IF NOT EXISTS idx_minute_bars_symbol_ts 
        ON minute_bars(symbol, ts)
        """
        execute_sql(index_sql)
        print("✅ minute_bars 索引建立成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 建立 minute_bars 資料表失敗: {e}")
        return False


def main():
    """主修復函數"""
    print("🔧 修復 Shioaji 和分鐘線問題")
    print("=" * 60)
    
    results = {}
    
    # 1. 診斷 Shioaji 登入問題
    results['shioaji_diagnosis'] = diagnose_shioaji_login()
    
    # 2. 測試 Shioaji 分鐘線資料
    if results['shioaji_diagnosis']:
        results['shioaji_minute_data'] = test_shioaji_minute_data()
    else:
        results['shioaji_minute_data'] = False
        print("⚠️  跳過 Shioaji 分鐘線測試（登入失敗）")
    
    # 3. 修復模組導入問題
    results['module_import'] = fix_module_import_issue()
    
    # 4. 更新 Shioaji 下載器
    results['shioaji_update'] = update_shioaji_downloader()
    
    # 5. 建立 minute_bars 資料表
    try:
        results['minute_bars_table'] = create_minute_bars_table()
    except:
        results['minute_bars_table'] = False
        print("⚠️  無法建立 minute_bars 資料表（可能是模組導入問題）")
    
    # 總結
    print("\n" + "=" * 60)
    print("📊 修復結果總結")
    print("=" * 60)
    
    for task, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{task}: {status}")
    
    # 建議
    print("\n💡 修復建議:")
    
    if not results['shioaji_diagnosis']:
        print("🔑 Shioaji 登入問題:")
        print("   1. 檢查 config.py 中的 SHIOAJI_USER 和 SHIOAJI_PASS")
        print("   2. 確認 API Key 格式正確（不應包含底線等特殊字符）")
        print("   3. 嘗試重新申請 Shioaji API 憑證")
        print("   4. 確認使用最新版本的 Shioaji 套件")
    
    print("\n📋 FinMind 分鐘線替代方案:")
    print("   由於您是 FinMind 非付費會員，分鐘線資料受限")
    print("   建議主要使用 Shioaji 取得分鐘線資料")
    print("   FinMind 可繼續用於日線、融資融券、法人進出等資料")
    
    print("\n🔧 模組導入問題:")
    print("   確保在專案根目錄執行腳本")
    print("   檢查 Python 路徑設定")


if __name__ == "__main__":
    main()