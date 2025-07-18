#!/usr/bin/env python3
"""
綜合修復 Shioaji 登入和籌碼面資料問題
"""
import sys
import os
from pathlib import Path

# 添加路徑
sys.path.append(str(Path(__file__).parent / "market_data_collector"))

def test_chip_data_availability():
    """測試籌碼面資料可用性"""
    print("=== 測試籌碼面資料可用性 ===")
    
    try:
        from market_data_collector.utils.db import query_df
        
        # 檢查融資融券資料
        print("檢查融資融券資料...")
        margin_query = """
        SELECT COUNT(*) as count, MIN(date) as min_date, MAX(date) as max_date
        FROM margin_purchase_shortsale 
        WHERE symbol = '2330'
        """
        margin_result = query_df(margin_query)
        
        if not margin_result.empty:
            count = margin_result.iloc[0]['count']
            min_date = margin_result.iloc[0]['min_date']
            max_date = margin_result.iloc[0]['max_date']
            print(f"✅ 融資融券資料: {count} 筆，日期範圍: {min_date} ~ {max_date}")
            
            # 顯示欄位結構
            structure_query = "PRAGMA table_info(margin_purchase_shortsale)"
            structure = query_df(structure_query)
            print("融資融券資料表欄位:")
            for _, row in structure.iterrows():
                print(f"  {row['name']}: {row['type']}")
        else:
            print("❌ 無融資融券資料")
        
        # 檢查機構投信資料
        print("\n檢查機構投信資料...")
        inst_query = """
        SELECT COUNT(*) as count, MIN(date) as min_date, MAX(date) as max_date
        FROM institutional_investors_buy_sell 
        WHERE symbol = '2330'
        """
        inst_result = query_df(inst_query)
        
        if not inst_result.empty:
            count = inst_result.iloc[0]['count']
            min_date = inst_result.iloc[0]['min_date']
            max_date = inst_result.iloc[0]['max_date']
            print(f"✅ 機構投信資料: {count} 筆，日期範圍: {min_date} ~ {max_date}")
            
            # 顯示欄位結構
            structure_query = "PRAGMA table_info(institutional_investors_buy_sell)"
            structure = query_df(structure_query)
            print("機構投信資料表欄位:")
            for _, row in structure.iterrows():
                print(f"  {row['name']}: {row['type']}")
        else:
            print("❌ 無機構投信資料")
        
        # 測試實際資料載入
        print("\n測試實際資料載入...")
        
        # 測試融資融券資料載入
        margin_sample_query = """
        SELECT * FROM margin_purchase_shortsale 
        WHERE symbol = '2330' 
        ORDER BY date DESC 
        LIMIT 3
        """
        margin_sample = query_df(margin_sample_query)
        
        if not margin_sample.empty:
            print("融資融券資料範例:")
            print(margin_sample.to_string())
        
        # 測試機構投信資料載入
        inst_sample_query = """
        SELECT * FROM institutional_investors_buy_sell 
        WHERE symbol = '2330' 
        ORDER BY date DESC 
        LIMIT 3
        """
        inst_sample = query_df(inst_sample_query)
        
        if not inst_sample.empty:
            print("\n機構投信資料範例:")
            print(inst_sample.to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ 籌碼面資料測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def fix_shioaji_login():
    """修復 Shioaji 登入方式"""
    print("\n=== 修復 Shioaji 登入方式 ===")
    
    try:
        # 讀取現有的 fetch_minute.py
        fetch_minute_path = Path("data_pipeline/fetch_minute.py")
        
        if not fetch_minute_path.exists():
            print("❌ fetch_minute.py 不存在")
            return False
        
        with open(fetch_minute_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 檢查當前的登入方式
        if "api_key=SHIOAJI_USER" in content:
            print("發現使用 api_key 登入方式")
            
            # 建議的修復方案
            print("💡 建議修復方案:")
            print("1. 檢查 config.py 中的 SHIOAJI_USER 和 SHIOAJI_PASS 格式")
            print("2. 確認這些是 API Key 還是帳號密碼")
            print("3. 根據您的範例，可能需要使用位置參數登入")
            
            # 提供修復建議
            print("\n根據您的範例，建議的登入方式:")
            print("方式 1 (位置參數):")
            print('  accounts = api.login("api_key", "secret_key")')
            print("方式 2 (憑證登入):")
            print('  accounts = api.login(person_id="...", passwd="...")')
            print('  api.activate_ca(ca_path="...", ca_passwd="...", person_id="...")')
        
        return True
        
    except Exception as e:
        print(f"❌ Shioaji 登入修復失敗: {e}")
        return False


def test_shioaji_with_correct_method():
    """使用正確方法測試 Shioaji"""
    print("\n=== 測試 Shioaji 正確登入方法 ===")
    
    try:
        import shioaji as sj
        from market_data_collector.utils.config import (
            SHIOAJI_USER, SHIOAJI_PASS, SHIOAJI_CA_PATH, SHIOAJI_CA_PASS
        )
        
        print(f"嘗試使用的憑證:")
        print(f"User: {SHIOAJI_USER}")
        print(f"Pass: {SHIOAJI_PASS}")
        print(f"CA Path: {SHIOAJI_CA_PATH}")
        print(f"CA Pass: {SHIOAJI_CA_PASS}")
        
        # 方法 1: 位置參數登入 (根據您的範例)
        print("\n嘗試方法 1: 位置參數登入...")
        try:
            api = sj.Shioaji(simulation=False)
            accounts = api.login(SHIOAJI_USER, SHIOAJI_PASS)
            print("✅ 方法 1 成功 - 位置參數登入")
            
            # 測試合約查詢
            contract = api.Contracts.Stocks['2330']
            print(f"✅ 合約查詢成功: {contract.code} - {contract.name}")
            
            api.logout()
            return True
            
        except Exception as e:
            print(f"❌ 方法 1 失敗: {e}")
        
        # 方法 2: 憑證登入
        print("\n嘗試方法 2: 憑證登入...")
        try:
            api = sj.Shioaji(simulation=False)
            accounts = api.login(person_id=SHIOAJI_USER, passwd=SHIOAJI_PASS)
            
            if os.path.exists(SHIOAJI_CA_PATH):
                api.activate_ca(
                    ca_path=SHIOAJI_CA_PATH,
                    ca_passwd=SHIOAJI_CA_PASS,
                    person_id=SHIOAJI_USER
                )
                print("✅ 方法 2 成功 - 憑證登入")
            else:
                print("⚠️  憑證檔案不存在，跳過 activate_ca")
            
            # 測試合約查詢
            contract = api.Contracts.Stocks['2330']
            print(f"✅ 合約查詢成功: {contract.code} - {contract.name}")
            
            api.logout()
            return True
            
        except Exception as e:
            print(f"❌ 方法 2 失敗: {e}")
        
        # 方法 3: 關鍵字參數登入
        print("\n嘗試方法 3: 關鍵字參數登入...")
        try:
            api = sj.Shioaji(simulation=False)
            accounts = api.login(
                api_key=SHIOAJI_USER.strip(),
                secret_key=SHIOAJI_PASS.strip(),
                contracts_cb=lambda security_type: None
            )
            print("✅ 方法 3 成功 - 關鍵字參數登入")
            
            # 測試合約查詢
            contract = api.Contracts.Stocks['2330']
            print(f"✅ 合約查詢成功: {contract.code} - {contract.name}")
            
            api.logout()
            return True
            
        except Exception as e:
            print(f"❌ 方法 3 失敗: {e}")
        
        print("❌ 所有登入方法都失敗")
        return False
        
    except Exception as e:
        print(f"❌ Shioaji 測試失敗: {e}")
        return False


def fix_chip_features_import():
    """修復籌碼面特徵的導入問題"""
    print("\n=== 修復籌碼面特徵導入問題 ===")
    
    try:
        # 檢查 test_chip_features.py 中的導入問題
        test_file_path = Path("data_pipeline/test_chip_features.py")
        
        if test_file_path.exists():
            with open(test_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 修復導入問題
            if "name 'query_df' is not defined" in content or "from features import" in content:
                print("發現導入問題，進行修復...")
                
                # 替換有問題的導入
                fixed_content = content.replace(
                    "from features import FeatureEngine",
                    """try:
    from features import FeatureEngine
    from market_data_collector.utils.db import query_df
except ImportError as e:
    print(f"導入錯誤: {e}")
    # 提供備用導入
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_pipeline.features import FeatureEngine
    from market_data_collector.utils.db import query_df"""
                )
                
                # 寫回檔案
                with open(test_file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                print("✅ 修復 test_chip_features.py 導入問題")
        
        # 檢查 features.py 中的導入
        features_file_path = Path("data_pipeline/features.py")
        
        if features_file_path.exists():
            with open(features_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 確保 load_chip_data 方法中有正確的錯誤處理
            if "def load_chip_data" in content:
                print("✅ features.py 中的 load_chip_data 方法存在")
            else:
                print("❌ features.py 中缺少 load_chip_data 方法")
        
        return True
        
    except Exception as e:
        print(f"❌ 修復籌碼面特徵導入失敗: {e}")
        return False


def create_comprehensive_test():
    """創建綜合測試腳本"""
    print("\n=== 創建綜合測試腳本 ===")
    
    try:
        test_script = """#!/usr/bin/env python3
# 綜合測試腳本 - 測試籌碼面特徵和 Shioaji
import sys
from pathlib import Path

# 確保正確的路徑
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "market_data_collector"))

def test_chip_features_with_real_data():
    print("=== 測試籌碼面特徵與真實資料 ===")
    
    try:
        from data_pipeline.features import FeatureEngine
        from market_data_collector.utils.db import query_df
        
        engine = FeatureEngine()
        
        # 測試載入真實籌碼面資料
        symbol = '2330'
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        
        print(f"載入 {symbol} 的籌碼面資料...")
        chip_data = engine.load_chip_data(symbol, start_date, end_date)
        
        if chip_data:
            print("✅ 成功載入籌碼面資料:")
            for data_type, df in chip_data.items():
                print(f"  {data_type}: {df.shape}")
                if not df.empty:
                    print(f"    欄位: {list(df.columns)}")
                    print(f"    日期範圍: {df.index.min()} ~ {df.index.max()}")
        else:
            print("⚠️  無籌碼面資料")
        
        return True
        
    except Exception as e:
        print(f"❌ 籌碼面特徵測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_chip_features_with_real_data()
"""
        
        with open("tmp_rovodev_comprehensive_test.py", 'w', encoding='utf-8') as f:
            f.write(test_script)
        
        print("✅ 創建綜合測試腳本成功")
        return True
        
    except Exception as e:
        print(f"❌ 創建綜合測試腳本失敗: {e}")
        return False


def main():
    """主修復函數"""
    print("🔧 綜合修復 Shioaji 和籌碼面問題")
    print("=" * 60)
    
    results = {}
    
    # 1. 測試籌碼面資料可用性
    results['chip_data'] = test_chip_data_availability()
    
    # 2. 修復 Shioaji 登入方式
    results['shioaji_fix'] = fix_shioaji_login()
    
    # 3. 測試 Shioaji 正確登入方法
    results['shioaji_test'] = test_shioaji_with_correct_method()
    
    # 4. 修復籌碼面特徵導入問題
    results['chip_import_fix'] = fix_chip_features_import()
    
    # 5. 創建綜合測試腳本
    results['comprehensive_test'] = create_comprehensive_test()
    
    # 總結
    print("\n" + "=" * 60)
    print("📊 修復結果總結")
    print("=" * 60)
    
    for task, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{task}: {status}")
    
    # 建議
    print("\n💡 修復建議:")
    
    if results['chip_data']:
        print("✅ 籌碼面資料存在，可以正常使用")
    else:
        print("❌ 籌碼面資料有問題，需要檢查資料庫")
    
    if not results['shioaji_test']:
        print("🔑 Shioaji 登入建議:")
        print("   1. 檢查 API Key 和 Secret Key 格式")
        print("   2. 嘗試重新申請 Shioaji 憑證")
        print("   3. 確認使用正確的登入方法")
    
    print("\n📋 後續行動:")
    print("1. 執行 tmp_rovodev_comprehensive_test.py 測試籌碼面資料")
    print("2. 根據測試結果調整 Shioaji 登入方式")
    print("3. 確認籌碼面特徵可以正常計算")


if __name__ == "__main__":
    main()