#!/usr/bin/env python3
"""
修復 Shioaji 登入問題 - 已解決 ✅
根據 sinotrade_api_test.txt 中的工作範例修復登入方式

測試結果: 2024-12-19
狀態: 成功解決登入問題
問題: API Key 格式正確，使用位置參數登入成功
"""

import sys
import os
from pathlib import Path

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "market_data_collector"))

def test_shioaji_login():
    """測試 Shioaji 登入"""
    print("=== 測試 Shioaji 登入修復 ===")
    
    try:
        import shioaji as sj
        print("✅ Shioaji 模組導入成功")
    except ImportError:
        print("❌ Shioaji 未安裝")
        return False
    
    # 檢查配置檔案
    try:
        from utils.config import SHIOAJI_USER, SHIOAJI_PASS, SHIOAJI_CA_PATH, SHIOAJI_CA_PASS
        print("✅ 配置檔案讀取成功")
        print(f"API Key 前10字元: {SHIOAJI_USER[:10]}...")
        print(f"Secret Key 前10字元: {SHIOAJI_PASS[:10]}...")
        print(f"憑證路徑: {SHIOAJI_CA_PATH}")
    except Exception as e:
        print(f"❌ 配置檔案讀取失敗: {e}")
        return False
    
    # 測試登入
    try:
        print("\n--- 測試登入方式 ---")
        api = sj.Shioaji(simulation=False)
        
        # 方式 1: 根據用戶範例的登入方式
        print("嘗試方式 1: 位置參數登入...")
        try:
            accounts = api.login(SHIOAJI_USER, SHIOAJI_PASS)
            print("✅ 位置參數登入成功！")
            print(f"帳戶數量: {len(accounts) if accounts else 0}")
            
            # 測試憑證啟動
            if SHIOAJI_CA_PATH and os.path.exists(SHIOAJI_CA_PATH):
                print("嘗試啟動憑證...")
                api.activate_ca(
                    ca_path=SHIOAJI_CA_PATH,
                    ca_passwd=SHIOAJI_CA_PASS,
                    person_id="ghdhy034_test"  # 根據用戶範例
                )
                print("✅ 憑證啟動成功")
            
            # 測試合約查詢
            try:
                contract = api.Contracts.Stocks["2330"]
                print(f"✅ 合約查詢成功: {contract}")
            except Exception as e:
                print(f"⚠️  合約查詢失敗: {e}")
            
            # 登出
            api.logout()
            print("✅ 登出成功")
            return True
            
        except Exception as e1:
            print(f"❌ 位置參數登入失敗: {e1}")
            return False
    
    except Exception as e:
        print(f"❌ 登入測試失敗: {e}")
        return False

def analyze_api_key_format():
    """分析 API Key 格式"""
    print("\n=== 分析 API Key 格式 ===")
    
    try:
        from utils.config import SHIOAJI_USER, SHIOAJI_PASS
        
        print("API Key 分析:")
        print(f"  長度: {len(SHIOAJI_USER)}")
        print(f"  前10字元: {SHIOAJI_USER[:10]}")
        print(f"  後10字元: {SHIOAJI_USER[-10:]}")
        print(f"  包含字符: {set(SHIOAJI_USER)}")
        
        print("\nSecret Key 分析:")
        print(f"  長度: {len(SHIOAJI_PASS)}")
        print(f"  前10字元: {SHIOAJI_PASS[:10]}")
        print(f"  後10字元: {SHIOAJI_PASS[-10:]}")
        print(f"  包含字符: {set(SHIOAJI_PASS)}")
        
        # 與工作範例比較
        example_api = "3rUiddxES8vXhDVAgWxuBebrCc8D2JbuzgX2M5qw8dRq"
        example_secret = "ChHe8N94yweHhu5cfQ5wQAxxL3ymisNritVfZH7tkJVh"
        
        print(f"\n工作範例 API Key 長度: {len(example_api)}")
        print(f"工作範例 Secret Key 長度: {len(example_secret)}")
        print(f"當前 API Key 長度匹配: {len(SHIOAJI_USER) == len(example_api)}")
        print(f"當前 Secret Key 長度匹配: {len(SHIOAJI_PASS) == len(example_secret)}")
        
    except Exception as e:
        print(f"❌ API Key 分析失敗: {e}")

def main():
    print("🔧 Shioaji 登入問題診斷和修復")
    print("=" * 50)
    
    # 分析 API Key 格式
    analyze_api_key_format()
    
    # 測試登入
    success = test_shioaji_login()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Shioaji 登入修復成功！")
    else:
        print("💥 Shioaji 登入仍有問題，請檢查 API Key 配置")

if __name__ == "__main__":
    main()