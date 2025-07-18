#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試修正後的 Shioaji 收集器
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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from tmp_rovodev_shioaji_collector_updated import ShioajiDataCollector
    
    print("🧪 測試修正後的 Shioaji 收集器...")
    print("=" * 50)
    
    # 1. 測試初始化
    print("1️⃣ 測試初始化...")
    collector = ShioajiDataCollector()
    print("✅ ShioajiDataCollector 初始化成功")
    
    # 2. 測試登入
    print("\n2️⃣ 測試 Shioaji 登入...")
    login_success = collector.login_shioaji()
    
    if login_success:
        print("✅ Shioaji 登入成功")
        
        # 3. 測試流量監控設置
        print("\n3️⃣ 測試流量監控設置...")
        collector.setup_flow_monitor()
        
        if collector.flow_monitor:
            print("✅ 流量監控器設置成功")
            
            # 顯示初始流量狀況
            print("\n4️⃣ 檢查初始流量狀況...")
            collector.flow_monitor.show_status()
        else:
            print("⚠️ 流量監控器設置失敗")
        
        # 4. 測試合約獲取
        print("\n5️⃣ 測試合約獲取...")
        test_symbol = "2330"  # 台積電
        
        try:
            # 檢查 API 是否有 Contracts 屬性
            if hasattr(collector.api, 'Contracts'):
                print("✅ API 已有 Contracts 屬性")
                if hasattr(collector.api.Contracts, 'Stocks'):
                    print("✅ Contracts.Stocks 屬性存在")
                    try:
                        contract = collector.api.Contracts.Stocks[test_symbol]
                        print(f"✅ 成功獲取 {test_symbol} 合約: {contract}")
                    except KeyError:
                        print(f"⚠️ {test_symbol} 合約不存在，需要手動獲取")
                else:
                    print("❌ Contracts.Stocks 屬性不存在")
            else:
                print("❌ API 沒有 Contracts 屬性，需要手動獲取合約")
                
                # 測試手動獲取合約
                print("🔄 測試手動獲取合約...")
                try:
                    contracts = collector.api.fetch_contracts(contract_download=True)
                    print("✅ 手動獲取合約成功")
                    
                    # 再次檢查
                    if hasattr(collector.api, 'Contracts') and hasattr(collector.api.Contracts, 'Stocks'):
                        contract = collector.api.Contracts.Stocks[test_symbol]
                        print(f"✅ 手動獲取後成功訪問 {test_symbol} 合約: {contract}")
                    else:
                        print("❌ 手動獲取後仍無法訪問 Contracts")
                        
                except Exception as e:
                    print(f"❌ 手動獲取合約失敗: {e}")
        
        except Exception as e:
            print(f"❌ 測試合約獲取失敗: {e}")
        
        # 5. 測試簡單的分鐘線資料獲取
        print(f"\n6️⃣ 測試簡單的分鐘線資料獲取 ({test_symbol})...")
        try:
            # 使用短時間範圍測試
            df_minute = collector.fetch_minute_data(
                symbol=test_symbol,
                start_date="2024-01-01",
                end_date="2024-01-01"
            )
            
            if not df_minute.empty:
                print(f"✅ 成功獲取 {len(df_minute)} 筆分鐘線資料")
                print(f"✅ 資料欄位: {list(df_minute.columns)}")
                print(f"✅ 第一筆資料: {df_minute.iloc[0].to_dict()}")
            else:
                print("⚠️ 獲取的分鐘線資料為空")
                
        except Exception as e:
            print(f"❌ 測試分鐘線資料獲取失敗: {e}")
            import traceback
            traceback.print_exc()
        
        # 6. 登出
        print("\n7️⃣ 測試登出...")
        try:
            collector.api.logout()
            print("✅ Shioaji 登出成功")
        except Exception as e:
            print(f"⚠️ 登出時發生錯誤: {e}")
    
    else:
        print("❌ Shioaji 登入失敗，無法進行後續測試")
    
    print("\n✅ 所有測試完成")
    
except Exception as e:
    print(f"❌ 測試失敗: {e}")
    import traceback
    traceback.print_exc()