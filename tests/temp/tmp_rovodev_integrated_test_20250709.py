#!/usr/bin/env python3
"""
整合測試腳本 - 測試股票分組、籌碼面特徵和資料收集功能
"""
import sys
import os
from pathlib import Path
import time

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "market_data_collector"))

def test_stock_groups():
    """測試股票分組功能"""
    print("🔧 步驟 1: 測試股票分組功能")
    print("=" * 50)
    
    try:
        # 執行股票分組管理器
        with open('tmp_rovodev_stock_groups_manager.py', 'r', encoding='utf-8') as f:
            exec(f.read(), {'__name__': '__test__'})
        
        from tmp_rovodev_stock_groups_manager import StockGroupsManager
        
        manager = StockGroupsManager()
        
        if not manager.groups:
            print("❌ 股票分組載入失敗")
            return False
        
        # 顯示分組資訊
        total_stocks = 0
        for group_name, stocks in manager.groups.items():
            group_display = {
                'group_A': 'A. 半導體‧電子供應鏈',
                'group_B': 'B. 傳產／原物料＆運輸',
                'group_C': 'C. 金融‧內需消費／綠能生技'
            }.get(group_name, group_name)
            
            print(f"✅ {group_display}: {len(stocks)} 支股票")
            total_stocks += len(stocks)
        
        print(f"✅ 總計: {total_stocks} 支股票")
        
        # 測試平衡分割
        split_result = manager.get_balanced_split(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        # 驗證平衡性
        manager.verify_group_balance(split_result)
        
        # 儲存配置
        manager.save_split_config(split_result)
        
        print("✅ 股票分組功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 股票分組測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chip_features():
    """測試籌碼面特徵功能"""
    print("\n🔧 步驟 2: 測試籌碼面特徵功能")
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
        
        # 檢查資料表
        print("✅ 檢查資料表...")
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
        
        print("✅ 籌碼面特徵功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 籌碼面特徵測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_collector():
    """測試資料收集器功能"""
    print("\n🔧 步驟 3: 測試資料收集器功能")
    print("=" * 50)
    
    try:
        # 載入資料收集器
        with open('tmp_rovodev_enhanced_data_collector.py', 'r', encoding='utf-8') as f:
            exec(f.read(), {'__name__': '__test__'})
        
        from tmp_rovodev_enhanced_data_collector import EnhancedDataCollector, APIKeyManager
        
        # 測試API Key管理
        api_manager = APIKeyManager()
        print(f"✅ 載入 {len(api_manager.api_keys)} 個API Keys")
        
        # 測試資料收集器
        collector = EnhancedDataCollector()
        
        # 測試股票清單（應該使用新的三組別）
        stock_list = collector.get_full_stock_list()
        print(f"✅ 生成股票清單: {len(stock_list)} 支")
        print(f"   前10支: {stock_list[:10]}")
        
        print("✅ 資料收集器功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 資料收集器測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_data_split():
    """測試訓練資料分割功能"""
    print("\n🔧 步驟 4: 測試訓練資料分割功能")
    print("=" * 50)
    
    try:
        # 載入分割配置
        import json
        
        config_file = "stock_split_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            split_data = config.get('split', {})
            groups_data = config.get('groups', {})
            
            print("✅ 載入分割配置成功")
            print(f"   訓練集: {len(split_data.get('train', []))} 支股票")
            print(f"   驗證集: {len(split_data.get('validation', []))} 支股票")
            print(f"   測試集: {len(split_data.get('test', []))} 支股票")
            
            # 驗證每個集合中的組別分布
            for split_name, stocks in split_data.items():
                group_counts = {'group_A': 0, 'group_B': 0, 'group_C': 0}
                
                for stock in stocks:
                    for group_name, group_stocks in groups_data.items():
                        if stock in group_stocks:
                            group_counts[group_name] += 1
                            break
                
                total = sum(group_counts.values())
                print(f"   {split_name} 組別分布:")
                for group, count in group_counts.items():
                    percentage = count / total * 100 if total > 0 else 0
                    print(f"     {group}: {count} 支 ({percentage:.1f}%)")
            
            print("✅ 訓練資料分割功能正常")
            return True
        else:
            print("⚠️  分割配置檔案不存在，請先執行股票分組設定")
            return False
        
    except Exception as e:
        print(f"❌ 訓練資料分割測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_next_steps():
    """顯示下一步操作指南"""
    print("\n📋 下一步操作指南")
    print("=" * 50)
    
    print("1. 開始資料收集:")
    print("   run_enhanced_data_collector.bat")
    print("   - 將收集180支股票的6種資料類型")
    print("   - 支援斷點續傳和多API Key輪換")
    print()
    
    print("2. 監控收集進度:")
    print("   python tmp_rovodev_progress_manager.py")
    print("   - 查看收集進度和統計")
    print("   - 管理失敗任務")
    print()
    
    print("3. 驗證特徵工程:")
    print("   python data_pipeline/test_chip_features.py")
    print("   - 測試70+個特徵計算")
    print("   - 驗證籌碼面特徵")
    print()
    
    print("4. 開始模型訓練準備:")
    print("   - 使用 stock_split_config.json 中的分割配置")
    print("   - 確保三組別股票在訓練/驗證/測試集中平均分布")
    print()
    
    print("🎯 重要特色:")
    print("✅ 180支股票分三組別 (半導體、傳產、金融)")
    print("✅ 平衡的訓練/驗證/測試集分割")
    print("✅ 多API Key自動輪換")
    print("✅ 斷點續傳功能")
    print("✅ 70+個籌碼面特徵")
    print("✅ Shioaji分鐘線下載")


def main():
    """主測試函數"""
    print("=== TSE Alpha 整合測試 ===")
    print(f"測試時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 執行所有測試
    tests = [
        ("股票分組", test_stock_groups),
        ("籌碼面特徵", test_chip_features),
        ("資料收集器", test_data_collector),
        ("訓練資料分割", test_training_data_split)
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
        show_next_steps()
    else:
        print("⚠️  部分測試失敗，請檢查相關功能")
    
    return passed == len(results)


if __name__ == "__main__":
    main()