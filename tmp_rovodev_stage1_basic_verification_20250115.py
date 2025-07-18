#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生產級煙霧測試 - 階段1: 基礎驗證
驗證75維特徵配置和核心組件初始化
"""
import sys
import os
from pathlib import Path
import traceback
from datetime import datetime

# 強制UTF-8輸出
sys.stdout.reconfigure(encoding='utf-8')

# 添加路徑
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent / "data_pipeline"))
sys.path.append(str(Path(__file__).parent / "market_data_collector"))

def print_status(task, status, details=""):
    """統一的狀態輸出格式"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_icon = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "🔄"
    print(f"[{timestamp}] {status_icon} {task}: {status}")
    if details:
        print(f"    詳情: {details}")

def task_1_1_verify_75d_config():
    """任務1.1: 驗證75維特徵配置載入"""
    print("\n" + "="*60)
    print("🎯 任務1.1: 驗證75維特徵配置載入")
    print("="*60)
    
    try:
        # 測試 TrainingConfig 載入
        from models.config.training_config import TrainingConfig
        config = TrainingConfig()
        
        # 驗證特徵維度
        expected_total = 75
        expected_fundamental = 18
        expected_other = 53
        expected_account = 4
        
        actual_total = config.total_features
        actual_fundamental = config.fundamental_features
        actual_other = config.other_features
        actual_account = config.account_features
        
        print(f"📊 特徵配置檢查:")
        print(f"   總特徵: {actual_total} (期望: {expected_total})")
        print(f"   基本面: {actual_fundamental} (期望: {expected_fundamental})")
        print(f"   其他: {actual_other} (期望: {expected_other})")
        print(f"   帳戶: {actual_account} (期望: {expected_account})")
        
        # 驗證配置正確性
        if actual_total != expected_total:
            raise ValueError(f"總特徵數不匹配: {actual_total} != {expected_total}")
        if actual_fundamental != expected_fundamental:
            raise ValueError(f"基本面特徵數不匹配: {actual_fundamental} != {expected_fundamental}")
        if actual_other != expected_other:
            raise ValueError(f"其他特徵數不匹配: {actual_other} != {expected_other}")
        if actual_account != expected_account:
            raise ValueError(f"帳戶特徵數不匹配: {actual_account} != {expected_account}")
        
        # 驗證特徵總和
        calculated_total = actual_fundamental + actual_other + actual_account
        if calculated_total != actual_total:
            raise ValueError(f"特徵總和不匹配: {calculated_total} != {actual_total}")
        
        print_status("任務1.1", "SUCCESS", "75維特徵配置正確載入")
        return True
        
    except Exception as e:
        print_status("任務1.1", "FAILED", str(e))
        traceback.print_exc()
        return False

def task_1_2_check_core_components():
    """任務1.2: 檢查核心組件初始化"""
    print("\n" + "="*60)
    print("🎯 任務1.2: 檢查核心組件初始化")
    print("="*60)
    
    components_status = {}
    
    try:
        # 1. 檢查 ModelConfig
        print("🔧 檢查 ModelConfig...")
        from models.model_architecture import ModelConfig
        model_config = ModelConfig()
        print(f"   價格框架形狀: {model_config.price_frame_shape}")
        print(f"   基本面維度: {model_config.fundamental_dim}")
        print(f"   帳戶維度: {model_config.account_dim}")
        components_status['ModelConfig'] = True
        
        # 2. 檢查 TSEAlphaModel
        print("🤖 檢查 TSEAlphaModel...")
        from models.model_architecture import TSEAlphaModel
        model = TSEAlphaModel(model_config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   模型參數數量: {param_count:,}")
        components_status['TSEAlphaModel'] = True
        
        # 3. 檢查 FeatureEngine
        print("⚙️ 檢查 FeatureEngine...")
        from data_pipeline.features import FeatureEngine
        feature_engine = FeatureEngine(symbols=['2330'])
        print(f"   特徵引擎初始化成功")
        print(f"   日內處理器: {'可用' if feature_engine.intraday_processor else '不可用'}")
        components_status['FeatureEngine'] = True
        
        # 4. 檢查 DataConfig
        print("📊 檢查 DataConfig...")
        from models.data_loader import DataConfig
        data_config = DataConfig(symbols=['2330', '2317'])
        print(f"   序列長度: {data_config.sequence_length}")
        print(f"   批次大小: {data_config.batch_size}")
        components_status['DataConfig'] = True
        
        # 5. 檢查 股票配置
        print("📈 檢查股票配置...")
        from stock_config import get_all_stocks, get_split_info, validate_splits
        all_stocks = get_all_stocks()
        split_info = get_split_info()
        is_valid, message = validate_splits()
        
        print(f"   總股票數: {len(all_stocks)}")
        print(f"   分割配置: {split_info}")
        print(f"   配置驗證: {message}")
        
        if not is_valid:
            raise ValueError(f"股票分割配置錯誤: {message}")
        components_status['StockConfig'] = True
        
        # 總結
        failed_components = [name for name, status in components_status.items() if not status]
        if failed_components:
            raise ValueError(f"組件初始化失敗: {failed_components}")
        
        print_status("任務1.2", "SUCCESS", f"所有{len(components_status)}個核心組件初始化成功")
        return True
        
    except Exception as e:
        print_status("任務1.2", "FAILED", str(e))
        traceback.print_exc()
        return False

def task_1_3_verify_database_connection():
    """任務1.3: 驗證資料庫連接和基本查詢"""
    print("\n" + "="*60)
    print("🎯 任務1.3: 驗證資料庫連接和基本查詢")
    print("="*60)
    
    try:
        # 檢查資料庫連接
        print("🗄️ 檢查資料庫連接...")
        from market_data_collector.utils.db import query_df, get_conn
        
        # 測試基本連接
        conn = get_conn()
        print(f"   資料庫連接: 成功")
        
        # 檢查主要資料表
        tables_to_check = [
            'candlesticks_daily',
            'candlesticks_min', 
            'technical_indicators',
            'margin_purchase_shortsale',
            'institutional_investors_buy_sell',
            'financials',
            'monthly_revenue'
        ]
        
        table_status = {}
        for table in tables_to_check:
            try:
                # 檢查表是否存在並獲取記錄數
                count_query = f"SELECT COUNT(*) as count FROM {table} LIMIT 1"
                result = query_df(count_query)
                if not result.empty:
                    # 獲取實際記錄數
                    count_query_full = f"SELECT COUNT(*) as count FROM {table}"
                    count_result = query_df(count_query_full)
                    record_count = count_result.iloc[0]['count'] if not count_result.empty else 0
                    table_status[table] = record_count
                    print(f"   ✅ {table}: {record_count:,} 筆記錄")
                else:
                    table_status[table] = 0
                    print(f"   ⚠️ {table}: 空表")
            except Exception as e:
                table_status[table] = None
                print(f"   ❌ {table}: 錯誤 - {str(e)}")
        
        # 測試特定股票資料查詢
        print("\n📊 測試股票資料查詢 (2330)...")
        test_queries = [
            ("日線資料", "SELECT COUNT(*) as count FROM candlesticks_daily WHERE symbol = '2330'"),
            ("分鐘線資料", "SELECT COUNT(*) as count FROM candlesticks_min WHERE symbol = '2330' LIMIT 1000"),
            ("技術指標", "SELECT COUNT(*) as count FROM technical_indicators WHERE symbol = '2330'")
        ]
        
        query_results = {}
        for name, query in test_queries:
            try:
                result = query_df(query)
                count = result.iloc[0]['count'] if not result.empty else 0
                query_results[name] = count
                print(f"   ✅ {name}: {count:,} 筆")
            except Exception as e:
                query_results[name] = None
                print(f"   ❌ {name}: 錯誤 - {str(e)}")
        
        # 驗證關鍵資料存在
        critical_tables = ['candlesticks_daily', 'technical_indicators']
        missing_critical = [table for table in critical_tables if table_status.get(table, 0) == 0]
        
        if missing_critical:
            raise ValueError(f"關鍵資料表缺少資料: {missing_critical}")
        
        # 驗證2330資料存在
        if query_results.get('日線資料', 0) == 0:
            raise ValueError("2330日線資料不存在")
        
        total_records = sum(count for count in table_status.values() if isinstance(count, int))
        print_status("任務1.3", "SUCCESS", f"資料庫連接正常，總計{total_records:,}筆記錄")
        return True
        
    except Exception as e:
        print_status("任務1.3", "FAILED", str(e))
        traceback.print_exc()
        return False

def run_stage1_verification():
    """執行階段1: 基礎驗證"""
    print("🚀 開始階段1: 基礎驗證")
    print("="*80)
    
    start_time = datetime.now()
    
    # 執行所有任務
    tasks = [
        ("任務1.1", task_1_1_verify_75d_config),
        ("任務1.2", task_1_2_check_core_components), 
        ("任務1.3", task_1_3_verify_database_connection)
    ]
    
    results = {}
    for task_name, task_func in tasks:
        print(f"\n🔄 執行 {task_name}...")
        results[task_name] = task_func()
    
    # 總結
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("📋 階段1執行總結")
    print("="*80)
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for task_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"   {task_name}: {status}")
    
    print(f"\n📊 總體結果: {success_count}/{total_count} 任務成功")
    print(f"⏱️ 執行時間: {duration:.1f} 秒")
    
    if success_count == total_count:
        print("🎉 階段1: 基礎驗證 - 全部通過！")
        print("✅ 系統準備就緒，可以進入階段2")
        return True
    else:
        print("⚠️ 階段1: 基礎驗證 - 部分失敗")
        print("❌ 需要修復問題後再繼續")
        return False

if __name__ == "__main__":
    try:
        success = run_stage1_verification()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 用戶中斷測試")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未預期的錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)