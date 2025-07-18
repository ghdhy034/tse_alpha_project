#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生產級煙霧測試 - 階段3: 小規模多股票測試
測試5支股票的批次處理和記憶體管理
"""
import sys
import os
from pathlib import Path
import traceback
from datetime import datetime
import numpy as np
import torch
import psutil
import gc

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

def get_memory_usage():
    """獲取當前記憶體使用情況"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # 實際使用記憶體
        'vms_mb': memory_info.vms / 1024 / 1024,  # 虛擬記憶體
        'percent': process.memory_percent()        # 記憶體使用百分比
    }

def task_3_1_multi_stock_feature_processing():
    """任務3.1: 5支股票特徵處理"""
    print("\n" + "="*60)
    print("🎯 任務3.1: 5支股票特徵處理")
    print("="*60)
    
    try:
        from data_pipeline.features import FeatureEngine
        
        # 測試股票清單
        test_symbols = ['2330', '2317', '2603', '2454', '2412']
        
        print(f"📊 處理股票清單: {test_symbols}")
        
        # 記錄初始記憶體
        initial_memory = get_memory_usage()
        print(f"🧠 初始記憶體使用: {initial_memory['rss_mb']:.1f} MB ({initial_memory['percent']:.1f}%)")
        
        # 創建特徵引擎
        print("⚙️ 初始化特徵引擎...")
        feature_engine = FeatureEngine(symbols=test_symbols)
        
        # 批次處理多股票
        print("🔄 批次處理多股票特徵...")
        start_date = '2024-01-01'
        end_date = '2024-01-31'  # 小範圍測試
        
        results = feature_engine.process_multiple_symbols(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            normalize=True
        )
        
        # 驗證結果
        if not results:
            raise ValueError("沒有成功處理任何股票")
        
        print(f"📈 處理結果摘要:")
        total_features = 0
        total_records = 0
        
        for symbol, (features, labels, prices) in results.items():
            feature_count = features.shape[1] if not features.empty else 0
            record_count = features.shape[0] if not features.empty else 0
            
            print(f"   {symbol}: {feature_count}維特徵, {record_count}筆記錄")
            
            total_features += feature_count
            total_records += record_count
            
            # 檢查特徵完整性
            if not features.empty:
                null_count = features.isnull().sum().sum()
                inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
                
                if null_count > record_count * 0.1:  # 超過10%空值警告
                    print(f"     ⚠️ 空值較多: {null_count}")
                if inf_count > 0:
                    print(f"     ⚠️ 無限值: {inf_count}")
        
        # 記錄處理後記憶體
        after_memory = get_memory_usage()
        memory_increase = after_memory['rss_mb'] - initial_memory['rss_mb']
        
        print(f"🧠 處理後記憶體: {after_memory['rss_mb']:.1f} MB (+{memory_increase:.1f} MB)")
        
        # 驗證處理成功率
        success_rate = len(results) / len(test_symbols)
        if success_rate < 0.8:  # 至少80%成功率
            raise ValueError(f"處理成功率過低: {success_rate:.1%}")
        
        print_status("任務3.1", "SUCCESS", f"成功處理{len(results)}/{len(test_symbols)}支股票，總計{total_records}筆記錄")
        return True, results
        
    except Exception as e:
        print_status("任務3.1", "FAILED", str(e))
        traceback.print_exc()
        return False, None

def task_3_2_batch_data_loading_test(features_dict):
    """任務3.2: 批次資料載入測試"""
    print("\n" + "="*60)
    print("🎯 任務3.2: 批次資料載入測試")
    print("="*60)
    
    try:
        if not features_dict:
            raise ValueError("沒有可用的特徵資料")
        
        from models.data_loader import TSEDataLoader, DataConfig
        
        # 記錄初始記憶體
        initial_memory = get_memory_usage()
        print(f"🧠 初始記憶體: {initial_memory['rss_mb']:.1f} MB")
        
        # 創建資料配置
        print("📊 創建資料載入配置...")
        symbols = list(features_dict.keys())
        
        data_config = DataConfig(
            symbols=symbols,
            train_start_date='2024-01-01',
            train_end_date='2024-01-20',
            val_start_date='2024-01-21',
            val_end_date='2024-01-25',
            test_start_date='2024-01-26',
            test_end_date='2024-01-31',
            sequence_length=32,  # 較短序列用於測試
            batch_size=4,        # 小批次
            num_workers=0        # 避免多進程問題
        )
        
        print(f"   股票數量: {len(symbols)}")
        print(f"   序列長度: {data_config.sequence_length}")
        print(f"   批次大小: {data_config.batch_size}")
        
        # 創建資料載入器
        print("🔄 創建資料載入器...")
        data_loader = TSEDataLoader(data_config)
        
        # 手動設置特徵資料 (跳過重新處理)
        data_loader.features_dict = features_dict
        
        # 獲取資料載入器
        print("📦 創建批次載入器...")
        train_loader, val_loader, test_loader = data_loader.get_dataloaders()
        
        print(f"   訓練批次: {len(train_loader)}")
        print(f"   驗證批次: {len(val_loader)}")
        print(f"   測試批次: {len(test_loader)}")
        
        # 測試批次載入
        print("🧪 測試批次載入...")
        batch_count = 0
        total_samples = 0
        
        for loader_name, loader in [("訓練", train_loader), ("驗證", val_loader), ("測試", test_loader)]:
            if len(loader) == 0:
                print(f"   ⚠️ {loader_name}載入器為空")
                continue
            
            print(f"   測試{loader_name}載入器...")
            
            for i, batch in enumerate(loader):
                if i >= 2:  # 只測試前2個批次
                    break
                
                # 檢查批次格式
                observation = batch['observation']
                labels = batch['labels']
                metadata = batch['metadata']
                
                batch_size = observation['price_frame'].shape[0]
                total_samples += batch_size
                batch_count += 1
                
                print(f"     批次{i+1}: {batch_size}個樣本")
                print(f"       price_frame: {observation['price_frame'].shape}")
                print(f"       fundamental: {observation['fundamental'].shape}")
                print(f"       account: {observation['account'].shape}")
                print(f"       labels: {labels.shape}")
                
                # 檢查資料有效性
                for key, tensor in observation.items():
                    if torch.isnan(tensor).any():
                        print(f"       ⚠️ {key}包含NaN值")
                    if torch.isinf(tensor).any():
                        print(f"       ⚠️ {key}包含無限值")
        
        # 記錄載入後記憶體
        after_memory = get_memory_usage()
        memory_increase = after_memory['rss_mb'] - initial_memory['rss_mb']
        
        print(f"🧠 載入後記憶體: {after_memory['rss_mb']:.1f} MB (+{memory_increase:.1f} MB)")
        
        if batch_count == 0:
            raise ValueError("沒有成功載入任何批次")
        
        print_status("任務3.2", "SUCCESS", f"成功載入{batch_count}個批次，總計{total_samples}個樣本")
        return True, (train_loader, val_loader, test_loader)
        
    except Exception as e:
        print_status("任務3.2", "FAILED", str(e))
        traceback.print_exc()
        return False, None

def task_3_3_memory_monitoring():
    """任務3.3: 記憶體使用監控"""
    print("\n" + "="*60)
    print("🎯 任務3.3: 記憶體使用監控")
    print("="*60)
    
    try:
        from models.model_architecture import ModelConfig, TSEAlphaModel
        from models.config.training_config import TrainingConfig
        
        # 記錄基準記憶體
        baseline_memory = get_memory_usage()
        print(f"🧠 基準記憶體: {baseline_memory['rss_mb']:.1f} MB")
        
        # 載入配置
        training_config = TrainingConfig()
        
        # 測試模型記憶體使用
        print("🤖 測試模型記憶體使用...")
        model_config = ModelConfig(
            price_frame_shape=(5, 32, training_config.other_features),  # 5支股票
            fundamental_dim=training_config.fundamental_features,
            account_dim=training_config.account_features
        )
        
        model = TSEAlphaModel(model_config)
        
        model_memory = get_memory_usage()
        model_increase = model_memory['rss_mb'] - baseline_memory['rss_mb']
        print(f"   模型載入後: {model_memory['rss_mb']:.1f} MB (+{model_increase:.1f} MB)")
        
        # 測試批次處理記憶體
        print("📦 測試批次處理記憶體...")
        batch_sizes = [1, 2, 4, 8]
        memory_usage = {}
        
        for batch_size in batch_sizes:
            # 創建測試批次
            observation = {
                'price_frame': torch.randn(batch_size, 5, 32, training_config.other_features),
                'fundamental': torch.randn(batch_size, training_config.fundamental_features),
                'account': torch.randn(batch_size, training_config.account_features)
            }
            
            # 前向傳播
            with torch.no_grad():
                outputs = model(observation)
            
            # 記錄記憶體
            batch_memory = get_memory_usage()
            memory_usage[batch_size] = batch_memory['rss_mb']
            
            print(f"   批次大小{batch_size}: {batch_memory['rss_mb']:.1f} MB")
            
            # 清理
            del observation, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        # 測試記憶體洩漏
        print("🔍 測試記憶體洩漏...")
        initial_test_memory = get_memory_usage()
        
        # 執行多次前向傳播
        for i in range(10):
            observation = {
                'price_frame': torch.randn(2, 5, 32, training_config.other_features),
                'fundamental': torch.randn(2, training_config.fundamental_features),
                'account': torch.randn(2, training_config.account_features)
            }
            
            with torch.no_grad():
                outputs = model(observation)
            
            # 立即清理
            del observation, outputs
            
            if i % 3 == 0:  # 每3次強制垃圾回收
                gc.collect()
        
        final_test_memory = get_memory_usage()
        leak_amount = final_test_memory['rss_mb'] - initial_test_memory['rss_mb']
        
        print(f"   測試前: {initial_test_memory['rss_mb']:.1f} MB")
        print(f"   測試後: {final_test_memory['rss_mb']:.1f} MB")
        print(f"   記憶體變化: {leak_amount:+.1f} MB")
        
        # 檢查記憶體洩漏
        if leak_amount > 50:  # 超過50MB認為有洩漏
            print(f"   ⚠️ 可能存在記憶體洩漏: {leak_amount:.1f} MB")
        
        # 分析記憶體使用模式
        print("📊 記憶體使用分析:")
        max_memory = max(memory_usage.values())
        min_memory = min(memory_usage.values())
        memory_range = max_memory - min_memory
        
        print(f"   最大使用: {max_memory:.1f} MB")
        print(f"   最小使用: {min_memory:.1f} MB")
        print(f"   使用範圍: {memory_range:.1f} MB")
        
        # 檢查記憶體效率
        if memory_range > 200:  # 記憶體使用變化過大
            print(f"   ⚠️ 記憶體使用變化較大: {memory_range:.1f} MB")
        
        print_status("任務3.3", "SUCCESS", f"記憶體監控完成，洩漏檢測: {leak_amount:+.1f} MB")
        return True
        
    except Exception as e:
        print_status("任務3.3", "FAILED", str(e))
        traceback.print_exc()
        return False

def run_stage3_multi_stock_test():
    """執行階段3: 小規模多股票測試"""
    print("🚀 開始階段3: 小規模多股票測試")
    print("="*80)
    
    start_time = datetime.now()
    
    # 執行任務3.1
    success_3_1, features_dict = task_3_1_multi_stock_feature_processing()
    
    # 執行任務3.2
    success_3_2, data_loaders = task_3_2_batch_data_loading_test(features_dict) if success_3_1 else (False, None)
    
    # 執行任務3.3
    success_3_3 = task_3_3_memory_monitoring()
    
    # 總結
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results = {
        "任務3.1": success_3_1,
        "任務3.2": success_3_2,
        "任務3.3": success_3_3
    }
    
    print("\n" + "="*80)
    print("📋 階段3執行總結")
    print("="*80)
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for task_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"   {task_name}: {status}")
    
    print(f"\n📊 總體結果: {success_count}/{total_count} 任務成功")
    print(f"⏱️ 執行時間: {duration:.1f} 秒")
    
    # 最終記憶體檢查
    final_memory = get_memory_usage()
    print(f"🧠 最終記憶體使用: {final_memory['rss_mb']:.1f} MB ({final_memory['percent']:.1f}%)")
    
    if success_count == total_count:
        print("🎉 階段3: 小規模多股票測試 - 全部通過！")
        print("✅ 準備進入階段4: 訓練流程驗證")
        return True
    else:
        print("⚠️ 階段3: 小規模多股票測試 - 部分失敗")
        print("❌ 需要修復問題後再繼續")
        return False

if __name__ == "__main__":
    try:
        success = run_stage3_multi_stock_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 用戶中斷測試")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未預期的錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)