#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速修復測試 - 驗證特徵維度和Tensor錯誤修復
"""
import sys
import os
from pathlib import Path
import traceback
from datetime import datetime
import numpy as np
import torch

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

def test_feature_dimension_fix():
    """測試特徵維度修復"""
    print("\n" + "="*60)
    print("🔧 測試特徵維度修復")
    print("="*60)
    
    try:
        from data_pipeline.features import FeatureEngine
        
        print("⚙️ 測試2330特徵工程...")
        feature_engine = FeatureEngine(symbols=['2330'])
        
        features, labels, prices = feature_engine.process_single_symbol(
            symbol='2330',
            start_date='2024-01-01',
            end_date='2024-01-10',  # 更小範圍
            normalize=True
        )
        
        if features.empty:
            raise ValueError("特徵資料為空")
        
        feature_count = features.shape[1]
        print(f"📊 特徵維度結果: {feature_count}")
        
        # 檢查特徵維度 (應該是68維，不包含4個帳戶特徵)
        expected_without_account = 68
        if feature_count == expected_without_account:
            print_status("特徵維度修復", "SUCCESS", f"成功達到{feature_count}維特徵 (不含4個帳戶特徵)")
            print("💡 總計72維: 68維特徵工程 + 4維帳戶特徵(由環境提供)")
            return True, features
        elif feature_count == 72:
            print_status("特徵維度修復", "SUCCESS", f"達到72維特徵 (可能包含帳戶特徵)")
            return True, features
        else:
            print_status("特徵維度修復", "FAILED", f"特徵維度為{feature_count}，期望68維(+4帳戶)或72維")
            return False, features
            
    except Exception as e:
        print_status("特徵維度修復", "FAILED", str(e))
        traceback.print_exc()
        return False, None

def test_model_tensor_fix(features):
    """測試模型Tensor錯誤修復"""
    print("\n" + "="*60)
    print("🔧 測試模型Tensor錯誤修復")
    print("="*60)
    
    try:
        if features is None or features.empty:
            raise ValueError("沒有可用的特徵資料")
        
        from models.model_architecture import ModelConfig, TSEAlphaModel
        from models.config.training_config import TrainingConfig
        
        print("🤖 創建測試模型...")
        training_config = TrainingConfig()
        model_config = ModelConfig(
            price_frame_shape=(1, 32, training_config.other_features),
            fundamental_dim=training_config.fundamental_features,
            account_dim=training_config.account_features
        )
        
        model = TSEAlphaModel(model_config)
        model.eval()
        
        # 創建測試觀測
        print("📊 創建測試觀測...")
        observation = {
            'price_frame': torch.randn(2, 1, 32, training_config.other_features),
            'fundamental': torch.randn(2, training_config.fundamental_features),
            'account': torch.randn(2, training_config.account_features)
        }
        
        # 測試前向傳播
        print("🔄 測試前向傳播...")
        with torch.no_grad():
            outputs = model(observation)
        
        print(f"   輸出形狀檢查:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape}")
        
        # 測試動作生成
        print("🎮 測試動作生成...")
        action = model.get_action(observation, deterministic=True)
        print(f"   生成動作: 股票={action[0]}, 倉位={action[1]}")
        
        # 測試動作評估 (修復後的版本)
        print("📊 測試動作評估...")
        evaluation = model.evaluate_action(observation, action)
        
        print(f"   評估結果:")
        for key, value in evaluation.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:  # 只有一個元素才能轉換為標量
                    print(f"     {key}: {value.item():.6f}")
                else:
                    print(f"     {key}: {value.shape} - 平均值={value.mean().item():.6f}")
        
        print_status("模型Tensor修復", "SUCCESS", "模型前向傳播和動作評估正常")
        return True
        
    except Exception as e:
        print_status("模型Tensor修復", "FAILED", str(e))
        traceback.print_exc()
        return False

def test_training_config_alignment():
    """測試訓練配置對齊"""
    print("\n" + "="*60)
    print("🔧 測試訓練配置對齊")
    print("="*60)
    
    try:
        from models.config.training_config import TrainingConfig
        
        config = TrainingConfig()
        
        print(f"📊 訓練配置檢查:")
        print(f"   總特徵: {config.total_features}")
        print(f"   基本面特徵: {config.fundamental_features}")
        print(f"   其他特徵: {config.other_features}")
        print(f"   帳戶特徵: {config.account_features}")
        
        # 驗證配置一致性
        calculated_total = config.fundamental_features + config.other_features + config.account_features
        
        if config.total_features == 72 and calculated_total == 72:
            print_status("訓練配置對齊", "SUCCESS", "72維配置正確")
            return True
        else:
            print_status("訓練配置對齊", "FAILED", f"配置不一致: 聲明{config.total_features} vs 計算{calculated_total}")
            return False
            
    except Exception as e:
        print_status("訓練配置對齊", "FAILED", str(e))
        traceback.print_exc()
        return False

def run_quick_fix_test():
    """執行快速修復測試"""
    print("🚀 開始快速修復測試")
    print("="*80)
    
    start_time = datetime.now()
    
    # 測試1: 特徵維度修復
    success_1, features = test_feature_dimension_fix()
    
    # 測試2: 模型Tensor修復
    success_2 = test_model_tensor_fix(features) if success_1 else False
    
    # 測試3: 訓練配置對齊
    success_3 = test_training_config_alignment()
    
    # 總結
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results = {
        "特徵維度修復": success_1,
        "模型Tensor修復": success_2,
        "訓練配置對齊": success_3
    }
    
    print("\n" + "="*80)
    print("📋 快速修復測試總結")
    print("="*80)
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"   {test_name}: {status}")
    
    print(f"\n📊 總體結果: {success_count}/{total_count} 測試成功")
    print(f"⏱️ 執行時間: {duration:.1f} 秒")
    
    if success_count == total_count:
        print("🎉 快速修復測試 - 全部通過！")
        print("✅ 可以重新執行階段2測試")
        return True
    else:
        print("⚠️ 快速修復測試 - 部分失敗")
        print("❌ 需要進一步修復")
        return False

if __name__ == "__main__":
    try:
        success = run_quick_fix_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 用戶中斷測試")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未預期的錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)