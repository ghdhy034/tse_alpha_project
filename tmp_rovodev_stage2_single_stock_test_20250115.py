#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生產級煙霧測試 - 階段2: 單股票測試
測試單一股票(2330)的完整特徵工程和模型整合
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
sys.path.append(str(Path(__file__).parent / "gym_env"))

def print_status(task, status, details=""):
    """統一的狀態輸出格式"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_icon = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "🔄"
    print(f"[{timestamp}] {status_icon} {task}: {status}")
    if details:
        print(f"    詳情: {details}")

def task_2_1_single_stock_feature_engineering():
    """任務2.1: 單股票特徵工程測試 (2330)"""
    print("\n" + "="*60)
    print("🎯 任務2.1: 單股票特徵工程測試 (2330)")
    print("="*60)
    
    try:
        from data_pipeline.features import FeatureEngine
        
        # 創建特徵引擎
        print("⚙️ 初始化特徵引擎...")
        feature_engine = FeatureEngine(symbols=['2330'])
        
        # 測試單股票特徵處理
        print("📊 處理2330特徵工程...")
        start_date = '2024-01-01'
        end_date = '2024-01-31'  # 小範圍測試
        
        features, labels, prices = feature_engine.process_single_symbol(
            symbol='2330',
            start_date=start_date,
            end_date=end_date,
            normalize=True
        )
        
        # 驗證特徵維度
        if features.empty:
            raise ValueError("特徵資料為空")
        
        feature_count = features.shape[1]
        record_count = features.shape[0]
        
        print(f"📈 特徵工程結果:")
        print(f"   特徵維度: {feature_count}")
        print(f"   記錄數量: {record_count}")
        print(f"   日期範圍: {features.index.min()} ~ {features.index.max()}")
        
        # 檢查特徵完整性
        null_features = features.isnull().sum().sum()
        inf_features = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"   空值數量: {null_features}")
        print(f"   無限值數量: {inf_features}")
        
        # 驗證75維特徵配置
        expected_features = 75
        if feature_count < expected_features * 0.8:  # 允許80%的容忍度
            raise ValueError(f"特徵數量過少: {feature_count} < {expected_features * 0.8}")
        
        # 檢查標籤
        if not labels.empty:
            label_count = labels.shape[1]
            print(f"   標籤維度: {label_count}")
        
        # 檢查價格資料
        if not prices.empty:
            price_columns = list(prices.columns)
            print(f"   價格欄位: {price_columns}")
        
        print_status("任務2.1", "SUCCESS", f"2330特徵工程完成: {feature_count}維特徵, {record_count}筆記錄")
        return True, features, labels, prices
        
    except Exception as e:
        print_status("任務2.1", "FAILED", str(e))
        traceback.print_exc()
        return False, None, None, None

def task_2_2_model_forward_pass_test(features, labels, prices):
    """任務2.2: 模型前向傳播測試"""
    print("\n" + "="*60)
    print("🎯 任務2.2: 模型前向傳播測試")
    print("="*60)
    
    try:
        if features is None or features.empty:
            raise ValueError("沒有可用的特徵資料")
        
        from models.model_architecture import ModelConfig, TSEAlphaModel
        from models.config.training_config import TrainingConfig
        
        # 載入配置
        print("🔧 載入模型配置...")
        training_config = TrainingConfig()
        model_config = ModelConfig(
            price_frame_shape=(1, 64, training_config.other_features),  # 單股票測試
            fundamental_dim=training_config.fundamental_features,
            account_dim=training_config.account_features
        )
        
        print(f"   價格框架形狀: {model_config.price_frame_shape}")
        print(f"   基本面維度: {model_config.fundamental_dim}")
        print(f"   帳戶維度: {model_config.account_dim}")
        
        # 創建模型
        print("🤖 創建模型...")
        model = TSEAlphaModel(model_config)
        model.eval()
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   模型參數: {param_count:,}")
        
        # 準備測試資料
        print("📊 準備測試資料...")
        batch_size = 2
        seq_len = 64
        
        # 模擬觀測資料 (與Gym環境格式相容)
        observation = {
            'price_frame': torch.randn(batch_size, 1, seq_len, training_config.other_features),
            'fundamental': torch.randn(batch_size, training_config.fundamental_features),
            'account': torch.randn(batch_size, training_config.account_features)
        }
        
        print(f"   觀測形狀:")
        for key, value in observation.items():
            print(f"     {key}: {value.shape}")
        
        # 前向傳播測試
        print("🔄 執行前向傳播...")
        with torch.no_grad():
            outputs = model(observation, return_attention=True)
        
        print(f"   輸出形狀:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape}")
        
        # 測試動作生成
        print("🎮 測試動作生成...")
        action = model.get_action(observation, deterministic=True)
        stock_idx, position_array = action
        
        print(f"   生成動作: 股票={stock_idx}, 倉位={position_array}")
        
        # 測試動作評估
        print("📊 測試動作評估...")
        evaluation = model.evaluate_action(observation, action)
        
        print(f"   評估結果:")
        for key, value in evaluation.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:  # 只有一個元素才能轉換為標量
                    print(f"     {key}: {value.item():.6f}")
                else:
                    print(f"     {key}: {value.shape} - {value.mean().item():.6f} (平均值)")
        
        print_status("任務2.2", "SUCCESS", "模型前向傳播和動作生成正常")
        return True, model, observation
        
    except Exception as e:
        print_status("任務2.2", "FAILED", str(e))
        traceback.print_exc()
        return False, None, None

def task_2_3_env_model_integration_test(model, observation):
    """任務2.3: 環境-模型整合測試"""
    print("\n" + "="*60)
    print("🎯 任務2.3: 環境-模型整合測試")
    print("="*60)
    
    try:
        if model is None:
            raise ValueError("沒有可用的模型")
        
        from gym_env.env import TSEAlphaEnv
        from gym_env.env import EnvConfig
        
        # 創建環境配置
        print("🌍 創建交易環境...")
        env_config = EnvConfig(
            symbols=['2330'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_capital=1000000,
            max_position_days=15
        )
        
        # 創建環境
        env = TSEAlphaEnv(env_config)
        
        print(f"   環境配置:")
        print(f"     股票數量: {len(env_config.symbols)}")
        print(f"     初始資金: {env_config.initial_capital:,}")
        print(f"     最大持倉天數: {env_config.max_position_days}")
        
        # 重置環境
        print("🔄 重置環境...")
        obs, info = env.reset()
        
        print(f"   初始觀測:")
        for key, value in obs.items():
            if hasattr(value, 'shape'):
                print(f"     {key}: {value.shape}")
            else:
                print(f"     {key}: {type(value)}")
        
        # 測試環境-模型互動
        print("🤝 測試環境-模型互動...")
        steps = 5
        total_reward = 0
        
        for step in range(steps):
            # 將環境觀測轉換為模型輸入格式
            model_obs = {
                'price_frame': torch.tensor(obs['price_frame'], dtype=torch.float32).unsqueeze(0),
                'fundamental': torch.tensor(obs['fundamental'], dtype=torch.float32).unsqueeze(0),
                'account': torch.tensor(obs['account'], dtype=torch.float32).unsqueeze(0)
            }
            
            # 模型決策
            action = model.get_action(model_obs, deterministic=True)
            
            # 環境執行
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"     步驟{step+1}: 動作={action}, 獎勵={reward:.6f}")
            
            if terminated or truncated:
                print(f"     環境結束: terminated={terminated}, truncated={truncated}")
                break
        
        # 獲取環境狀態
        account_state = env.get_account_state()
        print(f"   最終帳戶狀態:")
        print(f"     NAV: {account_state['nav']:.2f}")
        print(f"     現金: {account_state['cash']:.2f}")
        print(f"     總獎勵: {total_reward:.6f}")
        
        print_status("任務2.3", "SUCCESS", f"環境-模型整合正常，執行{steps}步，總獎勵{total_reward:.6f}")
        return True
        
    except Exception as e:
        print_status("任務2.3", "FAILED", str(e))
        traceback.print_exc()
        return False

def run_stage2_single_stock_test():
    """執行階段2: 單股票測試"""
    print("🚀 開始階段2: 單股票測試")
    print("="*80)
    
    start_time = datetime.now()
    
    # 執行任務2.1
    success_2_1, features, labels, prices = task_2_1_single_stock_feature_engineering()
    
    # 執行任務2.2
    success_2_2, model, observation = task_2_2_model_forward_pass_test(features, labels, prices) if success_2_1 else (False, None, None)
    
    # 執行任務2.3
    success_2_3 = task_2_3_env_model_integration_test(model, observation) if success_2_2 else False
    
    # 總結
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results = {
        "任務2.1": success_2_1,
        "任務2.2": success_2_2, 
        "任務2.3": success_2_3
    }
    
    print("\n" + "="*80)
    print("📋 階段2執行總結")
    print("="*80)
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for task_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"   {task_name}: {status}")
    
    print(f"\n📊 總體結果: {success_count}/{total_count} 任務成功")
    print(f"⏱️ 執行時間: {duration:.1f} 秒")
    
    if success_count == total_count:
        print("🎉 階段2: 單股票測試 - 全部通過！")
        print("✅ 準備進入階段3: 小規模多股票測試")
        return True
    else:
        print("⚠️ 階段2: 單股票測試 - 部分失敗")
        print("❌ 需要修復問題後再繼續")
        return False

if __name__ == "__main__":
    try:
        success = run_stage2_single_stock_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 用戶中斷測試")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未預期的錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)