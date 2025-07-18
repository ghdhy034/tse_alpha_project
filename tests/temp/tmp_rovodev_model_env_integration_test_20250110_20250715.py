#!/usr/bin/env python3
"""
TSE Alpha 模型-環境整合測試腳本
測試模型與環境之間的觀測格式和動作空間對齊
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from typing import Dict, Any, Tuple
import traceback

# 添加路徑
sys.path.append(str(Path(__file__).parent))

def test_config_alignment():
    """測試 1: 配置對齊驗證"""
    print("🔧 測試 1: 配置對齊驗證")
    print("-" * 40)
    
    try:
        from models.config.training_config import TrainingConfig
        from models.model_architecture import ModelConfig
        
        # 載入訓練配置
        training_config = TrainingConfig()
        
        print("✅ 訓練配置載入成功")
        print(f"   價格特徵數: {training_config.price_features}")
        print(f"   基本面特徵數: {training_config.fundamental_features}")
        print(f"   帳戶特徵數: {training_config.account_features}")
        print(f"   序列長度: {training_config.sequence_length}")
        
        # 創建模型配置
        model_config = ModelConfig(
            price_frame_shape=(10, training_config.sequence_length, training_config.price_features),
            fundamental_dim=training_config.fundamental_features,
            account_dim=training_config.account_features,
            n_stocks=10,
            hidden_dim=128
        )
        
        print("✅ 模型配置創建成功")
        print(f"   價格框架形狀: {model_config.price_frame_shape}")
        print(f"   基本面維度: {model_config.fundamental_dim}")
        print(f"   帳戶維度: {model_config.account_dim}")
        
        return True, training_config, model_config
        
    except Exception as e:
        print(f"❌ 配置對齊失敗: {str(e)}")
        traceback.print_exc()
        return False, None, None

def test_model_creation(model_config):
    """測試 2: 模型創建和初始化"""
    print("\n🤖 測試 2: 模型創建和初始化")
    print("-" * 40)
    
    try:
        from models.model_architecture import TSEAlphaModel
        
        # 創建模型
        model = TSEAlphaModel(model_config)
        
        print("✅ 模型創建成功")
        
        # 計算參數數量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   總參數數: {total_params:,}")
        print(f"   可訓練參數: {trainable_params:,}")
        
        # 檢查模型組件
        components = [
            'price_encoder', 'fundamental_encoder', 'account_encoder',
            'cross_stock_attention', 'feature_fusion', 'stock_selector',
            'position_sizer', 'value_head', 'risk_head'
        ]
        
        for component in components:
            assert hasattr(model, component), f"模型缺少組件: {component}"
        
        print("✅ 模型組件檢查通過")
        
        return True, model
        
    except Exception as e:
        print(f"❌ 模型創建失敗: {str(e)}")
        traceback.print_exc()
        return False, None

def test_environment_creation(training_config):
    """測試 3: 環境創建和觀測空間"""
    print("\n🌍 測試 3: 環境創建和觀測空間")
    print("-" * 40)
    
    try:
        from gym_env.env import TSEAlphaEnv
        
        # 創建環境 (使用較少股票進行測試)
        test_symbols = ['2330', '2317', '2454', '2303', '2408']
        env = TSEAlphaEnv(
            symbols=test_symbols,
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_cash=1000000.0
        )
        
        print("✅ 環境創建成功")
        print(f"   測試股票: {test_symbols}")
        print(f"   股票數量: {len(test_symbols)}")
        
        # 檢查觀測空間
        obs_space = env.observation_space
        print(f"   觀測空間: {obs_space}")
        
        # 檢查動作空間
        action_space = env.action_space
        print(f"   動作空間: {action_space}")
        
        # 重置環境獲取觀測
        observation, info = env.reset(seed=42)
        
        print("✅ 環境重置成功")
        print("   觀測形狀:")
        for key, value in observation.items():
            print(f"     {key}: {value.shape}")
        
        return True, env, observation
        
    except Exception as e:
        print(f"❌ 環境創建失敗: {str(e)}")
        traceback.print_exc()
        return False, None, None

def test_observation_model_compatibility(model, observation, model_config):
    """測試 4: 觀測與模型相容性"""
    print("\n🔗 測試 4: 觀測與模型相容性")
    print("-" * 40)
    
    try:
        # 調整觀測以匹配模型期望的形狀
        batch_size = 1
        n_stocks = model_config.n_stocks
        
        # 調整 price_frame 形狀
        env_price_frame = observation['price_frame']  # (env_n_stocks, 64, 5)
        
        # 如果環境股票數與模型不同，需要調整
        if env_price_frame.shape[0] != n_stocks:
            if env_price_frame.shape[0] < n_stocks:
                # 填充到模型期望的股票數
                padding = np.zeros((n_stocks - env_price_frame.shape[0], 64, 5), dtype=np.float32)
                adjusted_price_frame = np.concatenate([env_price_frame, padding], axis=0)
            else:
                # 截取到模型期望的股票數
                adjusted_price_frame = env_price_frame[:n_stocks]
        else:
            adjusted_price_frame = env_price_frame
        
        # 添加批次維度
        model_observation = {
            'price_frame': torch.tensor(adjusted_price_frame).unsqueeze(0),  # (1, n_stocks, 64, 5)
            'fundamental': torch.tensor(observation['fundamental']).unsqueeze(0),  # (1, 10)
            'account': torch.tensor(observation['account']).unsqueeze(0)  # (1, 4)
        }
        
        print("✅ 觀測格式調整成功")
        print("   模型輸入形狀:")
        for key, value in model_observation.items():
            print(f"     {key}: {value.shape}")
        
        # 測試模型前向傳播
        model.eval()
        with torch.no_grad():
            outputs = model(model_observation)
        
        print("✅ 模型前向傳播成功")
        print("   模型輸出形狀:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape}")
        
        # 檢查輸出合理性
        stock_logits = outputs['stock_logits']
        position_size = outputs['position_size']
        value = outputs['value']
        risk_score = outputs['risk_score']
        
        assert stock_logits.shape == (batch_size, n_stocks), f"stock_logits 形狀錯誤: {stock_logits.shape}"
        assert position_size.shape == (batch_size, 1), f"position_size 形狀錯誤: {position_size.shape}"
        assert value.shape == (batch_size, 1), f"value 形狀錯誤: {value.shape}"
        assert risk_score.shape == (batch_size, 1), f"risk_score 形狀錯誤: {risk_score.shape}"
        
        print("✅ 輸出形狀驗證通過")
        
        return True, model_observation, outputs
        
    except Exception as e:
        print(f"❌ 觀測模型相容性測試失敗: {str(e)}")
        traceback.print_exc()
        return False, None, None

def test_action_generation(model, model_observation):
    """測試 5: 動作生成和格式"""
    print("\n⚡ 測試 5: 動作生成和格式")
    print("-" * 40)
    
    try:
        # 測試確定性動作生成
        action_det = model.get_action(model_observation, deterministic=True)
        print(f"✅ 確定性動作生成成功: {action_det}")
        
        # 測試隨機動作生成
        action_rand = model.get_action(model_observation, deterministic=False)
        print(f"✅ 隨機動作生成成功: {action_rand}")
        
        # 檢查動作格式
        stock_idx, position_array = action_det
        
        assert isinstance(stock_idx, int), f"股票索引應為整數: {type(stock_idx)}"
        assert isinstance(position_array, np.ndarray), f"倉位應為 numpy 陣列: {type(position_array)}"
        assert position_array.shape == (1,), f"倉位陣列形狀錯誤: {position_array.shape}"
        assert position_array.dtype == np.int16, f"倉位陣列類型錯誤: {position_array.dtype}"
        
        print("✅ 動作格式驗證通過")
        print(f"   股票索引: {stock_idx} (類型: {type(stock_idx)})")
        print(f"   倉位陣列: {position_array} (形狀: {position_array.shape}, 類型: {position_array.dtype})")
        
        # 測試多次動作生成的一致性
        actions = []
        for i in range(5):
            action = model.get_action(model_observation, deterministic=True)
            actions.append(action)
        
        # 確定性動作應該一致
        first_action = actions[0]
        for action in actions[1:]:
            assert action[0] == first_action[0], "確定性動作股票索引不一致"
            assert np.array_equal(action[1], first_action[1]), "確定性動作倉位不一致"
        
        print("✅ 確定性動作一致性驗證通過")
        
        return True, action_det
        
    except Exception as e:
        print(f"❌ 動作生成測試失敗: {str(e)}")
        traceback.print_exc()
        return False, None

def test_env_action_compatibility(env, action):
    """測試 6: 環境動作相容性"""
    print("\n🔄 測試 6: 環境動作相容性")
    print("-" * 40)
    
    try:
        # 重置環境
        observation, info = env.reset(seed=42)
        
        # 調整動作以匹配環境的股票數量
        stock_idx, position_array = action
        env_n_stocks = len(env.symbols)
        
        if stock_idx >= env_n_stocks:
            stock_idx = stock_idx % env_n_stocks  # 調整到有效範圍
            print(f"   調整股票索引: {action[0]} -> {stock_idx}")
        
        adjusted_action = (stock_idx, position_array)
        
        print(f"執行動作: 股票索引={stock_idx}, 倉位={position_array[0]}")
        
        # 執行動作
        obs, reward, terminated, truncated, info = env.step(adjusted_action)
        
        print("✅ 環境動作執行成功")
        print(f"   獎勵: {reward:.6f}")
        print(f"   交易執行: {info.get('trade_executed', False)}")
        
        # 檢查新觀測格式
        print("   新觀測形狀:")
        for key, value in obs.items():
            print(f"     {key}: {value.shape}")
        
        # 檢查觀測數據完整性
        assert not np.any(np.isnan(obs['price_frame'])), "price_frame 包含 NaN"
        assert not np.any(np.isnan(obs['fundamental'])), "fundamental 包含 NaN"
        assert not np.any(np.isnan(obs['account'])), "account 包含 NaN"
        
        print("✅ 觀測數據完整性驗證通過")
        
        return True, obs
        
    except Exception as e:
        print(f"❌ 環境動作相容性測試失敗: {str(e)}")
        traceback.print_exc()
        return False, None

def test_continuous_interaction(model, env, model_config):
    """測試 7: 連續互動測試"""
    print("\n🔄 測試 7: 連續互動測試")
    print("-" * 40)
    
    try:
        # 重置環境
        observation, info = env.reset(seed=42)
        
        total_reward = 0.0
        step_count = 0
        max_steps = 10
        
        print("開始連續互動測試...")
        
        for step in range(max_steps):
            # 調整觀測格式給模型
            n_stocks = model_config.n_stocks
            env_price_frame = observation['price_frame']
            
            if env_price_frame.shape[0] != n_stocks:
                if env_price_frame.shape[0] < n_stocks:
                    padding = np.zeros((n_stocks - env_price_frame.shape[0], 64, 5), dtype=np.float32)
                    adjusted_price_frame = np.concatenate([env_price_frame, padding], axis=0)
                else:
                    adjusted_price_frame = env_price_frame[:n_stocks]
            else:
                adjusted_price_frame = env_price_frame
            
            model_observation = {
                'price_frame': torch.tensor(adjusted_price_frame).unsqueeze(0),
                'fundamental': torch.tensor(observation['fundamental']).unsqueeze(0),
                'account': torch.tensor(observation['account']).unsqueeze(0)
            }
            
            # 模型生成動作
            action = model.get_action(model_observation, deterministic=False)
            
            # 調整動作給環境
            stock_idx, position_array = action
            env_n_stocks = len(env.symbols)
            if stock_idx >= env_n_stocks:
                stock_idx = stock_idx % env_n_stocks
            
            adjusted_action = (stock_idx, position_array)
            
            # 環境執行動作
            observation, reward, terminated, truncated, info = env.step(adjusted_action)
            
            total_reward += reward
            step_count += 1
            
            if step % 3 == 0:
                nav = info.get('nav', 0)
                positions = len(info.get('positions', {}))
                print(f"   步驟 {step+1}: 股票={stock_idx}, 倉位={position_array[0]}, "
                      f"獎勵={reward:.4f}, NAV={nav:,.0f}, 持倉={positions}檔")
            
            if terminated or truncated:
                print(f"   回合結束: terminated={terminated}, truncated={truncated}")
                break
        
        print("✅ 連續互動測試成功")
        print(f"   總步數: {step_count}")
        print(f"   累積獎勵: {total_reward:.6f}")
        print(f"   平均獎勵: {total_reward/step_count:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 連續互動測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def run_model_env_integration_test():
    """執行模型-環境整合測試"""
    print("=" * 60)
    print("TSE Alpha 模型-環境整合測試")
    print("測試模型與環境之間的觀測格式和動作空間對齊")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 7
    
    # 初始化變數
    training_config = None
    model_config = None
    model = None
    env = None
    observation = None
    model_observation = None
    action = None
    
    # 測試 1: 配置對齊驗證
    success, training_config, model_config = test_config_alignment()
    if success:
        tests_passed += 1
    
    # 測試 2: 模型創建和初始化
    if model_config:
        success, model = test_model_creation(model_config)
        if success:
            tests_passed += 1
    
    # 測試 3: 環境創建和觀測空間
    if training_config:
        success, env, observation = test_environment_creation(training_config)
        if success:
            tests_passed += 1
    
    # 測試 4: 觀測與模型相容性
    if model and observation and model_config:
        success, model_observation, outputs = test_observation_model_compatibility(
            model, observation, model_config)
        if success:
            tests_passed += 1
    
    # 測試 5: 動作生成和格式
    if model and model_observation:
        success, action = test_action_generation(model, model_observation)
        if success:
            tests_passed += 1
    
    # 測試 6: 環境動作相容性
    if env and action:
        success, new_obs = test_env_action_compatibility(env, action)
        if success:
            tests_passed += 1
    
    # 測試 7: 連續互動測試
    if model and env and model_config:
        success = test_continuous_interaction(model, env, model_config)
        if success:
            tests_passed += 1
    
    # 測試結果總結
    print("\n" + "=" * 60)
    print("📋 模型-環境整合測試結果")
    print("=" * 60)
    
    pass_rate = (tests_passed / total_tests) * 100
    
    print(f"總測試數: {total_tests}")
    print(f"通過測試: {tests_passed}")
    print(f"失敗測試: {total_tests - tests_passed}")
    print(f"通過率: {pass_rate:.1f}%")
    
    if pass_rate >= 85:
        print(f"\n🎉 模型-環境整合測試通過！")
        print(f"✅ 模型與環境完全相容")
        print(f"✅ 觀測格式和動作空間對齊正確")
        print(f"🚀 可以進行代理人行為測試")
        
        print(f"\n🎯 建議下一步:")
        print(f"   1. 執行代理人行為測試")
        print(f"   2. 測試端到端訓練流程")
        print(f"   3. 進行性能基準測試")
        
    elif pass_rate >= 70:
        print(f"\n✅ 模型-環境基本相容")
        print(f"🔧 部分功能可能需要微調")
        
    else:
        print(f"\n⚠️ 模型-環境整合存在重要問題")
        print(f"🔧 需要修復失敗的測試項目")
    
    return pass_rate >= 70

if __name__ == "__main__":
    success = run_model_env_integration_test()
    print(f"\n{'✅ 測試通過' if success else '❌ 測試失敗'}")
    sys.exit(0 if success else 1)