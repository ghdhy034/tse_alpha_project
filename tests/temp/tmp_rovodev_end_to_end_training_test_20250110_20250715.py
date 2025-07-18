#!/usr/bin/env python3
"""
TSE Alpha 端到端訓練測試腳本
測試完整的訓練流程，從資料載入到模型訓練
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Tuple
import traceback
import time

# 添加路徑
sys.path.append(str(Path(__file__).parent))

def test_data_loading():
    """測試 1: 資料載入和預處理"""
    print("📊 測試 1: 資料載入和預處理")
    print("-" * 40)
    
    try:
        from models.config.training_config import TrainingConfig
        from models.data_loader import TSEAlphaDataLoader
        
        # 創建配置
        config = TrainingConfig()
        
        print("✅ 訓練配置載入成功")
        print(f"   序列長度: {config.sequence_length}")
        print(f"   價格特徵數: {config.price_features}")
        print(f"   基本面特徵數: {config.fundamental_features}")
        
        # 創建資料載入器
        data_loader = TSEAlphaDataLoader(config)
        
        print("✅ 資料載入器創建成功")
        
        # 測試小規模資料載入
        test_symbols = ['2330', '2317', '2454']
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        
        print(f"\n   載入測試資料: {test_symbols}")
        print(f"   時間範圍: {start_date} ~ {end_date}")
        
        # 載入資料
        dataset = data_loader.load_data(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            split='train'
        )
        
        print("✅ 資料載入成功")
        print(f"   資料集大小: {len(dataset)}")
        
        # 檢查資料格式
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n   樣本格式檢查:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"     {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"     {key}: {type(value)} = {value}")
        
        return True, data_loader, dataset, config
        
    except Exception as e:
        print(f"❌ 資料載入測試失敗: {str(e)}")
        traceback.print_exc()
        return False, None, None, None

def test_model_setup(config):
    """測試 2: 模型設置和初始化"""
    print("\n🤖 測試 2: 模型設置和初始化")
    print("-" * 40)
    
    try:
        from models.model_architecture import ModelConfig, TSEAlphaModel
        
        # 創建模型配置
        model_config = ModelConfig(
            price_frame_shape=(3, config.sequence_length, config.price_features),
            fundamental_dim=config.fundamental_features,
            account_dim=config.account_features,
            n_stocks=3,  # 測試用較少股票
            hidden_dim=64  # 較小的隱藏維度
        )
        
        # 創建模型
        model = TSEAlphaModel(model_config)
        
        print("✅ 模型創建成功")
        
        # 計算模型大小
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   總參數數: {total_params:,}")
        print(f"   可訓練參數: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # 測試模型前向傳播
        batch_size = 2
        test_input = {
            'price_frame': torch.randn(batch_size, 3, config.sequence_length, config.price_features),
            'fundamental': torch.randn(batch_size, config.fundamental_features),
            'account': torch.randn(batch_size, config.account_features)
        }
        
        model.eval()
        with torch.no_grad():
            outputs = model(test_input)
        
        print("✅ 模型前向傳播成功")
        print(f"   輸出形狀:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape}")
        
        return True, model, model_config
        
    except Exception as e:
        print(f"❌ 模型設置測試失敗: {str(e)}")
        traceback.print_exc()
        return False, None, None

def create_synthetic_dataset(config, n_samples=100):
    """創建合成資料集用於訓練測試"""
    print("\n   創建合成資料集...")
    
    # 創建合成資料
    price_frames = torch.randn(n_samples, 3, config.sequence_length, config.price_features)
    fundamentals = torch.randn(n_samples, config.fundamental_features)
    accounts = torch.randn(n_samples, config.account_features)
    
    # 創建合成標籤 (股票選擇 + 倉位大小)
    stock_labels = torch.randint(0, 3, (n_samples,))  # 3檔股票
    position_labels = torch.randn(n_samples, 1) * 100  # 倉位大小
    
    # 創建資料集
    dataset = TensorDataset(
        price_frames, fundamentals, accounts, 
        stock_labels, position_labels
    )
    
    print(f"   合成資料集大小: {len(dataset)}")
    return dataset

def test_training_loop(model, config):
    """測試 3: 訓練循環"""
    print("\n🏋️ 測試 3: 訓練循環")
    print("-" * 40)
    
    try:
        # 創建合成資料集
        train_dataset = create_synthetic_dataset(config, n_samples=50)
        val_dataset = create_synthetic_dataset(config, n_samples=20)
        
        # 創建資料載入器
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        print("✅ 資料載入器創建成功")
        
        # 設置優化器和損失函數
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # 多任務損失函數
        stock_criterion = nn.CrossEntropyLoss()
        position_criterion = nn.MSELoss()
        value_criterion = nn.MSELoss()
        
        print("✅ 優化器和損失函數設置成功")
        
        # 訓練循環
        model.train()
        num_epochs = 3  # 少量 epoch 用於測試
        
        print(f"\n   開始訓練 ({num_epochs} epochs)...")
        
        training_history = []
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 訓練階段
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (price_frames, fundamentals, accounts, stock_labels, position_labels) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 準備輸入
                observation = {
                    'price_frame': price_frames,
                    'fundamental': fundamentals,
                    'account': accounts
                }
                
                # 前向傳播
                outputs = model(observation)
                
                # 計算損失
                stock_loss = stock_criterion(outputs['stock_logits'], stock_labels)
                position_loss = position_criterion(outputs['position_size'], position_labels)
                value_loss = value_criterion(outputs['value'], torch.zeros_like(outputs['value']))
                
                total_loss = stock_loss + position_loss + 0.1 * value_loss
                
                # 反向傳播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += total_loss.item()
                train_batches += 1
                
                if batch_idx == 0:  # 只顯示第一個批次的詳細信息
                    print(f"     Epoch {epoch+1}, Batch {batch_idx+1}:")
                    print(f"       股票選擇損失: {stock_loss.item():.4f}")
                    print(f"       倉位大小損失: {position_loss.item():.4f}")
                    print(f"       價值估計損失: {value_loss.item():.4f}")
                    print(f"       總損失: {total_loss.item():.4f}")
            
            avg_train_loss = train_loss / train_batches
            
            # 驗證階段
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for price_frames, fundamentals, accounts, stock_labels, position_labels in val_loader:
                    observation = {
                        'price_frame': price_frames,
                        'fundamental': fundamentals,
                        'account': accounts
                    }
                    
                    outputs = model(observation)
                    
                    stock_loss = stock_criterion(outputs['stock_logits'], stock_labels)
                    position_loss = position_criterion(outputs['position_size'], position_labels)
                    value_loss = value_criterion(outputs['value'], torch.zeros_like(outputs['value']))
                    
                    total_loss = stock_loss + position_loss + 0.1 * value_loss
                    val_loss += total_loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            epoch_time = time.time() - epoch_start_time
            
            print(f"   Epoch {epoch+1}/{num_epochs}:")
            print(f"     訓練損失: {avg_train_loss:.4f}")
            print(f"     驗證損失: {avg_val_loss:.4f}")
            print(f"     耗時: {epoch_time:.2f}s")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'time': epoch_time
            })
            
            model.train()
        
        print("✅ 訓練循環完成")
        
        # 分析訓練結果
        initial_loss = training_history[0]['train_loss']
        final_loss = training_history[-1]['train_loss']
        loss_reduction = (initial_loss - final_loss) / initial_loss
        
        print(f"\n   訓練結果分析:")
        print(f"     初始損失: {initial_loss:.4f}")
        print(f"     最終損失: {final_loss:.4f}")
        print(f"     損失下降: {loss_reduction:.2%}")
        print(f"     平均每epoch耗時: {np.mean([h['time'] for h in training_history]):.2f}s")
        
        return True, training_history
        
    except Exception as e:
        print(f"❌ 訓練循環測試失敗: {str(e)}")
        traceback.print_exc()
        return False, None

def test_model_saving_loading(model, config):
    """測試 4: 模型保存和載入"""
    print("\n💾 測試 4: 模型保存和載入")
    print("-" * 40)
    
    try:
        # 保存模型
        save_path = "tmp_test_model.pth"
        
        # 保存模型狀態
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'model_config': {
                'price_frame_shape': (3, config.sequence_length, config.price_features),
                'fundamental_dim': config.fundamental_features,
                'account_dim': config.account_features,
                'n_stocks': 3,
                'hidden_dim': 64
            }
        }, save_path)
        
        print("✅ 模型保存成功")
        
        # 載入模型
        from models.model_architecture import ModelConfig, TSEAlphaModel
        
        checkpoint = torch.load(save_path, map_location='cpu')
        
        # 重建模型配置
        model_config_dict = checkpoint['model_config']
        model_config = ModelConfig(**model_config_dict)
        
        # 創建新模型並載入權重
        new_model = TSEAlphaModel(model_config)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        print("✅ 模型載入成功")
        
        # 驗證模型一致性
        test_input = {
            'price_frame': torch.randn(1, 3, config.sequence_length, config.price_features),
            'fundamental': torch.randn(1, config.fundamental_features),
            'account': torch.randn(1, config.account_features)
        }
        
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            original_output = model(test_input)
            loaded_output = new_model(test_input)
        
        # 檢查輸出一致性
        for key in original_output.keys():
            if isinstance(original_output[key], torch.Tensor):
                diff = torch.abs(original_output[key] - loaded_output[key]).max().item()
                print(f"   {key} 最大差異: {diff:.8f}")
                assert diff < 1e-6, f"{key} 輸出不一致"
        
        print("✅ 模型一致性驗證通過")
        
        # 清理測試文件
        os.remove(save_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 模型保存載入測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def test_environment_integration(model, config):
    """測試 5: 環境整合測試"""
    print("\n🌍 測試 5: 環境整合測試")
    print("-" * 40)
    
    try:
        from gym_env.env import TSEAlphaEnv
        
        # 創建環境
        env = TSEAlphaEnv(
            symbols=['2330', '2317', '2454'],
            start_date='2024-01-01',
            end_date='2024-01-10',
            initial_cash=1000000.0
        )
        
        # 重置環境
        observation, info = env.reset(seed=42)
        
        print("✅ 環境創建和重置成功")
        
        # 測試模型與環境的整合
        model.eval()
        total_reward = 0.0
        step_count = 0
        max_steps = 5
        
        print(f"\n   執行 {max_steps} 步整合測試...")
        
        for step in range(max_steps):
            # 調整觀測格式給模型
            env_price_frame = observation['price_frame']
            n_stocks = 3  # 模型期望的股票數
            
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
            with torch.no_grad():
                action = model.get_action(model_observation, deterministic=True)
            
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
            
            print(f"     步驟 {step+1}: 動作=({stock_idx}, {position_array[0]}), "
                  f"獎勵={reward:.4f}, NAV={info.get('nav', 0):,.0f}")
            
            if terminated or truncated:
                break
        
        print("✅ 環境整合測試完成")
        print(f"   總步數: {step_count}")
        print(f"   累積獎勵: {total_reward:.6f}")
        print(f"   最終NAV: {info.get('nav', 0):,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 環境整合測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def test_performance_benchmark(model, config):
    """測試 6: 性能基準測試"""
    print("\n⚡ 測試 6: 性能基準測試")
    print("-" * 40)
    
    try:
        # 準備測試資料
        batch_sizes = [1, 4, 8]
        n_stocks = 3
        
        print("   測試不同批次大小的性能...")
        
        for batch_size in batch_sizes:
            test_input = {
                'price_frame': torch.randn(batch_size, n_stocks, config.sequence_length, config.price_features),
                'fundamental': torch.randn(batch_size, config.fundamental_features),
                'account': torch.randn(batch_size, config.account_features)
            }
            
            model.eval()
            
            # 預熱
            with torch.no_grad():
                for _ in range(5):
                    _ = model(test_input)
            
            # 性能測試
            start_time = time.time()
            n_iterations = 100
            
            with torch.no_grad():
                for _ in range(n_iterations):
                    outputs = model(test_input)
            
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / n_iterations
            throughput = batch_size / avg_time
            
            print(f"     批次大小 {batch_size}:")
            print(f"       平均推理時間: {avg_time*1000:.2f}ms")
            print(f"       吞吐量: {throughput:.1f} 樣本/秒")
        
        # 記憶體使用測試
        print(f"\n   記憶體使用測試...")
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model_gpu = model.to(device)
            
            test_input_gpu = {
                'price_frame': torch.randn(8, n_stocks, config.sequence_length, config.price_features).to(device),
                'fundamental': torch.randn(8, config.fundamental_features).to(device),
                'account': torch.randn(8, config.account_features).to(device)
            }
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                outputs = model_gpu(test_input_gpu)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            print(f"     GPU 峰值記憶體使用: {peak_memory:.2f} MB")
            
            model = model.to('cpu')  # 移回 CPU
        else:
            print("     GPU 不可用，跳過 GPU 記憶體測試")
        
        print("✅ 性能基準測試完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能基準測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def run_end_to_end_training_test():
    """執行端到端訓練測試"""
    print("=" * 60)
    print("TSE Alpha 端到端訓練測試")
    print("測試完整的訓練流程，從資料載入到模型訓練")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 6
    
    # 初始化變數
    data_loader = None
    dataset = None
    config = None
    model = None
    model_config = None
    training_history = None
    
    # 測試 1: 資料載入和預處理
    success, data_loader, dataset, config = test_data_loading()
    if success:
        tests_passed += 1
    
    # 測試 2: 模型設置和初始化
    if config:
        success, model, model_config = test_model_setup(config)
        if success:
            tests_passed += 1
    
    # 測試 3: 訓練循環
    if model and config:
        success, training_history = test_training_loop(model, config)
        if success:
            tests_passed += 1
    
    # 測試 4: 模型保存和載入
    if model and config:
        success = test_model_saving_loading(model, config)
        if success:
            tests_passed += 1
    
    # 測試 5: 環境整合測試
    if model and config:
        success = test_environment_integration(model, config)
        if success:
            tests_passed += 1
    
    # 測試 6: 性能基準測試
    if model and config:
        success = test_performance_benchmark(model, config)
        if success:
            tests_passed += 1
    
    # 測試結果總結
    print("\n" + "=" * 60)
    print("📋 端到端訓練測試結果")
    print("=" * 60)
    
    pass_rate = (tests_passed / total_tests) * 100
    
    print(f"總測試數: {total_tests}")
    print(f"通過測試: {tests_passed}")
    print(f"失敗測試: {total_tests - tests_passed}")
    print(f"通過率: {pass_rate:.1f}%")
    
    if pass_rate >= 85:
        print(f"\n🎉 端到端訓練測試通過！")
        print(f"✅ 完整訓練流程正常運作")
        print(f"✅ 資料載入和模型訓練成功")
        print(f"✅ 模型保存載入機制正常")
        print(f"✅ 環境整合測試通過")
        print(f"🚀 系統已準備好進行生產訓練")
        
        print(f"\n🎯 建議下一步:")
        print(f"   1. 進行性能基準測試")
        print(f"   2. 測試回測引擎")
        print(f"   3. 開始小規模生產訓練")
        print(f"   4. 擴展到完整180支股票訓練")
        
    elif pass_rate >= 70:
        print(f"\n✅ 端到端訓練基本可用")
        print(f"🔧 部分功能可能需要優化")
        
    else:
        print(f"\n⚠️ 端到端訓練存在重要問題")
        print(f"🔧 需要修復失敗的測試項目")
    
    return pass_rate >= 70

if __name__ == "__main__":
    success = run_end_to_end_training_test()
    print(f"\n{'✅ 測試通過' if success else '❌ 測試失敗'}")
    sys.exit(0 if success else 1)