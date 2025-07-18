#!/usr/bin/env python3
"""
TSE Alpha 煙霧測試 - GTX 1660 Ti 專用
低配置快速驗證系統可用性
"""
import sys
import os
import time
import traceback
from pathlib import Path
import torch
import numpy as np

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))

# 強制使用煙霧測試配置
os.environ['TSE_ALPHA_MODE'] = 'smoke_test'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def check_gpu_availability():
    """檢查 GPU 可用性"""
    print("🔍 檢查 GPU 可用性...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_memory = gpu_props.total_memory / 1e9
    
    print(f"✅ GPU: {gpu_props.name}")
    print(f"✅ VRAM: {gpu_memory:.1f}GB")
    print(f"✅ 計算能力: {gpu_props.major}.{gpu_props.minor}")
    
    # 檢查是否為 GTX 1660 Ti
    if '1660' in gpu_props.name:
        print("✅ 檢測到 GTX 1660 Ti，使用低配置模式")
    else:
        print(f"⚠️  非 GTX 1660 Ti ({gpu_props.name})，仍使用低配置模式")
    
    return True

def test_basic_imports():
    """測試基本模組導入"""
    print("\n🔍 測試基本模組導入...")
    
    try:
        # 測試配置系統
        from configs.hardware_configs import ConfigManager, create_smoke_test_config
        config = create_smoke_test_config()
        print(f"✅ 硬體配置: batch_size={config['batch_size']}, seq_len={config['sequence_length']}")
        
        # 測試核心模組
        from models.config.training_config import TrainingConfig
        from models.model_architecture import TSEAlphaModel, ModelConfig
        from models.data_loader import TSEDataLoader, DataConfig
        from gym_env.env import TSEAlphaEnv
        
        print("✅ 核心模組導入成功")
        return True
        
    except Exception as e:
        print(f"❌ 模組導入失敗: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """測試模型創建 (低配置)"""
    print("\n🔍 測試模型創建...")
    
    try:
        from models.model_architecture import TSEAlphaModel, ModelConfig
        from configs.hardware_configs import create_smoke_test_config
        
        # 使用煙霧測試配置
        smoke_config = create_smoke_test_config()
        
        # 創建低配置模型
        model_config = ModelConfig(
            price_frame_shape=(smoke_config['n_stocks'], smoke_config['sequence_length'], 27),
            fundamental_dim=10,  # 簡化基本面特徵
            n_stocks=smoke_config['n_stocks'],
            hidden_dim=64,       # 大幅降低隱藏層維度
            num_heads=4,         # 減少注意力頭數
            num_layers=2,        # 減少層數
            dropout=0.1
        )
        
        model = TSEAlphaModel(model_config)
        
        # 計算模型參數數量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 模型創建成功")
        print(f"   總參數: {total_params:,}")
        print(f"   可訓練參數: {trainable_params:,}")
        print(f"   預估記憶體: {total_params * 4 / 1e6:.1f}MB")
        
        return model, model_config
        
    except Exception as e:
        print(f"❌ 模型創建失敗: {e}")
        traceback.print_exc()
        return None, None

def test_data_loading():
    """測試資料載入 (小數據集)"""
    print("\n🔍 測試資料載入...")
    
    try:
        from models.data_loader import TSEDataLoader, DataConfig
        from configs.hardware_configs import create_smoke_test_config
        
        smoke_config = create_smoke_test_config()
        
        # 創建小數據集配置
        data_config = DataConfig(
            symbols=['2330', '2317', '2603'],  # 只用3檔股票
            train_start_date='2024-01-01',
            train_end_date='2024-01-31',       # 只用1個月資料
            val_start_date='2024-02-01',
            val_end_date='2024-02-15',
            test_start_date='2024-02-16',
            test_end_date='2024-02-29',
            sequence_length=smoke_config['sequence_length'],
            batch_size=smoke_config['batch_size'],
            num_workers=0  # 避免多進程問題
        )
        
        loader = TSEDataLoader(data_config)
        
        print(f"✅ 資料載入器創建成功")
        print(f"   股票數: {len(data_config.symbols)}")
        print(f"   批次大小: {data_config.batch_size}")
        print(f"   序列長度: {data_config.sequence_length}")
        
        return loader, data_config
        
    except Exception as e:
        print(f"❌ 資料載入失敗: {e}")
        traceback.print_exc()
        return None, None

def test_forward_pass(model, data_config):
    """測試前向傳播 (GPU 記憶體測試)"""
    print("\n🔍 測試前向傳播...")
    
    try:
        from configs.hardware_configs import create_smoke_test_config
        
        smoke_config = create_smoke_test_config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 創建測試輸入
        batch_size = smoke_config['batch_size']
        n_stocks = smoke_config['n_stocks']
        seq_len = smoke_config['sequence_length']
        
        test_input = {
            'price_frame': torch.randn(batch_size, n_stocks, seq_len, 27, device=device),
            'fundamental': torch.randn(batch_size, 10, device=device),
            'account': torch.randn(batch_size, 4, device=device)
        }
        
        # 記錄 GPU 記憶體使用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1e6
        
        # 前向傳播
        with torch.no_grad():
            outputs = model(test_input)
        
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / 1e6
            memory_used = memory_after - memory_before
            
            print(f"✅ 前向傳播成功")
            print(f"   GPU 記憶體使用: {memory_used:.1f}MB")
            print(f"   總 GPU 記憶體: {memory_after:.1f}MB")
        else:
            print(f"✅ 前向傳播成功 (CPU)")
        
        # 檢查輸出形狀
        for key, value in outputs.items():
            print(f"   {key}: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向傳播失敗: {e}")
        traceback.print_exc()
        return False

def test_training_step(model, data_config):
    """測試訓練步驟 (梯度計算)"""
    print("\n🔍 測試訓練步驟...")
    
    try:
        from configs.hardware_configs import create_smoke_test_config
        
        smoke_config = create_smoke_test_config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 創建優化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=smoke_config['learning_rate'])
        criterion = torch.nn.MSELoss()
        
        # 創建測試資料
        batch_size = smoke_config['batch_size']
        n_stocks = smoke_config['n_stocks']
        seq_len = smoke_config['sequence_length']
        
        test_input = {
            'price_frame': torch.randn(batch_size, n_stocks, seq_len, 27, device=device),
            'fundamental': torch.randn(batch_size, 10, device=device),
            'account': torch.randn(batch_size, 4, device=device)
        }
        test_labels = torch.randn(batch_size, 1, device=device)
        
        # 記錄初始 loss
        model.eval()
        with torch.no_grad():
            initial_outputs = model(test_input)
            initial_loss = criterion(initial_outputs['value'], test_labels)
        
        # 訓練步驟
        model.train()
        optimizer.zero_grad()
        
        outputs = model(test_input)
        loss = criterion(outputs['value'], test_labels)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 檢查 loss 變化
        model.eval()
        with torch.no_grad():
            final_outputs = model(test_input)
            final_loss = criterion(final_outputs['value'], test_labels)
        
        print(f"✅ 訓練步驟成功")
        print(f"   初始 Loss: {initial_loss.item():.6f}")
        print(f"   最終 Loss: {final_loss.item():.6f}")
        print(f"   Loss 變化: {(final_loss - initial_loss).item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 訓練步驟失敗: {e}")
        traceback.print_exc()
        return False

def test_environment_creation():
    """測試交易環境創建"""
    print("\n🔍 測試交易環境...")
    
    try:
        from gym_env.env import TSEAlphaEnv
        
        # 創建小規模環境
        env = TSEAlphaEnv(
            symbols=['2330', '2317', '2603'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_cash=100000.0  # 10萬初始資金
        )
        
        # 重置環境
        obs, info = env.reset()
        
        print(f"✅ 環境創建成功")
        print(f"   觀測空間: {env.observation_space}")
        print(f"   動作空間: {env.action_space}")
        print(f"   初始 NAV: {info['nav']:,.0f}")
        
        # 測試幾步動作
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   步驟 {step+1}: reward={reward:.6f}, NAV={info['nav']:,.0f}")
            
            if terminated or truncated:
                break
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 環境測試失敗: {e}")
        traceback.print_exc()
        return False

def main():
    """主要煙霧測試流程"""
    print("=" * 60)
    print("🧪 TSE Alpha 煙霧測試 - GTX 1660 Ti 專用")
    print("=" * 60)
    
    start_time = time.time()
    results = []
    
    # 1. GPU 檢查
    results.append(("GPU 可用性", check_gpu_availability()))
    
    # 2. 模組導入
    results.append(("模組導入", test_basic_imports()))
    
    # 3. 模型創建
    model, model_config = test_model_creation()
    results.append(("模型創建", model is not None))
    
    # 4. 資料載入
    loader, data_config = test_data_loading()
    results.append(("資料載入", loader is not None))
    
    # 5. 前向傳播 (如果模型創建成功)
    if model is not None:
        results.append(("前向傳播", test_forward_pass(model, data_config)))
        results.append(("訓練步驟", test_training_step(model, data_config)))
    
    # 6. 環境測試
    results.append(("交易環境", test_environment_creation()))
    
    # 總結結果
    print("\n" + "=" * 60)
    print("📊 煙霧測試結果總結")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"{test_name:15} : {status}")
        if success:
            passed += 1
    
    elapsed_time = time.time() - start_time
    
    print(f"\n通過率: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"執行時間: {elapsed_time:.1f} 秒")
    
    if passed == total:
        print("\n🎉 所有煙霧測試通過！GTX 1660 Ti 環境可用")
        return True
    else:
        print(f"\n⚠️  {total-passed} 項測試失敗，需要修正")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)