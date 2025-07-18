#!/usr/bin/env python3
"""
測試訓練腳本的所有導入是否正常
修復導入問題後的驗證腳本
"""

import sys
from pathlib import Path

def test_imports():
    """測試所有必要的導入"""
    print("🔍 測試訓練腳本導入...")
    
    try:
        # 測試模型架構導入
        print("📦 測試模型架構...")
        from models.model_architecture import TSEAlphaModel, ModelConfig
        print("✅ TSEAlphaModel 導入成功")
        print("✅ ModelConfig 導入成功")
        
        # 測試資料載入器導入
        print("📊 測試資料載入器...")
        from models.data_loader import TSEDataLoader, TSEAlphaDataLoader, DataConfig
        print("✅ TSEDataLoader 導入成功")
        print("✅ TSEAlphaDataLoader 別名導入成功")
        print("✅ DataConfig 導入成功")
        
        # 測試訓練器導入
        print("🏋️ 測試訓練器...")
        from models.trainer import ModelTrainer
        print("✅ ModelTrainer 導入成功")
        
        # 測試環境導入
        print("🎮 測試交易環境...")
        from gym_env.env import TSEAlphaEnv
        print("✅ TSEAlphaEnv 導入成功")
        
        # 測試股票配置導入
        print("📈 測試股票配置...")
        from stock_config import TRAIN_STOCKS, VALIDATION_STOCKS, TEST_STOCKS
        print("✅ 股票配置導入成功")
        print(f"  訓練股票: {len(TRAIN_STOCKS)} 檔")
        print(f"  驗證股票: {len(VALIDATION_STOCKS)} 檔")
        print(f"  測試股票: {len(TEST_STOCKS)} 檔")
        
        # 測試特徵工程導入
        print("🔧 測試特徵工程...")
        from data_pipeline.features import FeatureEngine
        print("✅ FeatureEngine 導入成功")
        
        print("\n🎉 所有導入測試通過！")
        return True
        
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他錯誤: {e}")
        return False

def test_model_creation():
    """測試模型創建"""
    print("\n🔧 測試模型創建...")
    
    try:
        from models.model_architecture import TSEAlphaModel, ModelConfig
        from stock_config import TRAIN_STOCKS
        
        # 創建模型配置
        model_config = ModelConfig(
            price_frame_shape=(len(TRAIN_STOCKS), 64, 27),  # (n_stocks, seq_len, features)
            fundamental_dim=43,
            account_dim=4,
            hidden_dim=640,
            num_heads=8,
            num_layers=6,
            dropout=0.2,
            n_stocks=len(TRAIN_STOCKS),
            max_position=300
        )
        
        # 創建模型
        model = TSEAlphaModel(model_config)
        
        # 計算參數數量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 模型創建成功")
        print(f"  總參數: {total_params:,}")
        print(f"  可訓練參數: {trainable_params:,}")
        print(f"  預估記憶體: {total_params * 4 / 1024**2:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型創建失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader_creation():
    """測試資料載入器創建"""
    print("\n📊 測試資料載入器創建...")
    
    try:
        from models.data_loader import TSEDataLoader, DataConfig
        from stock_config import TRAIN_STOCKS, VALIDATION_STOCKS
        
        # 創建資料配置
        data_config = DataConfig(
            symbols=TRAIN_STOCKS[:5],  # 只用前5檔測試
            train_start_date='2024-01-01',
            train_end_date='2024-03-31',
            val_start_date='2024-04-01',
            val_end_date='2024-05-31',
            test_start_date='2024-06-01',
            test_end_date='2024-06-30',
            sequence_length=64,
            batch_size=32,
            num_workers=0  # 避免多進程問題
        )
        
        # 創建資料載入器
        data_loader = TSEDataLoader(data_config)
        
        print(f"✅ 資料載入器創建成功")
        print(f"  配置股票: {len(data_config.symbols)} 檔")
        print(f"  序列長度: {data_config.sequence_length}")
        print(f"  批次大小: {data_config.batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ 資料載入器創建失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yaml_config():
    """測試YAML配置載入"""
    print("\n⚙️ 測試YAML配置載入...")
    
    try:
        import yaml
        
        config_path = "training_config_full.yaml"
        if not Path(config_path).exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 檢查關鍵配置段
        required_sections = ['system', 'data', 'model', 'training']
        for section in required_sections:
            if section not in config:
                print(f"❌ 缺少配置段: {section}")
                return False
        
        print("✅ YAML配置載入成功")
        print(f"  系統設備: {config['system']['device']}")
        print(f"  批次大小: {config['data']['loading']['batch_size']}")
        print(f"  模型維度: {config['model']['transformer']['d_model']}")
        print(f"  最大epochs: {config['training']['max_epochs']}")
        
        return True
        
    except Exception as e:
        print(f"❌ YAML配置載入失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("=" * 60)
    print("TSE Alpha 訓練導入測試")
    print("=" * 60)
    
    tests = [
        ("基本導入測試", test_imports),
        ("模型創建測試", test_model_creation),
        ("資料載入器測試", test_data_loader_creation),
        ("YAML配置測試", test_yaml_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} 失敗")
    
    print("\n" + "="*60)
    print("測試結果總結")
    print("="*60)
    print(f"通過: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有測試通過！訓練腳本準備就緒。")
        print("\n下一步可以執行:")
        print("  python pre_training_check.py")
        print("  run_full_training.bat")
        return True
    else:
        print("⚠️ 部分測試失敗，請檢查上述錯誤。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)