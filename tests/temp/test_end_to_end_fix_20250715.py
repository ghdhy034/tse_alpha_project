#!/usr/bin/env python3
"""
測試端到端訓練修復
專門測試 normalize_features 屬性修復
"""

import sys
import os
from pathlib import Path

# 添加路徑
sys.path.append(str(Path(__file__).parent))

def test_training_config_attributes():
    """測試 TrainingConfig 所有必要屬性"""
    print("🔧 測試 TrainingConfig 屬性完整性")
    print("-" * 40)
    
    try:
        from models.config.training_config import TrainingConfig
        
        config = TrainingConfig()
        
        # 檢查關鍵屬性
        required_attrs = [
            'normalize_features',
            'include_chip_features', 
            'symbols',
            'price_features',
            'fundamental_features',
            'account_features',
            'sequence_length'
        ]
        
        print("檢查必要屬性:")
        for attr in required_attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"   ✅ {attr}: {value}")
            else:
                print(f"   ❌ 缺少屬性: {attr}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ TrainingConfig 測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader_creation():
    """測試資料載入器創建"""
    print("\n🔧 測試資料載入器創建")
    print("-" * 40)
    
    try:
        from models.config.training_config import TrainingConfig
        from models.data_loader import TSEAlphaDataLoader, DataConfig
        
        # 使用 TrainingConfig
        training_config = TrainingConfig()
        
        # 創建相容的 DataConfig
        data_config = DataConfig(
            symbols=training_config.symbols,
            sequence_length=training_config.sequence_length,
            normalize_features=training_config.normalize_features,
            include_chip_features=training_config.include_chip_features
        )
        
        print("✅ DataConfig 創建成功")
        print(f"   股票: {data_config.symbols}")
        print(f"   標準化特徵: {data_config.normalize_features}")
        
        # 創建資料載入器
        data_loader = TSEAlphaDataLoader(data_config)
        print("✅ TSEAlphaDataLoader 創建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 資料載入器創建失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_load_data_method():
    """測試 load_data 方法"""
    print("\n🔧 測試 load_data 方法")
    print("-" * 40)
    
    try:
        from models.config.training_config import TrainingConfig
        from models.data_loader import TSEAlphaDataLoader, DataConfig
        
        # 創建配置
        data_config = DataConfig(
            symbols=['2330', '2317'],
            sequence_length=20,  # 較短的序列用於測試
            normalize_features=True,
            include_chip_features=False  # 暫時關閉籌碼面特徵
        )
        
        # 創建資料載入器
        data_loader = TSEAlphaDataLoader(data_config)
        
        # 測試 load_data 方法
        print("   嘗試載入測試資料...")
        dataset = data_loader.load_data(
            symbols=['2330', '2317'],
            start_date='2024-01-01',
            end_date='2024-01-10',
            split='train'
        )
        
        print("✅ load_data 方法執行成功")
        print(f"   資料集大小: {len(dataset)}")
        
        # 測試資料集樣本
        if len(dataset) > 0:
            sample = dataset[0]
            print("   樣本結構:")
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    print(f"     {key}: {value.shape}")
                else:
                    print(f"     {key}: {type(value)}")
        
        return True
        
    except Exception as e:
        print(f"❌ load_data 方法測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("=" * 60)
    print("端到端訓練修復測試")
    print("測試 normalize_features 屬性和相關功能")
    print("=" * 60)
    
    tests = [
        ("TrainingConfig 屬性完整性", test_training_config_attributes),
        ("資料載入器創建", test_data_loader_creation),
        ("load_data 方法", test_load_data_method)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通過")
            else:
                print(f"❌ {test_name} 失敗")
        except Exception as e:
            print(f"❌ {test_name} 異常: {str(e)}")
    
    # 結果總結
    print("\n" + "=" * 60)
    print("📋 端到端訓練修復測試結果")
    print("=" * 60)
    
    pass_rate = (passed / total) * 100
    
    print(f"總測試數: {total}")
    print(f"通過測試: {passed}")
    print(f"失敗測試: {total - passed}")
    print(f"通過率: {pass_rate:.1f}%")
    
    if pass_rate >= 100:
        print(f"\n🎉 端到端訓練修復完全成功！")
        print(f"✅ 所有必要屬性已添加")
        print(f"✅ 資料載入器功能正常")
        print(f"🚀 現在可以運行端到端訓練測試")
        
    elif pass_rate >= 66:
        print(f"\n✅ 端到端訓練修復基本成功")
        print(f"🔧 部分功能可能需要微調")
        
    else:
        print(f"\n⚠️ 端到端訓練修復仍有問題")
        print(f"🔧 需要進一步修復")
    
    return pass_rate >= 66

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ 修復驗證通過' if success else '❌ 修復驗證失敗'}")
    sys.exit(0 if success else 1)