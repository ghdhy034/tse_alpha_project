#!/usr/bin/env python3
"""
驗證重複實作清理結果
確認系統中只保留統一的實作版本
"""
import sys
import os
from pathlib import Path

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))

def validate_training_config():
    """驗證 TrainingConfig 統一性"""
    print("🔍 檢查 TrainingConfig 統一性...")
    
    try:
        # 應該只有這個主要版本
        from models.config.training_config import TrainingConfig
        config = TrainingConfig()
        
        print(f"✅ 主要 TrainingConfig 載入成功")
        print(f"   - sequence_length: {config.sequence_length}")
        print(f"   - price_features: {config.price_features}")
        print(f"   - fundamental_features: {config.fundamental_features}")
        print(f"   - n_stocks: {config.n_stocks}")
        
        return True
        
    except Exception as e:
        print(f"❌ TrainingConfig 載入失敗: {e}")
        return False

def validate_data_loader():
    """驗證 DataLoader 統一性"""
    print("\n🔍 檢查 DataLoader 統一性...")
    
    try:
        # 應該只有這個主要版本
        from models.data_loader import TSEDataLoader, DataConfig
        
        config = DataConfig()
        loader = TSEDataLoader(config)
        
        print(f"✅ 主要 TSEDataLoader 載入成功")
        print(f"   - symbols: {len(config.symbols)} 檔股票")
        print(f"   - batch_size: {config.batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ TSEDataLoader 載入失敗: {e}")
        return False

def validate_trainer():
    """驗證 Trainer 使用統一配置"""
    print("\n🔍 檢查 Trainer 配置統一性...")
    
    try:
        from models.trainer import ModelTrainer, TrainingConfig
        from models.model_architecture import TSEAlphaModel, ModelConfig
        
        # 檢查是否使用統一的 TrainingConfig
        training_config = TrainingConfig()
        model_config = ModelConfig()
        model = TSEAlphaModel(model_config)
        trainer = ModelTrainer(model, training_config)
        
        print(f"✅ Trainer 使用統一配置成功")
        print(f"   - device: {trainer.device}")
        print(f"   - model_name: {training_config.model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer 配置檢查失敗: {e}")
        return False

def check_deleted_files():
    """檢查已刪除的重複檔案"""
    print("\n🔍 檢查已刪除的重複檔案...")
    
    deleted_files = [
        "models/data/data_loader.py",
        "models/data/dataset.py"
    ]
    
    all_deleted = True
    for file_path in deleted_files:
        if Path(file_path).exists():
            print(f"❌ 檔案仍存在: {file_path}")
            all_deleted = False
        else:
            print(f"✅ 檔案已刪除: {file_path}")
    
    return all_deleted

def validate_stock_config():
    """驗證股票配置統一性"""
    print("\n🔍 檢查股票配置統一性...")
    
    try:
        from stock_config import TRAIN_STOCKS, VALIDATION_STOCKS, TEST_STOCKS, validate_splits
        
        # 檢查分割配置
        is_valid, message = validate_splits()
        
        if is_valid:
            print(f"✅ 股票分割配置正確")
            print(f"   - 訓練集: {len(TRAIN_STOCKS)} 檔")
            print(f"   - 驗證集: {len(VALIDATION_STOCKS)} 檔")
            print(f"   - 測試集: {len(TEST_STOCKS)} 檔")
            return True
        else:
            print(f"❌ 股票分割配置錯誤: {message}")
            return False
            
    except Exception as e:
        print(f"❌ 股票配置檢查失敗: {e}")
        return False

def main():
    """主要驗證流程"""
    print("=" * 60)
    print("🧹 TSE Alpha 重複實作清理驗證")
    print("=" * 60)
    
    results = []
    
    # 執行各項檢查
    results.append(validate_training_config())
    results.append(validate_data_loader())
    results.append(validate_trainer())
    results.append(check_deleted_files())
    results.append(validate_stock_config())
    
    # 總結結果
    print("\n" + "=" * 60)
    print("📊 驗證結果總結")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 所有檢查通過 ({passed}/{total})")
        print("✅ 重複實作清理完成，系統統一性良好")
        return True
    else:
        print(f"⚠️  部分檢查失敗 ({passed}/{total})")
        print("❌ 需要進一步修正")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)