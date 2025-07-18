#!/usr/bin/env python3
"""
簡單調試 TrainingConfig 問題
"""

import sys
import traceback
from pathlib import Path

# 添加路徑
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

print("🔧 簡單調試 TrainingConfig")
print("=" * 40)

try:
    print("1. 導入 TrainingConfig...")
    from models.config.training_config import TrainingConfig
    print("   ✅ 導入成功")
    
    print("\n2. 嘗試創建默認配置...")
    config = TrainingConfig()
    print("   ✅ 創建成功！")
    
    print(f"\n3. 檢查關鍵日期:")
    print(f"   data_start_date: {config.data_start_date}")
    print(f"   train_end_date: {config.train_end_date}")
    print(f"   val_start_date: {config.val_start_date}")
    print(f"   val_end_date: {config.val_end_date}")
    print(f"   test_start_date: {config.test_start_date}")
    print(f"   test_end_date: {config.test_end_date}")
    print(f"   data_end_date: {config.data_end_date}")
    print(f"   effective_test_end: {config.effective_test_end}")
    
    print(f"\n4. 檢查 patience:")
    print(f"   patience: {config.patience}")
    print(f"   early_stopping_patience: {config.early_stopping_patience}")
    
    print(f"\n🎉 TrainingConfig 完全正常！")
    
except Exception as e:
    print(f"\n❌ TrainingConfig 失敗: {e}")
    print(f"錯誤類型: {type(e).__name__}")
    print(f"完整錯誤:\n{traceback.format_exc()}")
    
    # 檢查具體的日期問題
    if "日期順序錯誤" in str(e):
        print(f"\n🔍 日期順序問題分析:")
        error_msg = str(e)
        print(f"   錯誤信息: {error_msg}")
        
        # 手動檢查日期
        print(f"\n   手動檢查當前配置文件中的日期...")
        try:
            config_file = Path("models/config/training_config.py")
            with open(config_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            date_lines = [line.strip() for line in lines if 'date:' in line and '=' in line]
            print(f"   配置文件中的日期行:")
            for i, line in enumerate(date_lines[:10]):  # 只顯示前10行
                print(f"      {i+1}. {line}")
                
        except Exception as file_e:
            print(f"   無法讀取配置文件: {file_e}")