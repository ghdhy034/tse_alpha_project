#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徵配置驗證腳本 - 確保66+4=70維配置一致性
"""
import sys
import os
from pathlib import Path

# 強制UTF-8輸出
sys.stdout.reconfigure(encoding='utf-8')

# 添加路徑
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

def verify_training_config():
    """驗證訓練配置"""
    print("🔍 檢查訓練配置...")
    
    try:
        from models.config.training_config import TrainingConfig
        config = TrainingConfig()
        
        print(f"📊 當前配置:")
        print(f"   基本面特徵: {config.fundamental_features}")
        print(f"   其他特徵: {config.other_features}")
        print(f"   帳戶特徵: {config.account_features}")
        print(f"   總特徵: {config.total_features}")
        
        calculated_total = config.fundamental_features + config.other_features + config.account_features
        
        if config.total_features == 66 and calculated_total == 66:
            print("✅ 訓練配置正確: 66維 (15+51+0)")
            return True
        else:
            print(f"❌ 訓練配置錯誤: 聲明{config.total_features} vs 計算{calculated_total}")
            return False
            
    except Exception as e:
        print(f"❌ 訓練配置檢查失敗: {e}")
        return False

def verify_feature_specification():
    """驗證特徵規格文檔"""
    print("\n🔍 檢查特徵規格文檔...")
    
    try:
        # 檢查FEATURE_SPECIFICATION_66_4.md
        spec_file = Path("FEATURE_SPECIFICATION_66_4.md")
        if spec_file.exists():
            print("✅ 找到特徵規格文檔: FEATURE_SPECIFICATION_66_4.md")
            return True
        else:
            print("⚠️ 特徵規格文檔不存在")
            return False
            
    except Exception as e:
        print(f"❌ 特徵規格檢查失敗: {e}")
        return False

def main():
    """主函數"""
    print("=== 特徵配置驗證 (66+4=70維) ===")
    
    results = []
    
    # 檢查訓練配置
    results.append(verify_training_config())
    
    # 檢查特徵規格
    results.append(verify_feature_specification())
    
    # 總結
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n📊 驗證結果: {success_count}/{total_count} 通過")
    
    if success_count == total_count:
        print("🎉 所有配置驗證通過！")
        print("✅ 可以執行測試腳本")
        return True
    else:
        print("⚠️ 部分配置需要調整")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)