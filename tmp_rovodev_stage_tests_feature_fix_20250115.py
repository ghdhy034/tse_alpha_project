#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
階段測試腳本特徵維度修復 - 統一調整為70維配置
"""
import sys
import os
from pathlib import Path

# 強制UTF-8輸出
sys.stdout.reconfigure(encoding='utf-8')

def fix_stage_test_scripts():
    """修復所有階段測試腳本的特徵維度期望"""
    
    stage_files = [
        "tmp_rovodev_stage1_basic_verification_20250115.py",
        "tmp_rovodev_stage3_multi_stock_test_20250115.py", 
        "tmp_rovodev_stage4_training_validation_20250115.py",
        "tmp_rovodev_stage5_stability_test_20250115.py"
    ]
    
    print("🔧 修復階段測試腳本特徵維度期望...")
    
    for file_path in stage_files:
        if Path(file_path).exists():
            print(f"📝 檢查 {file_path}...")
            
            # 讀取文件內容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 檢查是否需要修復
            needs_fix = False
            
            # 常見的需要修復的模式
            fix_patterns = [
                ("75維", "70維"),
                ("75個特徵", "70個特徵"),
                ("expected_features = 75", "expected_features = 70"),
                ("feature_count == 75", "feature_count == 70"),
                ("總計75維", "總計70維"),
                ("68維", "66維"),
                ("68個特徵", "66個特徵"),
                ("expected_without_account = 68", "expected_without_account = 66"),
                ("72維", "70維"),
                ("72個特徵", "70個特徵")
            ]
            
            original_content = content
            for old_pattern, new_pattern in fix_patterns:
                if old_pattern in content:
                    content = content.replace(old_pattern, new_pattern)
                    needs_fix = True
                    print(f"   🔄 修復: {old_pattern} → {new_pattern}")
            
            # 如果有修改，寫回文件
            if needs_fix:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"   ✅ {file_path} 修復完成")
            else:
                print(f"   ✅ {file_path} 無需修復")
        else:
            print(f"   ⚠️ {file_path} 不存在")
    
    print("\n🎉 階段測試腳本特徵維度修復完成！")

def verify_feature_consistency():
    """驗證特徵配置一致性"""
    print("\n🔍 驗證特徵配置一致性...")
    
    try:
        from models.config.training_config import TrainingConfig
        config = TrainingConfig()
        
        print(f"📊 訓練配置:")
        print(f"   基本面特徵: {config.fundamental_features}")
        print(f"   其他特徵: {config.other_features}")
        print(f"   帳戶特徵: {config.account_features}")
        print(f"   總特徵: {config.total_features}")
        
        # 驗證配置
        calculated_total = config.fundamental_features + config.other_features + config.account_features
        
        if config.total_features == 70 and calculated_total == 70:
            print("✅ 訓練配置一致: 70維 (15+51+4)")
            return True
        else:
            print(f"❌ 訓練配置不一致: 聲明{config.total_features} vs 計算{calculated_total}")
            return False
            
    except Exception as e:
        print(f"❌ 配置驗證失敗: {e}")
        return False

def main():
    """主函數"""
    print("=== 階段測試腳本特徵維度修復 ===")
    
    # 修復階段測試腳本
    fix_stage_test_scripts()
    
    # 驗證配置一致性
    config_ok = verify_feature_consistency()
    
    if config_ok:
        print("\n🎉 所有修復完成！現在可以執行測試腳本：")
        print("   1. run_quick_fix_test_20250115.bat")
        print("   2. run_stage2_single_stock_20250115.bat") 
        print("   3. run_complete_smoke_test_20250115.bat")
    else:
        print("\n⚠️ 配置仍有問題，需要進一步檢查")

if __name__ == "__main__":
    main()