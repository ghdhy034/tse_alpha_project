#!/usr/bin/env python3
"""
診斷和修復模組導入問題
解決 'No module named market_data_collector' 錯誤
"""

import sys
import os
from pathlib import Path

def diagnose_import_issue():
    """診斷模組導入問題"""
    print("🔍 診斷模組導入問題")
    print("=" * 50)
    
    # 1. 檢查當前工作目錄
    current_dir = Path.cwd()
    print(f"📁 當前工作目錄: {current_dir}")
    
    # 2. 檢查 Python 路徑
    print(f"\n📋 Python 路徑:")
    for i, path in enumerate(sys.path):
        print(f"  {i+1}. {path}")
    
    # 3. 檢查 market_data_collector 目錄
    mdc_path = current_dir / "market_data_collector"
    print(f"\n📂 market_data_collector 檢查:")
    print(f"   路徑: {mdc_path}")
    print(f"   存在: {mdc_path.exists()}")
    
    if mdc_path.exists():
        print(f"   是目錄: {mdc_path.is_dir()}")
        
        # 檢查 __init__.py
        init_file = mdc_path / "__init__.py"
        print(f"   __init__.py: {init_file.exists()}")
        
        # 檢查 utils 目錄
        utils_dir = mdc_path / "utils"
        print(f"   utils/ 目錄: {utils_dir.exists()}")
        
        if utils_dir.exists():
            utils_init = utils_dir / "__init__.py"
            config_file = utils_dir / "config.py"
            db_file = utils_dir / "db.py"
            
            print(f"   utils/__init__.py: {utils_init.exists()}")
            print(f"   utils/config.py: {config_file.exists()}")
            print(f"   utils/db.py: {db_file.exists()}")
    
    # 4. 測試不同的導入方式
    print(f"\n🧪 測試導入方式:")
    
    # 方式 1: 直接導入
    try:
        import market_data_collector
        print("✅ 方式 1: import market_data_collector - 成功")
    except ImportError as e:
        print(f"❌ 方式 1: import market_data_collector - 失敗: {e}")
    
    # 方式 2: 添加路徑後導入
    try:
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        import market_data_collector
        print("✅ 方式 2: 添加當前目錄到 sys.path - 成功")
    except ImportError as e:
        print(f"❌ 方式 2: 添加當前目錄到 sys.path - 失敗: {e}")
    
    # 方式 3: 測試 utils 模組
    try:
        from market_data_collector.utils import config
        print("✅ 方式 3: from market_data_collector.utils import config - 成功")
    except ImportError as e:
        print(f"❌ 方式 3: from market_data_collector.utils import config - 失敗: {e}")
    
    # 方式 4: 測試 db 模組
    try:
        from market_data_collector.utils import db
        print("✅ 方式 4: from market_data_collector.utils import db - 成功")
    except ImportError as e:
        print(f"❌ 方式 4: from market_data_collector.utils import db - 失敗: {e}")

def check_missing_init_files():
    """檢查缺失的 __init__.py 檔案"""
    print(f"\n📝 檢查 __init__.py 檔案:")
    
    current_dir = Path.cwd()
    
    # 需要檢查的目錄
    dirs_to_check = [
        "market_data_collector",
        "market_data_collector/utils",
        "market_data_collector/fetch_data",
        "data_pipeline",
        "gym_env",
        "backtest"
    ]
    
    missing_init = []
    
    for dir_path in dirs_to_check:
        full_path = current_dir / dir_path
        init_file = full_path / "__init__.py"
        
        if full_path.exists() and full_path.is_dir():
            if init_file.exists():
                print(f"✅ {dir_path}/__init__.py")
            else:
                print(f"❌ {dir_path}/__init__.py - 缺失")
                missing_init.append(init_file)
        else:
            print(f"⚠️  {dir_path} - 目錄不存在")
    
    return missing_init

def create_missing_init_files(missing_files):
    """創建缺失的 __init__.py 檔案"""
    if not missing_files:
        print("\n✅ 所有必要的 __init__.py 檔案都存在")
        return
    
    print(f"\n🔧 創建缺失的 __init__.py 檔案:")
    
    for init_file in missing_files:
        try:
            # 確保目錄存在
            init_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 創建 __init__.py
            init_file.write_text('"""模組初始化檔案"""\n')
            print(f"✅ 創建: {init_file}")
            
        except Exception as e:
            print(f"❌ 創建失敗 {init_file}: {e}")

def test_chip_features_import():
    """測試籌碼面特徵導入"""
    print(f"\n🧪 測試籌碼面特徵導入:")
    
    try:
        # 確保路徑正確
        current_dir = Path.cwd()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        # 測試 data_pipeline 導入
        from data_pipeline import features
        print("✅ from data_pipeline import features - 成功")
        
        # 測試特徵引擎
        engine = features.FeatureEngine()
        print("✅ FeatureEngine() 初始化 - 成功")
        
        # 測試籌碼面指標
        chip_indicators = features.ChipIndicators()
        print("✅ ChipIndicators() 初始化 - 成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 籌碼面特徵導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 籌碼面特徵測試失敗: {e}")
        return False

def suggest_fixes():
    """建議修復方案"""
    print(f"\n💡 修復建議:")
    print("1. 確保所有目錄都有 __init__.py 檔案")
    print("2. 檢查 Python 路徑設定")
    print("3. 使用相對導入或絕對路徑")
    print("4. 確認工作目錄正確")

def main():
    """主函數"""
    print("🚀 模組導入問題診斷和修復")
    
    # 1. 診斷問題
    diagnose_import_issue()
    
    # 2. 檢查 __init__.py 檔案
    missing_init = check_missing_init_files()
    
    # 3. 創建缺失的檔案
    create_missing_init_files(missing_init)
    
    # 4. 測試籌碼面特徵
    chip_success = test_chip_features_import()
    
    # 5. 建議修復方案
    if not chip_success:
        suggest_fixes()
    
    print(f"\n" + "=" * 50)
    if chip_success:
        print("🎉 模組導入問題修復成功！")
        print("💡 籌碼面特徵現在可以正常使用了")
    else:
        print("💥 仍有問題需要進一步修復")

if __name__ == "__main__":
    main()