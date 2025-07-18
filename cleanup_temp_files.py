#!/usr/bin/env python3
"""
臨時檔案清理工具
自動整理和移動臨時測試檔案到 tests/temp/ 資料夾
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

def cleanup_temp_files():
    """清理主目錄中的臨時檔案"""
    print("🧹 開始清理臨時檔案...")
    
    # 確保目標資料夾存在
    temp_dir = Path("tests/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 當前日期標記
    date_suffix = datetime.now().strftime("_%Y%m%d")
    
    # 需要清理的檔案模式
    patterns_to_clean = [
        "tmp_rovodev_*.py",
        "run_*_test.bat",
        "run_*_fix.bat", 
        "test_*.py"  # 一次性測試檔案
    ]
    
    moved_files = []
    
    # 掃描主目錄
    for pattern in patterns_to_clean:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                # 生成新檔名（加上日期）
                stem = file_path.stem
                suffix = file_path.suffix
                new_name = f"{stem}{date_suffix}{suffix}"
                target_path = temp_dir / new_name
                
                # 移動檔案
                try:
                    shutil.move(str(file_path), str(target_path))
                    moved_files.append((str(file_path), str(target_path)))
                    print(f"✅ 移動: {file_path} → {target_path}")
                except Exception as e:
                    print(f"❌ 移動失敗: {file_path} - {e}")
    
    # 總結
    if moved_files:
        print(f"\n🎉 成功移動 {len(moved_files)} 個檔案到 tests/temp/")
        print("主目錄現在更整潔了！")
    else:
        print("\n✨ 主目錄已經很整潔，沒有需要移動的臨時檔案")
    
    return moved_files

def list_temp_files():
    """列出 tests/temp/ 中的檔案"""
    temp_dir = Path("tests/temp")
    if not temp_dir.exists():
        print("📁 tests/temp/ 資料夾不存在")
        return
    
    files = list(temp_dir.glob("*"))
    if files:
        print(f"\n📂 tests/temp/ 中有 {len(files)} 個檔案:")
        for file_path in sorted(files):
            if file_path.is_file():
                size = file_path.stat().st_size
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                print(f"  📄 {file_path.name} ({size} bytes, {mtime.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("\n📂 tests/temp/ 資料夾是空的")

def main():
    print("🗂️  臨時檔案管理工具")
    print("=" * 50)
    
    # 列出當前臨時檔案
    print("📋 當前狀況:")
    list_temp_files()
    
    # 清理主目錄
    print("\n🧹 清理主目錄:")
    moved_files = cleanup_temp_files()
    
    # 更新後的狀況
    if moved_files:
        print("\n📋 清理後狀況:")
        list_temp_files()

if __name__ == "__main__":
    main()