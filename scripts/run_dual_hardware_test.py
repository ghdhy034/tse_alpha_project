#!/usr/bin/env python3
"""
TSE Alpha 雙硬體環境測試腳本
自動檢測硬體並執行對應測試
"""
import sys
import os
import subprocess
import time
from pathlib import Path

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))

def detect_and_test_hardware():
    """檢測硬體並執行對應測試"""
    print("=" * 60)
    print("🔍 TSE Alpha 雙硬體環境自動測試")
    print("=" * 60)
    
    try:
        from configs.hardware_configs import HardwareDetector, ConfigManager
        
        # 檢測硬體
        gpu_info = HardwareDetector.detect_gpu()
        profile = HardwareDetector.get_hardware_profile()
        
        print(f"檢測到硬體:")
        print(f"  GPU: {gpu_info['name']}")
        print(f"  VRAM: {gpu_info['memory_gb']:.1f}GB")
        print(f"  配置檔案: {profile}")
        
        # 根據硬體執行對應測試
        if profile == 'gtx1660ti':
            print("\n🧪 執行 GTX 1660 Ti 煙霧測試...")
            result = run_gtx1660ti_test()
            
        elif profile == 'rtx4090':
            print("\n🚀 執行 RTX 4090 高配置測試...")
            result = run_rtx4090_test()
            
        elif profile in ['high_end', 'low_end']:
            print(f"\n⚙️  執行通用 GPU 測試 ({profile})...")
            result = run_generic_test(profile)
            
        else:
            print("\n💻 執行 CPU 測試...")
            result = run_cpu_test()
        
        return result
        
    except Exception as e:
        print(f"❌ 硬體檢測失敗: {e}")
        return False

def run_gtx1660ti_test():
    """執行 GTX 1660 Ti 專用測試"""
    try:
        print("執行 GTX 1660 Ti 煙霧測試...")
        result = subprocess.run([
            sys.executable, 'scripts/smoke_test_gtx1660ti.py'
        ], capture_output=True, text=True, timeout=300)  # 5分鐘超時
        
        if result.returncode == 0:
            print("✅ GTX 1660 Ti 測試通過")
            print("建議: 使用此環境進行開發和初步驗證")
            return True
        else:
            print("❌ GTX 1660 Ti 測試失敗")
            print(f"錯誤輸出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ GTX 1660 Ti 測試超時")
        return False
    except Exception as e:
        print(f"❌ GTX 1660 Ti 測試異常: {e}")
        return False

def run_rtx4090_test():
    """執行 RTX 4090 高配置測試"""
    try:
        print("執行 RTX 4090 快速驗證...")
        
        # 先執行快速驗證
        result = subprocess.run([
            sys.executable, 'scripts/full_training_rtx4090.py',
            '--mode', 'supervised',
            '--epochs', '1',
            '--batch-size', '64',
            '--n-stocks', '10',
            '--force'
        ], capture_output=True, text=True, timeout=600)  # 10分鐘超時
        
        if result.returncode == 0:
            print("✅ RTX 4090 快速驗證通過")
            print("建議: 使用此環境進行完整訓練")
            
            # 詢問是否執行完整測試
            response = input("是否執行完整配置測試？(y/N): ")
            if response.lower() == 'y':
                return run_rtx4090_full_test()
            return True
        else:
            print("❌ RTX 4090 快速驗證失敗")
            print(f"錯誤輸出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ RTX 4090 測試超時")
        return False
    except Exception as e:
        print(f"❌ RTX 4090 測試異常: {e}")
        return False

def run_rtx4090_full_test():
    """執行 RTX 4090 完整配置測試"""
    try:
        print("執行 RTX 4090 完整配置測試...")
        
        result = subprocess.run([
            sys.executable, 'scripts/full_training_rtx4090.py',
            '--mode', 'supervised',
            '--epochs', '3',
            '--batch-size', '128',
            '--n-stocks', '50',
            '--force'
        ], capture_output=True, text=True, timeout=1800)  # 30分鐘超時
        
        if result.returncode == 0:
            print("✅ RTX 4090 完整配置測試通過")
            print("🎉 可以進行大規模生產訓練！")
            return True
        else:
            print("❌ RTX 4090 完整配置測試失敗")
            print(f"錯誤輸出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ RTX 4090 完整測試超時")
        return False
    except Exception as e:
        print(f"❌ RTX 4090 完整測試異常: {e}")
        return False

def run_generic_test(profile):
    """執行通用 GPU 測試"""
    try:
        print(f"執行 {profile} GPU 測試...")
        
        # 使用煙霧測試但調整配置
        result = subprocess.run([
            sys.executable, 'scripts/smoke_test_gtx1660ti.py'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {profile} GPU 測試通過")
            return True
        else:
            print(f"❌ {profile} GPU 測試失敗")
            return False
            
    except Exception as e:
        print(f"❌ {profile} GPU 測試異常: {e}")
        return False

def run_cpu_test():
    """執行 CPU 測試"""
    try:
        print("執行 CPU 模式測試...")
        
        # 設定 CPU 模式
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ''
        
        result = subprocess.run([
            sys.executable, 'scripts/smoke_test_gtx1660ti.py'
        ], env=env, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ CPU 模式測試通過")
            print("⚠️  建議使用 GPU 進行實際訓練")
            return True
        else:
            print("❌ CPU 模式測試失敗")
            return False
            
    except Exception as e:
        print(f"❌ CPU 測試異常: {e}")
        return False

def provide_recommendations(test_result, profile):
    """提供使用建議"""
    print("\n" + "=" * 60)
    print("💡 使用建議")
    print("=" * 60)
    
    if not test_result:
        print("❌ 測試失敗，請檢查環境配置")
        return
    
    if profile == 'gtx1660ti':
        print("🔧 GTX 1660 Ti 環境建議:")
        print("  ✅ 適合: 開發、調試、煙霧測試")
        print("  ✅ 配置: 低批次、小數據集、快速迭代")
        print("  ⚠️  不適合: 大規模完整訓練")
        print("\n📝 下一步:")
        print("  1. 使用此環境進行開發和測試")
        print("  2. 完成功能驗證後轉移到 RTX 4090")
        
    elif profile == 'rtx4090':
        print("🚀 RTX 4090 環境建議:")
        print("  ✅ 適合: 完整訓練、大規模實驗、生產部署")
        print("  ✅ 配置: 高批次、完整數據集、長時間訓練")
        print("  ✅ 特色: 支援大規模 Optuna 超參數搜索")
        print("\n📝 下一步:")
        print("  1. 執行完整的 180 檔股票訓練")
        print("  2. 進行大規模超參數優化")
        print("  3. 部署生產模型")
        
    else:
        print(f"⚙️  {profile} 環境建議:")
        print("  ✅ 可用於中等規模訓練")
        print("  ⚠️  建議根據 VRAM 調整批次大小")

def main():
    """主程式"""
    start_time = time.time()
    
    # 檢測並測試硬體
    test_result = detect_and_test_hardware()
    
    # 獲取配置檔案
    try:
        from configs.hardware_configs import HardwareDetector
        profile = HardwareDetector.get_hardware_profile()
    except:
        profile = 'unknown'
    
    # 提供建議
    provide_recommendations(test_result, profile)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  總執行時間: {elapsed_time:.1f} 秒")
    
    return test_result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)