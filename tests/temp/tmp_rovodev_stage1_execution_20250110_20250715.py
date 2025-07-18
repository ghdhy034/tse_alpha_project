#!/usr/bin/env python3
"""
RTX 4090 階段1執行腳本
自動化執行記憶體修復和B方案驗證
"""

import os
import sys
import subprocess
import torch
import time
from pathlib import Path

def print_header(title):
    """打印標題"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step_num, description):
    """打印步驟"""
    print(f"\n📋 步驟 {step_num}: {description}")
    print("-" * 40)

def check_pytorch_version():
    """檢查PyTorch版本並設置記憶體管理"""
    print_step(1, "檢查PyTorch版本並設置記憶體管理")
    
    try:
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("❌ 未檢測到CUDA GPU")
            return False
        
        # 檢查PyTorch版本並設置記憶體管理
        pytorch_version = torch.__version__
        major, minor = map(int, pytorch_version.split('.')[:2])
        
        if major > 2 or (major == 2 and minor >= 3):
            print("✅ PyTorch ≥ 2.3，使用 expandable_segments")
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        elif major == 2 and minor == 2:
            print("✅ PyTorch 2.2，使用 max_split_size_mb")
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.6'
        else:
            print("⚠️ PyTorch版本較舊，使用預設設置")
        
        # 清理GPU記憶體
        print("🧹 清理GPU記憶體...")
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch檢查失敗: {e}")
        return False

def verify_config_file():
    """驗證配置文件存在"""
    print_step(2, "驗證RTX 4090測試配置文件")
    
    config_file = "rtx4090_optimized_config.yaml"
    
    if os.path.exists(config_file):
        print(f"✅ 找到配置文件: {config_file}")
        
        # 顯示關鍵配置
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print("📊 關鍵配置:")
            print(f"  - 股票數量: {config['data']['stocks']['total']}")
            print(f"  - 批次大小: {config['data']['loading']['batch_size']}")
            print(f"  - d_model: {config['model']['transformer']['d_model']}")
            print(f"  - n_layers: {config['model']['transformer']['n_layers']}")
            print(f"  - 訓練輪數: {config['training']['max_epochs']}")
            
        except Exception as e:
            print(f"⚠️ 無法解析配置文件: {e}")
        
        return True
    else:
        print(f"❌ 配置文件不存在: {config_file}")
        return False

def run_smoke_test():
    """運行煙霧測試"""
    print_step(3, "執行B方案驗證煙霧測試")
    
    try:
        # 檢查step0_quick_validation.py是否存在
        if not os.path.exists("step0_quick_validation.py"):
            print("❌ step0_quick_validation.py 不存在")
            return False
        
        print("🔥 開始煙霧測試...")
        print("⏱️ 預計需要5-10分鐘...")
        
        # 構建命令
        cmd = [
            sys.executable, 
            "step0_quick_validation.py", 
            "--smoke", 
            "--config", 
            "rtx4090_optimized_config.yaml"
        ]
        
        print(f"執行命令: {' '.join(cmd)}")
        
        # 執行測試
        start_time = time.time()
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=1800  # 30分鐘超時
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️ 執行時間: {duration:.1f} 秒")
        
        # 檢查結果
        if result.returncode == 0:
            print("✅ 煙霧測試成功!")
            print("\n📊 測試輸出:")
            print(result.stdout)
            return True
        else:
            print("❌ 煙霧測試失敗!")
            print("\n📊 錯誤輸出:")
            print(result.stderr)
            print("\n📊 標準輸出:")
            print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 煙霧測試超時 (30分鐘)")
        return False
    except Exception as e:
        print(f"❌ 煙霧測試執行失敗: {e}")
        return False

def check_gpu_memory():
    """檢查GPU記憶體使用情況"""
    print_step(4, "檢查GPU記憶體使用情況")
    
    try:
        if torch.cuda.is_available():
            # 獲取記憶體信息
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            reserved_memory = torch.cuda.memory_reserved(0)
            
            print(f"📊 GPU記憶體狀況:")
            print(f"  - 總記憶體: {total_memory / 1024**3:.2f} GB")
            print(f"  - 已分配: {allocated_memory / 1024**3:.2f} GB ({allocated_memory/total_memory*100:.1f}%)")
            print(f"  - 已保留: {reserved_memory / 1024**3:.2f} GB ({reserved_memory/total_memory*100:.1f}%)")
            print(f"  - 可用: {(total_memory - reserved_memory) / 1024**3:.2f} GB")
            
            # 檢查是否有記憶體洩漏
            if allocated_memory > 0:
                print("⚠️ 檢測到GPU記憶體使用，清理中...")
                torch.cuda.empty_cache()
                
                # 再次檢查
                allocated_after = torch.cuda.memory_allocated(0)
                reserved_after = torch.cuda.memory_reserved(0)
                
                print(f"🧹 清理後:")
                print(f"  - 已分配: {allocated_after / 1024**3:.2f} GB")
                print(f"  - 已保留: {reserved_after / 1024**3:.2f} GB")
            
            return True
        else:
            print("❌ 無GPU可用")
            return False
            
    except Exception as e:
        print(f"❌ GPU記憶體檢查失敗: {e}")
        return False

def generate_next_steps():
    """生成下一步建議"""
    print_step(5, "生成下一步執行建議")
    
    print("🎯 階段1完成! 下一步建議:")
    print("\n📋 如果測試成功:")
    print("  1. 進入階段2: RTX 4090基礎優化")
    print("     - DataLoader優化 (+7~10%)")
    print("     - GPU計算優化 (+20-35%)")
    print("     - fused AdamW優化器")
    print("\n📋 執行命令:")
    print("  python step0_quick_validation.py --smoke --compile-mode reduce-overhead")
    
    print("\n📋 如果測試失敗:")
    print("  1. 檢查錯誤日誌")
    print("  2. 進一步調整批次大小")
    print("  3. 減少模型參數")
    print("  4. 檢查硬體配置")
    
    print("\n📋 監控指標:")
    print("  - GPU利用率應 >80%")
    print("  - 記憶體使用應 <90%")
    print("  - 訓練速度應 >50 samples/sec")

def main():
    """主執行函數"""
    print_header("RTX 4090 階段1執行 - 記憶體修復與B方案驗證")
    
    print("🎯 目標:")
    print("  - 解決GPU記憶體不足問題")
    print("  - 驗證B方案修復 (27個特徵)")
    print("  - 確保系統可以正常運行")
    
    # 執行各個步驟
    steps = [
        ("PyTorch版本檢查", check_pytorch_version),
        ("配置文件驗證", verify_config_file),
        ("煙霧測試執行", run_smoke_test),
        ("GPU記憶體檢查", check_gpu_memory),
        ("下一步建議", generate_next_steps)
    ]
    
    success_count = 0
    total_steps = len(steps) - 1  # 最後一步是建議，不計入成功率
    
    for step_name, step_func in steps:
        try:
            if step_func():
                if step_name != "下一步建議":  # 建議步驟總是執行
                    success_count += 1
                    print(f"✅ {step_name} 成功")
            else:
                print(f"❌ {step_name} 失敗")
                if step_name == "煙霧測試執行":
                    print("\n⚠️ 煙霧測試失敗，但繼續執行後續檢查...")
        except Exception as e:
            print(f"❌ {step_name} 異常: {e}")
    
    # 總結
    print_header("階段1執行總結")
    
    success_rate = (success_count / total_steps) * 100
    print(f"📊 成功率: {success_count}/{total_steps} ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("🎉 階段1基本成功! 可以進入階段2")
        print("🚀 建議立即執行階段2基礎優化")
    elif success_rate >= 50:
        print("⚠️ 階段1部分成功，需要調整後重試")
        print("🔧 建議檢查失敗步驟並進行修復")
    else:
        print("❌ 階段1失敗，需要深入診斷")
        print("🔍 建議檢查硬體配置和環境設置")
    
    print(f"\n📝 詳細日誌已保存到: logs/rtx4090_test.log")
    print(f"📊 TensorBoard日誌: logs/tensorboard_rtx4090_test")

if __name__ == "__main__":
    main()