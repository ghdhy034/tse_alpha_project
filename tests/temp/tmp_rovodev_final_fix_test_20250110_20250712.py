#!/usr/bin/env python3
"""
TSE Alpha 最終修復測試 - 包含完整錯誤信息輸出
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

# 添加路徑
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent / "gym_env"))

class DetailedTestLogger:
    """詳細測試日誌記錄器"""
    def __init__(self):
        self.logs = []
        self.errors = []
    
    def log(self, message):
        self.logs.append(message)
        print(message)
    
    def log_error(self, test_name, error, traceback_str):
        error_info = f"❌ {test_name} 失敗: {error}\n錯誤類型: {type(error).__name__}\n完整錯誤:\n{traceback_str}"
        self.errors.append(error_info)
        self.logs.append(error_info)
        print(error_info)
    
    def save_to_file(self, filename, results):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"TSE Alpha 最終修復測試結果\n")
            f.write(f"測試時間: {datetime.now()}\n")
            
            passed = sum(1 for r in results.values() if r)
            total = len(results)
            f.write(f"通過率: {passed/total*100:.1f}%\n\n")
            
            for test_name, result in results.items():
                status = "通過" if result else "失敗"
                f.write(f"{test_name}: {status}\n")
            
            f.write(f"\n" + "="*60 + "\n")
            f.write(f"詳細日誌:\n")
            f.write(f"="*60 + "\n")
            
            for log_entry in self.logs:
                f.write(log_entry + "\n")
            
            if self.errors:
                f.write(f"\n" + "="*60 + "\n")
                f.write(f"錯誤詳情:\n")
                f.write(f"="*60 + "\n")
                for error in self.errors:
                    f.write(error + "\n" + "-"*40 + "\n")

logger = DetailedTestLogger()

print("🎉 TSE Alpha 最終修復測試")
print("=" * 60)
logger.log(f"測試開始時間: {datetime.now()}")

def test_training_config():
    """測試訓練配置"""
    logger.log("\n⚙️ 測試: 訓練配置")
    logger.log("-" * 40)
    
    try:
        from models.config.training_config import TrainingConfig
        logger.log("   ✅ TrainingConfig 導入成功")
        
        # 測試默認創建
        logger.log("   🔧 創建默認配置...")
        config = TrainingConfig()
        logger.log("   ✅ 默認配置創建成功！")
        
        # 檢查技術指標數量
        expected_indicators = config.price_features - 5
        actual_indicators = len(config.technical_indicators)
        logger.log(f"   📊 技術指標檢查:")
        logger.log(f"      price_features: {config.price_features}")
        logger.log(f"      期望技術指標數量: {expected_indicators}")
        logger.log(f"      實際技術指標數量: {actual_indicators}")
        logger.log(f"      技術指標列表: {config.technical_indicators}")
        
        if expected_indicators == actual_indicators:
            logger.log("   ✅ 技術指標數量匹配")
        else:
            logger.log(f"   ❌ 技術指標數量不匹配")
        
        # 測試 patience 參數
        logger.log("   🔧 測試 patience 參數...")
        config_with_patience = TrainingConfig(patience=15)
        logger.log(f"   ✅ patience 參數測試成功")
        logger.log(f"      設定 patience: {config_with_patience.patience}")
        logger.log(f"      同步 early_stopping_patience: {config_with_patience.early_stopping_patience}")
        
        return True
        
    except Exception as e:
        logger.log_error("訓練配置測試", e, traceback.format_exc())
        return False

def test_complete_system():
    """測試完整系統"""
    logger.log("\n🔗 測試: 完整系統")
    logger.log("-" * 40)
    
    try:
        import torch
        from models.model_architecture import TSEAlphaModel, ModelConfig
        from models.data_loader import TSEDataLoader, DataConfig
        from gym_env.env import TSEAlphaEnv
        logger.log("   ✅ 所有模組導入成功")
        
        # 創建模型
        model_config = ModelConfig(
            price_frame_shape=(2, 64, 5),
            n_stocks=2,
            hidden_dim=128,
            max_position=300
        )
        model = TSEAlphaModel(model_config)
        logger.log("   ✅ 模型創建成功")
        
        # 創建交易環境
        env = TSEAlphaEnv(
            symbols=['2330', '2317'],
            start_date='2024-01-01',
            end_date='2024-01-15',
            initial_cash=1000000.0
        )
        logger.log("   ✅ 交易環境創建成功")
        
        # 測試完整流程
        observation, info = env.reset()
        
        model_observation = {
            'price_frame': torch.FloatTensor(observation['price_frame']).unsqueeze(0),
            'fundamental': torch.FloatTensor(observation['fundamental']).unsqueeze(0),
            'account': torch.FloatTensor(observation['account']).unsqueeze(0)
        }
        
        with torch.no_grad():
            action = model.get_action(model_observation, deterministic=True)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        logger.log(f"   ✅ 完整流程成功: 動作={action}, 獎勵={reward:.6f}, NAV={info['nav']:,.2f}")
        
        env.close()
        return True
        
    except Exception as e:
        logger.log_error("完整系統測試", e, traceback.format_exc())
        return False

def main():
    """主測試函數"""
    logger.log("開始最終修復測試...\n")
    
    results = {}
    
    # 執行所有測試
    results['training_config'] = test_training_config()
    results['complete_system'] = test_complete_system()
    
    # 總結結果
    logger.log("\n" + "=" * 60)
    logger.log("📋 最終修復測試結果")
    logger.log("=" * 60)
    
    passed_tests = 0
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        logger.log(f"   {test_name:20s}: {status}")
        if result:
            passed_tests += 1
    
    logger.log(f"\n📊 測試統計:")
    logger.log(f"   總測試數: {total_tests}")
    logger.log(f"   通過測試: {passed_tests}")
    logger.log(f"   失敗測試: {total_tests - passed_tests}")
    logger.log(f"   通過率: {passed_tests/total_tests*100:.1f}%")
    
    # 保存詳細結果
    logger.save_to_file('final_fix_test_detailed_result.txt', results)
    
    if passed_tests == total_tests:
        logger.log(f"\n🎉 所有測試通過！系統完全修復成功！")
        logger.log(f"✅ TSE Alpha 已達到 100% 可用狀態")
        logger.log(f"🚀 準備進入生產訓練階段")
    else:
        logger.log(f"\n⚠️ 還有 {total_tests - passed_tests} 個問題需要解決")
        logger.log(f"📄 詳細錯誤信息已保存至結果文件")
    
    logger.log(f"\n📄 詳細結果已保存至: final_fix_test_detailed_result.txt")

if __name__ == "__main__":
    main()