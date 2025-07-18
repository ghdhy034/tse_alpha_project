#!/usr/bin/env python3
"""
TSE Alpha 綜合調試測試腳本
包含詳細錯誤信息輸出到文件
"""

import sys
import traceback
import io
from pathlib import Path
from datetime import datetime

# 添加路徑
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent / "gym_env"))

class DetailedLogger:
    """詳細日誌記錄器"""
    def __init__(self, filename="comprehensive_debug_result.txt"):
        self.filename = filename
        self.logs = []
        self.console_output = []
    
    def log(self, message, level="INFO"):
        """記錄日誌"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.logs.append(log_entry)
        self.console_output.append(message)
        print(message)
    
    def log_error(self, test_name, error, traceback_str):
        """記錄錯誤"""
        self.log(f"❌ {test_name} 失敗: {error}", "ERROR")
        self.log(f"🔍 錯誤類型: {type(error).__name__}", "ERROR")
        self.log(f"📍 錯誤模組: {error.__class__.__module__}", "ERROR")
        self.log(f"📝 完整錯誤追蹤:", "ERROR")
        for line in traceback_str.split('\n'):
            if line.strip():
                self.log(f"    {line}", "ERROR")
    
    def log_success(self, test_name, details=None):
        """記錄成功"""
        self.log(f"✅ {test_name} 成功", "SUCCESS")
        if details:
            for detail in details:
                self.log(f"   {detail}", "SUCCESS")
    
    def save_to_file(self):
        """保存到文件"""
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(f"TSE Alpha 綜合調試測試結果\n")
            f.write(f"測試時間: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            
            for log_entry in self.logs:
                f.write(log_entry + "\n")
        
        print(f"\n📄 詳細結果已保存至: {self.filename}")

# 全局日誌器
logger = DetailedLogger()

print("🔧 TSE Alpha 綜合調試測試")
print("=" * 60)
logger.log(f"測試開始時間: {datetime.now()}")

def test_training_config():
    """測試訓練配置"""
    logger.log("\n⚙️ 測試: 訓練配置")
    logger.log("-" * 40)
    
    try:
        # 測試導入
        logger.log("🔧 嘗試導入 TrainingConfig...")
        from models.config.training_config import TrainingConfig
        logger.log("✅ TrainingConfig 導入成功")
        
        # 檢查類信息
        logger.log(f"🔍 TrainingConfig 類信息:")
        logger.log(f"   模組路徑: {TrainingConfig.__module__}")
        logger.log(f"   類名稱: {TrainingConfig.__name__}")
        
        # 檢查 __init__ 方法簽名
        import inspect
        sig = inspect.signature(TrainingConfig.__init__)
        params = list(sig.parameters.keys())
        logger.log(f"   __init__ 參數: {params}")
        
        # 檢查類屬性
        annotations = getattr(TrainingConfig, '__annotations__', {})
        logger.log(f"   類屬性數量: {len(annotations)}")
        if 'patience' in annotations:
            logger.log(f"   ✅ patience 屬性存在")
        else:
            logger.log(f"   ❌ patience 屬性不存在")
        
        # 測試默認創建
        logger.log("🔧 嘗試創建默認配置...")
        config1 = TrainingConfig()
        logger.log("✅ 默認配置創建成功")
        logger.log(f"   patience: {getattr(config1, 'patience', 'NOT_FOUND')}")
        logger.log(f"   early_stopping_patience: {getattr(config1, 'early_stopping_patience', 'NOT_FOUND')}")
        
        # 測試帶 patience 參數創建
        logger.log("🔧 嘗試創建帶 patience 參數的配置...")
        config2 = TrainingConfig(patience=7)
        logger.log("✅ 帶 patience 參數創建成功")
        logger.log(f"   設定 patience: {config2.patience}")
        logger.log(f"   同步 early_stopping_patience: {config2.early_stopping_patience}")
        
        # 測試完整參數創建
        logger.log("🔧 嘗試創建完整參數配置...")
        config3 = TrainingConfig(
            learning_rate=0.001,
            batch_size=2,
            num_epochs=1,
            patience=5,
            device='cpu'
        )
        logger.log("✅ 完整參數配置創建成功")
        logger.log(f"   patience: {config3.patience}")
        logger.log(f"   learning_rate: {config3.learning_rate}")
        logger.log(f"   batch_size: {config3.batch_size}")
        logger.log(f"   device: {config3.device}")
        
        # 驗證參數
        assert config3.patience == 5, f"patience 應該是 5，實際是 {config3.patience}"
        assert config3.early_stopping_patience == 5, f"early_stopping_patience 應該是 5，實際是 {config3.early_stopping_patience}"
        logger.log("✅ 所有參數驗證通過")
        
        logger.log_success("訓練配置測試", [
            f"默認配置創建: 成功",
            f"帶參數創建: 成功", 
            f"完整配置創建: 成功",
            f"參數驗證: 通過"
        ])
        return True
        
    except Exception as e:
        logger.log_error("訓練配置測試", e, traceback.format_exc())
        
        # 額外調試信息
        try:
            logger.log("🔧 額外調試信息:")
            from models.config.training_config import TrainingConfig
            logger.log(f"   TrainingConfig 可導入: True")
            
            # 嘗試檢查文件內容
            config_file = Path("models/config/training_config.py")
            if config_file.exists():
                logger.log(f"   配置文件存在: True")
                logger.log(f"   文件大小: {config_file.stat().st_size} bytes")
                
                # 檢查文件中是否包含 patience
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    has_patience = 'patience:' in content
                    logger.log(f"   文件包含 patience: {has_patience}")
            else:
                logger.log(f"   配置文件存在: False")
                
        except Exception as debug_e:
            logger.log(f"🚨 調試信息獲取失敗: {debug_e}")
        
        return False

def test_data_loader():
    """測試資料載入器"""
    logger.log("\n📊 測試: 資料載入器")
    logger.log("-" * 40)
    
    try:
        # 檢查資料庫文件
        logger.log("🔧 檢查資料庫文件...")
        db_path = Path("market_data_collector/data/stock_data.db")
        logger.log(f"   資料庫文件存在: {db_path.exists()}")
        if db_path.exists():
            size_mb = db_path.stat().st_size / 1024 / 1024
            logger.log(f"   資料庫文件大小: {size_mb:.1f} MB")
        
        # 測試導入
        logger.log("🔧 嘗試導入資料載入器模組...")
        from models.data_loader import TSEDataLoader, DataConfig
        logger.log("✅ 資料載入器模組導入成功")
        
        # 檢查特徵工程模組
        logger.log("🔧 檢查特徵工程模組...")
        try:
            from data_pipeline.features import FeatureEngine
            logger.log("✅ FeatureEngine 可導入")
        except Exception as fe_e:
            logger.log(f"❌ FeatureEngine 導入失敗: {fe_e}")
        
        # 創建配置
        logger.log("🔧 創建資料配置...")
        config = DataConfig(
            symbols=['2330', '2317'],
            train_start_date='2024-01-01',
            train_end_date='2024-04-30',
            val_start_date='2024-05-01',
            val_end_date='2024-06-30',
            test_start_date='2024-07-01',
            test_end_date='2024-09-30',
            sequence_length=20,
            prediction_horizon=3,
            batch_size=2,
            normalize_features=True
        )
        logger.log("✅ 資料配置創建成功")
        logger.log(f"   配置股票: {config.symbols}")
        logger.log(f"   序列長度: {config.sequence_length}")
        logger.log(f"   批次大小: {config.batch_size}")
        
        # 創建資料載入器
        logger.log("🔧 創建 TSEDataLoader...")
        data_loader = TSEDataLoader(config)
        logger.log("✅ TSEDataLoader 創建成功")
        
        # 獲取資料載入器
        logger.log("🔧 獲取 DataLoaders...")
        train_loader, val_loader, test_loader = data_loader.get_dataloaders()
        logger.log("✅ DataLoaders 獲取成功")
        logger.log(f"   訓練批次: {len(train_loader)}")
        logger.log(f"   驗證批次: {len(val_loader)}")
        logger.log(f"   測試批次: {len(test_loader)}")
        
        # 測試批次載入
        if len(train_loader) > 0:
            logger.log("🔧 測試第一個批次載入...")
            for batch_idx, batch in enumerate(train_loader):
                logger.log(f"✅ 批次 {batch_idx + 1} 載入成功")
                logger.log(f"   觀測形狀:")
                for key, value in batch['observation'].items():
                    logger.log(f"      {key}: {value.shape}")
                logger.log(f"   標籤形狀: {batch['labels'].shape}")
                logger.log(f"   元資料類型: {type(batch['metadata'])}")
                break
            logger.log("✅ 批次載入測試成功")
        else:
            logger.log("⚠️ 訓練資料為空")
            return False
        
        logger.log_success("資料載入器測試", [
            f"模組導入: 成功",
            f"配置創建: 成功",
            f"載入器創建: 成功",
            f"批次載入: 成功"
        ])
        return True
        
    except Exception as e:
        logger.log_error("資料載入器測試", e, traceback.format_exc())
        return False

def test_model_env_integration():
    """測試模型環境整合"""
    logger.log("\n🔗 測試: 模型環境整合")
    logger.log("-" * 40)
    
    try:
        # 檢查系統環境
        logger.log("🔧 檢查系統環境...")
        import torch
        logger.log(f"   PyTorch 版本: {torch.__version__}")
        logger.log(f"   CUDA 可用: {torch.cuda.is_available()}")
        
        # 測試模組導入
        logger.log("🔧 導入模組...")
        from models.model_architecture import TSEAlphaModel, ModelConfig
        from gym_env.env import TSEAlphaEnv
        logger.log("✅ 所有模組導入成功")
        
        # 創建模型
        logger.log("🔧 創建模型...")
        model_config = ModelConfig(
            price_frame_shape=(2, 20, 5),
            n_stocks=2,
            hidden_dim=128,
            max_position=300
        )
        model = TSEAlphaModel(model_config)
        logger.log("✅ 模型創建成功")
        logger.log(f"   模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 創建環境
        logger.log("🔧 創建交易環境...")
        env = TSEAlphaEnv(
            symbols=['2330', '2317'],
            start_date='2024-01-01',
            end_date='2024-01-15',
            initial_cash=1000000.0
        )
        logger.log("✅ 交易環境創建成功")
        
        # 重置環境
        logger.log("🔧 重置環境...")
        observation, info = env.reset()
        logger.log("✅ 環境重置成功")
        logger.log(f"   初始 NAV: {info['nav']:,.2f}")
        logger.log(f"   初始現金: {info['cash']:,.2f}")
        
        # 檢查觀測
        logger.log("🔧 檢查觀測...")
        for key, value in observation.items():
            logger.log(f"   {key}: {value.shape}")
        
        # 準備模型輸入
        logger.log("🔧 準備模型輸入...")
        model_observation = {
            'price_frame': torch.FloatTensor(observation['price_frame']).unsqueeze(0),
            'fundamental': torch.FloatTensor(observation['fundamental']).unsqueeze(0),
            'account': torch.FloatTensor(observation['account']).unsqueeze(0)
        }
        logger.log("✅ 模型輸入準備成功")
        
        # 模型前向傳播
        logger.log("🔧 執行模型前向傳播...")
        with torch.no_grad():
            outputs = model(model_observation)
        logger.log("✅ 模型前向傳播成功")
        
        # 動作生成
        logger.log("🔧 生成動作...")
        action = model.get_action(model_observation, deterministic=True)
        logger.log(f"✅ 動作生成成功: {action}")
        
        # 環境執行
        logger.log("🔧 執行環境步驟...")
        next_observation, reward, terminated, truncated, info = env.step(action)
        logger.log("✅ 環境執行成功")
        logger.log(f"   獎勵: {reward:.6f}")
        logger.log(f"   NAV: {info['nav']:,.2f}")
        
        env.close()
        
        logger.log_success("模型環境整合測試", [
            f"模組導入: 成功",
            f"模型創建: 成功",
            f"環境創建: 成功",
            f"前向傳播: 成功",
            f"動作生成: 成功",
            f"環境執行: 成功"
        ])
        return True
        
    except Exception as e:
        logger.log_error("模型環境整合測試", e, traceback.format_exc())
        return False

def main():
    """主測試函數"""
    logger.log("開始綜合調試測試...\n")
    
    results = {}
    
    # 執行所有測試
    results['training_config'] = test_training_config()
    results['data_loader'] = test_data_loader()
    results['model_env_integration'] = test_model_env_integration()
    
    # 總結結果
    logger.log("\n" + "=" * 60)
    logger.log("📋 綜合測試結果總結")
    logger.log("=" * 60)
    
    passed_tests = 0
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        logger.log(f"   {test_name:25s}: {status}")
        if result:
            passed_tests += 1
    
    logger.log(f"\n📊 測試統計:")
    logger.log(f"   總測試數: {total_tests}")
    logger.log(f"   通過測試: {passed_tests}")
    logger.log(f"   失敗測試: {total_tests - passed_tests}")
    logger.log(f"   通過率: {passed_tests/total_tests*100:.1f}%")
    
    # 保存詳細結果
    logger.save_to_file()
    
    # 同時保存簡化結果
    with open('comprehensive_test_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"TSE Alpha 綜合測試結果摘要\n")
        f.write(f"測試時間: {datetime.now()}\n")
        f.write(f"通過率: {passed_tests/total_tests*100:.1f}%\n\n")
        
        for test_name, result in results.items():
            status = "通過" if result else "失敗"
            f.write(f"{test_name}: {status}\n")
        
        f.write(f"\n詳細信息請查看: comprehensive_debug_result.txt\n")
    
    if passed_tests == total_tests:
        logger.log(f"\n🎉 所有測試通過！系統修復成功！")
    else:
        logger.log(f"\n⚠️ 還有 {total_tests - passed_tests} 個問題需要解決")
        logger.log(f"📄 詳細錯誤信息已保存，請檢查 comprehensive_debug_result.txt")

if __name__ == "__main__":
    main()