#!/usr/bin/env python3
"""
TSE Alpha 整合系統測試腳本
測試除了資料收集模組外的所有模組互動
包括：模型架構、資料載入器、訓練器、交易環境、回測引擎
"""

import sys
import os
import time
import traceback
from pathlib import Path
from datetime import datetime, date
import numpy as np
import torch
import logging

# 設定日誌系統
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('integrated_test_errors.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 錯誤收集器
class ErrorCollector:
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def add_error(self, test_name, error_msg, traceback_str=None):
        error_info = {
            'test': test_name,
            'error': str(error_msg),
            'traceback': traceback_str,
            'timestamp': datetime.now().isoformat()
        }
        self.errors.append(error_info)
        logger.error(f"[{test_name}] {error_msg}")
        if traceback_str:
            logger.error(f"[{test_name}] Traceback: {traceback_str}")
    
    def add_warning(self, test_name, warning_msg):
        warning_info = {
            'test': test_name,
            'warning': str(warning_msg),
            'timestamp': datetime.now().isoformat()
        }
        self.warnings.append(warning_info)
        logger.warning(f"[{test_name}] {warning_msg}")
    
    def save_error_report(self, filename='error_report.txt'):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("TSE Alpha 整合測試錯誤報告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成時間: {datetime.now()}\n\n")
            
            if self.errors:
                f.write("🚨 錯誤列表:\n")
                f.write("-" * 30 + "\n")
                for i, error in enumerate(self.errors, 1):
                    f.write(f"{i}. 測試: {error['test']}\n")
                    f.write(f"   錯誤: {error['error']}\n")
                    f.write(f"   時間: {error['timestamp']}\n")
                    if error['traceback']:
                        f.write(f"   詳細: {error['traceback']}\n")
                    f.write("\n")
            
            if self.warnings:
                f.write("⚠️ 警告列表:\n")
                f.write("-" * 30 + "\n")
                for i, warning in enumerate(self.warnings, 1):
                    f.write(f"{i}. 測試: {warning['test']}\n")
                    f.write(f"   警告: {warning['warning']}\n")
                    f.write(f"   時間: {warning['timestamp']}\n\n")
        
        print(f"📄 錯誤報告已保存至: {filename}")

# 全局錯誤收集器
error_collector = ErrorCollector()

# 添加路徑
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent / "gym_env"))
sys.path.append(str(Path(__file__).parent / "backtest"))

print("=" * 60)
print("🚀 TSE Alpha 整合系統測試")
print("=" * 60)
print(f"測試時間: {datetime.now()}")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print()

# ============================================================================
# 測試1: 模型架構測試
# ============================================================================
def test_model_architecture():
    """測試模型架構模組"""
    print("🧠 測試1: 模型架構模組")
    print("-" * 40)
    
    try:
        from models.model_architecture import TSEAlphaModel, ModelConfig
        
        # 創建模型配置
        config = ModelConfig(
            price_frame_shape=(3, 64, 5),  # (n_stocks, seq_len, features)
            fundamental_dim=10,
            account_dim=4,
            hidden_dim=128,         # 隱藏層維度
            num_heads=8,            # 注意力頭數
            num_layers=4,           # Transformer層數
            dropout=0.1,
            n_stocks=3,             # 測試用3檔股票
            max_position=300        # 最大持倉
        )
        
        print(f"   📋 模型配置: {config}")
        
        # 創建模型
        model = TSEAlphaModel(config)
        print(f"   ✅ 模型創建成功")
        print(f"   📊 模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 測試前向傳播
        batch_size = 2
        test_observation = {
            'price_frame': torch.randn(batch_size, config.n_stocks, 64, 5),
            'fundamental': torch.randn(batch_size, config.fundamental_dim),
            'account': torch.randn(batch_size, config.account_dim)
        }
        
        with torch.no_grad():
            output = model(test_observation)
        
        print(f"   📥 輸入形狀:")
        for key, value in test_observation.items():
            print(f"      {key}: {value.shape}")
        print(f"   📤 輸出形狀:")
        for key, value in output.items():
            print(f"      {key}: {value.shape}")
        print(f"   ✅ 前向傳播成功")
        
        # 測試動作生成
        action = model.get_action(test_observation, deterministic=True)
        print(f"   🎯 動作生成: {action}")
        print(f"   ✅ 動作生成成功")
        
        return True, model, config
        
    except Exception as e:
        error_msg = f"模型架構測試失敗: {e}"
        traceback_str = traceback.format_exc()
        print(f"   ❌ {error_msg}")
        error_collector.add_error("model_architecture", error_msg, traceback_str)
        return False, None, None

# ============================================================================
# 測試2: 資料載入器測試
# ============================================================================
def test_data_loader():
    """測試資料載入器模組"""
    print("\n📊 測試2: 資料載入器模組")
    print("-" * 40)
    
    try:
        from models.data_loader import TSEDataLoader, DataConfig
        
        # 使用小規模股票池進行測試
        test_stocks = ['2330', '2317', '2603']
        print(f"   📈 測試股票: {test_stocks}")
        
        # 創建資料載入器配置 (擴大日期範圍解決資料不足問題)
        config = DataConfig(
            symbols=test_stocks,
            train_start_date='2024-01-01',
            train_end_date='2024-04-30',
            val_start_date='2024-05-01',
            val_end_date='2024-06-30',
            test_start_date='2024-07-01',
            test_end_date='2024-09-30',
            sequence_length=32,  # 減少序列長度要求
            prediction_horizon=3,  # 減少預測視野
            batch_size=4,
            normalize_features=True
        )
        
        # 創建資料載入器
        data_loader = TSEDataLoader(config)
        print(f"   ✅ 資料載入器創建成功")
        
        # 獲取資料載入器
        train_loader, val_loader, test_loader = data_loader.get_dataloaders()
        
        print(f"   📚 訓練集批次數: {len(train_loader)}")
        print(f"   📚 驗證集批次數: {len(val_loader)}")
        print(f"   📚 測試集批次數: {len(test_loader)}")
        
        # 測試一個批次
        for batch_idx, (features, labels) in enumerate(train_loader):
            print(f"   📦 批次 {batch_idx + 1}:")
            print(f"      特徵形狀: {features.shape}")
            print(f"      標籤形狀: {labels.shape}")
            print(f"      特徵範圍: [{features.min():.3f}, {features.max():.3f}]")
            break
        
        print(f"   ✅ 資料載入測試成功")
        
        return True, data_loader, (train_loader, val_loader, test_loader)
        
    except Exception as e:
        error_msg = f"資料載入器測試失敗: {e}"
        traceback_str = traceback.format_exc()
        print(f"   ❌ {error_msg}")
        error_collector.add_error("data_loader", error_msg, traceback_str)
        return False, None, None

# ============================================================================
# 測試3: 訓練器測試
# ============================================================================
def test_trainer(model, config, data_loaders):
    """測試訓練器模組"""
    print("\n🏋️ 測試3: 訓練器模組")
    print("-" * 40)
    
    try:
        from models.trainer import ModelTrainer
        from models.config.training_config import TrainingConfig
        
        if model is None or data_loaders is None:
            warning_msg = "跳過訓練器測試 (依賴模組失敗)"
            print(f"   ⚠️ {warning_msg}")
            error_collector.add_warning("trainer", warning_msg)
            return False, None
        
        train_loader, val_loader, test_loader = data_loaders
        
        # 創建訓練配置
        training_config = TrainingConfig(
            learning_rate=0.001,
            batch_size=4,
            num_epochs=2,  # 快速測試
            patience=10,
            min_delta=0.001,
            save_best_model=False,  # 測試時不保存
            device='cpu'  # 強制使用CPU進行測試
        )
        
        print(f"   ⚙️ 訓練配置: {training_config}")
        
        # 創建訓練器
        trainer = ModelTrainer(model, training_config)
        print(f"   ✅ 訓練器創建成功")
        
        # 執行快速訓練測試
        print(f"   🚀 開始快速訓練測試...")
        start_time = time.time()
        
        results = trainer.train_supervised(
            train_loader, 
            val_loader, 
            test_loader,
            verbose=True
        )
        
        training_time = time.time() - start_time
        print(f"   ⏱️ 訓練時間: {training_time:.2f} 秒")
        print(f"   📈 訓練結果: {results}")
        print(f"   ✅ 訓練器測試成功")
        
        return True, trainer
        
    except Exception as e:
        error_msg = f"訓練器測試失敗: {e}"
        traceback_str = traceback.format_exc()
        print(f"   ❌ {error_msg}")
        error_collector.add_error("trainer", error_msg, traceback_str)
        return False, None

# ============================================================================
# 測試4: 交易環境測試
# ============================================================================
def test_trading_environment():
    """測試交易環境模組"""
    print("\n🏪 測試4: 交易環境模組")
    print("-" * 40)
    
    try:
        from gym_env.env import TSEAlphaEnv
        
        # 創建交易環境
        env = TSEAlphaEnv(
            symbols=['2330', '2317', '2603'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_cash=1000000.0,
            max_holding_days=15,
            max_position_per_stock=300
        )
        
        print(f"   ✅ 交易環境創建成功")
        print(f"   📊 觀測空間: {env.observation_space}")
        print(f"   🎯 動作空間: {env.action_space}")
        
        # 重置環境
        observation, info = env.reset()
        print(f"   🔄 環境重置成功")
        print(f"   💰 初始NAV: {info['nav']:,.2f}")
        print(f"   💵 初始現金: {info['cash']:,.2f}")
        
        # 執行幾步交易
        total_reward = 0
        for step in range(5):
            # 生成隨機動作
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            print(f"   📈 步驟 {step + 1}: 獎勵={reward:.6f}, NAV={info['nav']:,.2f}")
            
            if terminated or truncated:
                print(f"   🏁 環境在第 {step + 1} 步結束")
                break
        
        print(f"   🎯 總獎勵: {total_reward:.6f}")
        print(f"   ✅ 交易環境測試成功")
        
        env.close()
        return True, env
        
    except Exception as e:
        error_msg = f"交易環境測試失敗: {e}"
        traceback_str = traceback.format_exc()
        print(f"   ❌ {error_msg}")
        error_collector.add_error("trading_environment", error_msg, traceback_str)
        return False, None

# ============================================================================
# 測試5: 回測引擎測試
# ============================================================================
def test_backtest_engine():
    """測試回測引擎模組"""
    print("\n⚙️ 測試5: 回測引擎模組")
    print("-" * 40)
    
    try:
        from backtest.engine import BacktestEngine
        from backtest.config import create_smoke_test_config
        
        # 創建回測配置
        config = create_smoke_test_config()
        config.stock_universe = ['2330', '2317']  # 小規模測試
        
        print(f"   ⚙️ 回測配置: 訓練{config.train_window_months}月/測試{config.test_window_months}月")
        
        # 創建回測引擎
        engine = BacktestEngine(config)
        print(f"   ✅ 回測引擎創建成功")
        
        # 創建虛擬模型
        class TestModel:
            def get_action(self, observation, deterministic=True):
                # 簡單的買入策略
                return (0, [10])  # 買入10股第一檔股票
        
        model = TestModel()
        
        # 執行快速回測
        print(f"   🚀 開始快速回測...")
        start_time = time.time()
        
        results = engine.run_backtest(
            model=model,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31)
        )
        
        backtest_time = time.time() - start_time
        print(f"   ⏱️ 回測時間: {backtest_time:.2f} 秒")
        
        # 顯示結果
        if results:
            print(f"   📊 回測結果:")
            # 檢查results是否為BacktestResult物件
            if hasattr(results, 'total_return'):
                print(f"      總收益: {results.total_return:.4f}")
                print(f"      Sharpe比率: {results.sharpe_ratio:.4f}")
                print(f"      最大回撤: {results.max_drawdown:.4f}")
                print(f"      交易次數: {results.total_trades}")
            elif isinstance(results, dict):
                print(f"      總收益: {results.get('total_return', 0):.4f}")
                print(f"      Sharpe比率: {results.get('sharpe_ratio', 0):.4f}")
                print(f"      最大回撤: {results.get('max_drawdown', 0):.4f}")
                print(f"      交易次數: {results.get('total_trades', 0)}")
            else:
                print(f"      結果類型: {type(results)}")
                print(f"      結果內容: {results}")
        
        print(f"   ✅ 回測引擎測試成功")
        
        return True, engine
        
    except Exception as e:
        error_msg = f"回測引擎測試失敗: {e}"
        traceback_str = traceback.format_exc()
        print(f"   ❌ {error_msg}")
        error_collector.add_error("backtest_engine", error_msg, traceback_str)
        return False, None

# ============================================================================
# 測試6: 模組間整合測試
# ============================================================================
def test_module_integration(model, trainer, env):
    """測試模組間整合"""
    print("\n🔗 測試6: 模組間整合測試")
    print("-" * 40)
    
    try:
        if model is None or trainer is None or env is None:
            warning_msg = "跳過整合測試 (依賴模組失敗)"
            print(f"   ⚠️ {warning_msg}")
            error_collector.add_warning("module_integration", warning_msg)
            return False
        
        # 測試模型與環境的整合
        print("   🤝 測試模型與環境整合...")
        
        # 重置環境
        observation, info = env.reset()
        
        # 將觀測轉換為模型輸入格式
        model_observation = {
            'price_frame': torch.FloatTensor(observation['price_frame']).unsqueeze(0),  # 添加batch維度
            'fundamental': torch.FloatTensor(observation['fundamental']).unsqueeze(0),
            'account': torch.FloatTensor(observation['account']).unsqueeze(0)
        }
        
        print(f"   📥 模型輸入形狀:")
        for key, value in model_observation.items():
            print(f"      {key}: {value.shape}")
        
        # 使用模型生成動作
        with torch.no_grad():
            action = model.get_action(model_observation, deterministic=True)
        
        print(f"   🎯 模型生成動作: {action}")
        
        # 在環境中執行動作
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"   📈 執行結果: 獎勵={reward:.6f}, NAV={info['nav']:,.2f}")
        print(f"   ✅ 模型-環境整合成功")
        
        # 測試訓練器與模型的整合
        print("   🤝 測試訓練器與模型整合...")
        
        # 檢查模型是否可以進入訓練模式
        model.train()
        print(f"   📚 模型訓練模式: {model.training}")
        
        model.eval()
        print(f"   🔍 模型評估模式: {not model.training}")
        
        print(f"   ✅ 訓練器-模型整合成功")
        
        return True
        
    except Exception as e:
        error_msg = f"模組整合測試失敗: {e}"
        traceback_str = traceback.format_exc()
        print(f"   ❌ {error_msg}")
        error_collector.add_error("module_integration", error_msg, traceback_str)
        return False

# ============================================================================
# 主測試函數
# ============================================================================
def main():
    """主測試函數"""
    print("開始整合系統測試...\n")
    
    test_results = {}
    start_time = time.time()
    
    # 測試1: 模型架構
    success, model, model_config = test_model_architecture()
    test_results['model_architecture'] = success
    
    # 測試2: 資料載入器
    success, data_loader, data_loaders = test_data_loader()
    test_results['data_loader'] = success
    
    # 測試3: 訓練器
    success, trainer = test_trainer(model, model_config, data_loaders)
    test_results['trainer'] = success
    
    # 測試4: 交易環境
    success, env = test_trading_environment()
    test_results['trading_environment'] = success
    
    # 測試5: 回測引擎
    success, backtest_engine = test_backtest_engine()
    test_results['backtest_engine'] = success
    
    # 測試6: 模組整合
    success = test_module_integration(model, trainer, env)
    test_results['module_integration'] = success
    
    # 總結測試結果
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📋 測試結果總結")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"   {test_name:20s}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n📊 測試統計:")
    print(f"   總測試數: {total_tests}")
    print(f"   通過測試: {passed_tests}")
    print(f"   失敗測試: {total_tests - passed_tests}")
    print(f"   通過率: {passed_tests/total_tests*100:.1f}%")
    print(f"   總耗時: {total_time:.2f} 秒")
    
    if passed_tests == total_tests:
        print(f"\n🎉 所有測試通過！系統整合成功！")
        print(f"✅ TSE Alpha 系統已準備就緒")
    else:
        print(f"\n⚠️ 部分測試失敗，需要進一步調試")
    
    # 保存測試結果
    with open('integrated_test_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"TSE Alpha 整合系統測試結果\n")
        f.write(f"測試時間: {datetime.now()}\n")
        f.write(f"通過率: {passed_tests/total_tests*100:.1f}%\n")
        f.write(f"總耗時: {total_time:.2f} 秒\n\n")
        
        for test_name, result in test_results.items():
            status = "通過" if result else "失敗"
            f.write(f"{test_name}: {status}\n")
    
    print(f"\n📄 測試結果已保存至: integrated_test_result.txt")
    
    # 生成錯誤報告
    if error_collector.errors or error_collector.warnings:
        print(f"\n🚨 發現 {len(error_collector.errors)} 個錯誤和 {len(error_collector.warnings)} 個警告")
        error_collector.save_error_report('integrated_test_error_report.txt')
        
        # 顯示關鍵錯誤摘要
        if error_collector.errors:
            print(f"\n🔥 關鍵錯誤摘要:")
            for i, error in enumerate(error_collector.errors[:3], 1):  # 顯示前3個錯誤
                print(f"   {i}. [{error['test']}] {error['error']}")
            if len(error_collector.errors) > 3:
                print(f"   ... 還有 {len(error_collector.errors) - 3} 個錯誤，詳見錯誤報告")
    else:
        print(f"\n✅ 沒有發現錯誤或警告！")
    
    print(f"\n📊 測試完成統計:")
    print(f"   錯誤數量: {len(error_collector.errors)}")
    print(f"   警告數量: {len(error_collector.warnings)}")
    print(f"   日誌檔案: integrated_test_errors.log")

if __name__ == "__main__":
    main()