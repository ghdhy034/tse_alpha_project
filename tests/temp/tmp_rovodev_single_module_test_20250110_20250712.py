#!/usr/bin/env python3
"""
TSE Alpha 單一模組測試腳本
可以單獨測試各個模組的功能
"""

import sys
import os
import time
import argparse
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
        logging.FileHandler('single_module_test_errors.log', encoding='utf-8')
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
    
    def save_error_report(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("TSE Alpha 單一模組測試錯誤報告\n")
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

def test_model_only():
    """僅測試模型架構"""
    print("🧠 測試模型架構模組")
    print("-" * 40)
    
    try:
        from models.model_architecture import TSEAlphaModel, ModelConfig
        
        # 創建不同規模的配置進行測試
        configs = [
            ("小型模型", ModelConfig(input_dim=71, hidden_dim=64, num_heads=4, num_layers=2)),
            ("中型模型", ModelConfig(input_dim=71, hidden_dim=128, num_heads=8, num_layers=4)),
            ("大型模型", ModelConfig(input_dim=71, hidden_dim=256, num_heads=16, num_layers=6))
        ]
        
        for name, config in configs:
            print(f"\n   📋 測試 {name}:")
            model = TSEAlphaModel(config)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"      參數數量: {param_count:,}")
            
            # 測試前向傳播
            test_input = torch.randn(1, config.num_stocks, config.sequence_length, config.input_dim)
            with torch.no_grad():
                output = model(test_input)
                action = model.get_action(test_input, deterministic=True)
            
            print(f"      輸入形狀: {test_input.shape}")
            print(f"      輸出形狀: {output.shape}")
            print(f"      動作: {action}")
            print(f"      ✅ {name} 測試通過")
        
        return True
        
    except Exception as e:
        error_msg = f"模型測試失敗: {e}"
        traceback_str = traceback.format_exc()
        print(f"   ❌ {error_msg}")
        error_collector.add_error("model_architecture", error_msg, traceback_str)
        return False

def test_data_loader_only():
    """僅測試資料載入器"""
    print("📊 測試資料載入器模組")
    print("-" * 40)
    
    try:
        from models.data_loader import TSEDataLoader, DataConfig
        
        # 測試不同的配置
        test_configs = [
            {
                'name': '小規模測試',
                'stocks': ['2330', '2317'],
                'sequence_length': 20,
                'batch_size': 2,
                'train_start_date': '2024-01-01',
                'train_end_date': '2024-03-31',
                'val_start_date': '2024-04-01',
                'val_end_date': '2024-05-31',
                'test_start_date': '2024-06-01',
                'test_end_date': '2024-08-31'
            },
            {
                'name': '中規模測試',
                'stocks': ['2330', '2317', '2603'],
                'sequence_length': 32,
                'batch_size': 4,
                'train_start_date': '2024-01-01',
                'train_end_date': '2024-04-30',
                'val_start_date': '2024-05-01',
                'val_end_date': '2024-06-30',
                'test_start_date': '2024-07-01',
                'test_end_date': '2024-09-30'
            }
        ]
        
        for test_config in test_configs:
            print(f"\n   📋 {test_config['name']}:")
            
            config = DataConfig(
                symbols=test_config['stocks'],
                train_start_date=test_config['train_start_date'],
                train_end_date=test_config['train_end_date'],
                val_start_date=test_config['val_start_date'],
                val_end_date=test_config['val_end_date'],
                test_start_date=test_config['test_start_date'],
                test_end_date=test_config['test_end_date'],
                sequence_length=test_config['sequence_length'],
                batch_size=test_config['batch_size'],
                prediction_horizon=3,
                normalize_features=True
            )
            
            data_loader = TSEDataLoader(config)
            train_loader, val_loader, test_loader = data_loader.get_dataloaders()
            
            print(f"      股票: {test_config['stocks']}")
            print(f"      訓練批次: {len(train_loader)}")
            print(f"      驗證批次: {len(val_loader)}")
            print(f"      測試批次: {len(test_loader)}")
            
            # 測試一個批次
            if len(train_loader) > 0:
                for features, labels in train_loader:
                    print(f"      特徵形狀: {features.shape}")
                    print(f"      標籤形狀: {labels.shape}")
                    break
            
            print(f"      ✅ {test_config['name']} 通過")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 資料載入器測試失敗: {e}")
        traceback.print_exc()
        return False

def test_trainer_only():
    """僅測試訓練器"""
    print("🏋️ 測試訓練器模組")
    print("-" * 40)
    
    try:
        from models.trainer import ModelTrainer
        from models.model_architecture import TSEAlphaModel, ModelConfig
        from models.config.training_config import TrainingConfig
        
        # 創建簡單模型
        model_config = ModelConfig(input_dim=71, hidden_dim=64, num_heads=4, num_layers=2)
        model = TSEAlphaModel(model_config)
        
        # 創建訓練配置
        training_config = TrainingConfig(
            learning_rate=0.001,
            batch_size=2,
            num_epochs=1,  # 快速測試
            patience=5,
            device='cpu'
        )
        
        # 創建訓練器
        trainer = ModelTrainer(model, training_config)
        print(f"   ✅ 訓練器創建成功")
        
        # 創建虛擬資料
        batch_size = 2
        num_stocks = model_config.num_stocks
        seq_len = model_config.sequence_length
        input_dim = model_config.input_dim
        
        # 虛擬訓練資料
        train_data = []
        for i in range(5):  # 5個批次
            features = torch.randn(batch_size, num_stocks, seq_len, input_dim)
            labels = torch.randint(0, 3, (batch_size, num_stocks))  # 3個動作類別
            train_data.append((features, labels))
        
        val_data = train_data[:2]  # 使用部分資料作為驗證
        test_data = train_data[:1]  # 使用部分資料作為測試
        
        print(f"   📚 虛擬資料準備完成")
        print(f"   🚀 開始訓練測試...")
        
        # 執行訓練
        results = trainer.train_supervised(train_data, val_data, test_data, verbose=True)
        
        print(f"   📈 訓練結果: {results}")
        print(f"   ✅ 訓練器測試通過")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 訓練器測試失敗: {e}")
        traceback.print_exc()
        return False

def test_env_only():
    """僅測試交易環境"""
    print("🏪 測試交易環境模組")
    print("-" * 40)
    
    try:
        from gym_env.env import TSEAlphaEnv
        
        # 測試不同的環境配置
        env_configs = [
            {
                'name': '基本配置',
                'symbols': ['2330', '2317'],
                'start_date': '2024-01-01',
                'end_date': '2024-01-15',
                'initial_cash': 500000.0
            },
            {
                'name': '多股票配置',
                'symbols': ['2330', '2317', '2603'],
                'start_date': '2024-01-01',
                'end_date': '2024-01-31',
                'initial_cash': 1000000.0
            }
        ]
        
        for env_config in env_configs:
            print(f"\n   📋 {env_config['name']}:")
            
            env = TSEAlphaEnv(
                symbols=env_config['symbols'],
                start_date=env_config['start_date'],
                end_date=env_config['end_date'],
                initial_cash=env_config['initial_cash']
            )
            
            print(f"      股票: {env_config['symbols']}")
            print(f"      觀測空間: {env.observation_space}")
            print(f"      動作空間: {env.action_space}")
            
            # 重置並運行幾步
            observation, info = env.reset()
            print(f"      初始NAV: {info['nav']:,.2f}")
            
            total_reward = 0
            for step in range(3):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            print(f"      總獎勵: {total_reward:.6f}")
            print(f"      最終NAV: {info['nav']:,.2f}")
            print(f"      ✅ {env_config['name']} 通過")
            
            env.close()
        
        return True
        
    except Exception as e:
        print(f"   ❌ 環境測試失敗: {e}")
        traceback.print_exc()
        return False

def test_backtest_only():
    """僅測試回測引擎"""
    print("⚙️ 測試回測引擎模組")
    print("-" * 40)
    
    try:
        from backtest.engine import BacktestEngine
        from backtest.config import create_smoke_test_config, BacktestConfig
        
        # 測試不同的回測配置
        configs = [
            ("快速測試", create_smoke_test_config()),
            ("自定義配置", BacktestConfig(
                train_window_months=2,
                test_window_months=1,
                stock_universe=['2330', '2317'],
                backend='seq'
            ))
        ]
        
        for name, config in configs:
            print(f"\n   📋 {name}:")
            
            engine = BacktestEngine(config)
            print(f"      後端: {config.backend}")
            print(f"      股票池: {len(config.stock_universe)} 檔")
            
            # 創建簡單策略
            class SimpleStrategy:
                def get_action(self, observation, deterministic=True):
                    return (0, [5])  # 總是買入5股第一檔股票
            
            model = SimpleStrategy()
            
            # 執行回測
            start_time = time.time()
            results = engine.run_backtest(
                model=model,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 2, 29)
            )
            execution_time = time.time() - start_time
            
            print(f"      執行時間: {execution_time:.2f} 秒")
            if results:
                print(f"      總收益: {results.get('total_return', 0):.4f}")
                print(f"      交易次數: {results.get('total_trades', 0)}")
            
            print(f"      ✅ {name} 通過")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 回測引擎測試失敗: {e}")
        traceback.print_exc()
        return False

def test_features_only():
    """僅測試特徵工程"""
    print("🔧 測試特徵工程模組")
    print("-" * 40)
    
    try:
        from data_pipeline.features import FeatureEngine
        
        # 創建特徵引擎
        feature_engine = FeatureEngine()
        print(f"   ✅ 特徵引擎創建成功")
        
        # 測試特徵計算
        test_stocks = ['2330', '2317']
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        
        print(f"   📊 測試股票: {test_stocks}")
        print(f"   📅 時間範圍: {start_date} ~ {end_date}")
        
        for stock in test_stocks:
            print(f"\n   📈 處理股票 {stock}:")
            
            # 計算技術指標特徵
            tech_features = feature_engine.calculate_technical_features(
                stock, start_date, end_date
            )
            
            if tech_features is not None and not tech_features.empty:
                print(f"      技術特徵: {tech_features.shape[1]} 個特徵, {tech_features.shape[0]} 筆記錄")
                print(f"      特徵名稱: {list(tech_features.columns)[:5]}...")  # 顯示前5個
            else:
                print(f"      ⚠️ 技術特徵資料為空")
            
            # 計算籌碼面特徵
            chip_features = feature_engine.calculate_chip_features(
                stock, start_date, end_date
            )
            
            if chip_features is not None and not chip_features.empty:
                print(f"      籌碼特徵: {chip_features.shape[1]} 個特徵, {chip_features.shape[0]} 筆記錄")
            else:
                print(f"      ⚠️ 籌碼特徵資料為空")
            
            print(f"      ✅ 股票 {stock} 特徵計算完成")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 特徵工程測試失敗: {e}")
        traceback.print_exc()
        return False

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='TSE Alpha 單一模組測試')
    parser.add_argument('--module', type=str, choices=[
        'model', 'data', 'trainer', 'env', 'backtest', 'features', 'all'
    ], default='all', help='選擇要測試的模組')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 TSE Alpha 單一模組測試")
    print("=" * 60)
    print(f"測試時間: {datetime.now()}")
    print(f"測試模組: {args.module}")
    print()
    
    test_functions = {
        'model': test_model_only,
        'data': test_data_loader_only,
        'trainer': test_trainer_only,
        'env': test_env_only,
        'backtest': test_backtest_only,
        'features': test_features_only
    }
    
    start_time = time.time()
    results = {}
    
    if args.module == 'all':
        # 測試所有模組
        for module_name, test_func in test_functions.items():
            print(f"\n{'='*20} {module_name.upper()} {'='*20}")
            results[module_name] = test_func()
    else:
        # 測試指定模組
        if args.module in test_functions:
            results[args.module] = test_functions[args.module]()
        else:
            print(f"❌ 未知模組: {args.module}")
            return
    
    # 總結結果
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📋 測試結果總結")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for module, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"   {module:15s}: {status}")
    
    print(f"\n📊 統計:")
    print(f"   通過: {passed}/{total}")
    print(f"   通過率: {passed/total*100:.1f}%")
    print(f"   耗時: {total_time:.2f} 秒")
    
    # 保存結果
    result_file = f"single_module_test_{args.module}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"TSE Alpha 單一模組測試結果\n")
        f.write(f"測試模組: {args.module}\n")
        f.write(f"測試時間: {datetime.now()}\n")
        f.write(f"通過率: {passed/total*100:.1f}%\n\n")
        
        for module, result in results.items():
            status = "通過" if result else "失敗"
            f.write(f"{module}: {status}\n")
    
    print(f"\n📄 結果已保存至: {result_file}")
    
    # 生成錯誤報告
    if error_collector.errors or error_collector.warnings:
        print(f"\n🚨 發現 {len(error_collector.errors)} 個錯誤和 {len(error_collector.warnings)} 個警告")
        error_report_file = f"single_module_error_report_{args.module}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        error_collector.save_error_report(error_report_file)
        
        # 顯示關鍵錯誤摘要
        if error_collector.errors:
            print(f"\n🔥 關鍵錯誤摘要:")
            for i, error in enumerate(error_collector.errors[:3], 1):
                print(f"   {i}. [{error['test']}] {error['error']}")
    else:
        print(f"\n✅ 沒有發現錯誤或警告！")

if __name__ == "__main__":
    main()