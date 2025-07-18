#!/usr/bin/env python3
"""
TSE Alpha 完整訓練腳本 - RTX 4090 專用
高配置大規模訓練
"""
import sys
import os
import time
import argparse
from pathlib import Path
import torch
import logging
from datetime import datetime

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))

# 強制使用生產配置
os.environ['TSE_ALPHA_MODE'] = 'production'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_rtx4090_environment():
    """檢查 RTX 4090 環境"""
    logger.info("🔍 檢查 RTX 4090 環境...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA 不可用")
        return False
    
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_memory = gpu_props.total_memory / 1e9
    
    logger.info(f"GPU: {gpu_props.name}")
    logger.info(f"VRAM: {gpu_memory:.1f}GB")
    logger.info(f"計算能力: {gpu_props.major}.{gpu_props.minor}")
    
    # 檢查是否為 RTX 4090 或高階 GPU
    if '4090' in gpu_props.name or gpu_memory > 20:
        logger.info("✅ 檢測到 RTX 4090 或同等級 GPU，使用高配置模式")
        return True
    else:
        logger.warning(f"⚠️  非 RTX 4090 ({gpu_props.name}, {gpu_memory:.1f}GB)")
        logger.warning("⚠️  建議使用 RTX 4090 進行完整訓練")
        
        # 詢問是否繼續
        response = input("是否繼續使用當前 GPU？(y/N): ")
        return response.lower() == 'y'

def create_production_config(args):
    """創建生產環境配置"""
    from configs.hardware_configs import ConfigManager
    
    # 獲取 RTX 4090 配置
    hw_config = ConfigManager.get_config('rtx4090')
    
    # 根據命令行參數調整
    config = {
        # 硬體配置
        'hardware_profile': hw_config.name,
        'batch_size': args.batch_size or hw_config.batch_size,
        'accumulate_grad_batches': hw_config.accumulate_grad_batches,
        'precision': hw_config.precision,
        'num_workers': hw_config.num_workers,
        'pin_memory': hw_config.pin_memory,
        
        # 模型配置
        'sequence_length': hw_config.seq_len,
        'n_stocks': args.n_stocks or hw_config.n_stocks,
        
        # 訓練配置
        'num_epochs': args.epochs or hw_config.num_epochs,
        'data_subset_ratio': args.data_ratio or hw_config.data_subset_ratio,
        'learning_rate': args.lr or (1e-4 * hw_config.batch_size / 64),
        
        # 高階配置
        'early_stopping_patience': 30,
        'save_top_k': 5,
        'log_every_n_steps': 50,
        'val_check_interval': 0.25,  # 每 25% epoch 驗證一次
        
        # Optuna 配置 (如果啟用)
        'optuna_trials': args.optuna_trials or 100,
        'optuna_timeout': args.optuna_timeout or 3600 * 24,  # 24小時
        
        # 檢查點配置
        'checkpoint_every_n_epochs': 10,
        'auto_resume': True,
        
        # 資料配置
        'train_start_date': '2020-03-02',
        'train_end_date': '2023-12-31',
        'val_start_date': '2024-01-01',
        'val_end_date': '2024-06-30',
        'test_start_date': '2024-07-01',
        'test_end_date': '2024-12-31'
    }
    
    logger.info("生產配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    return config

def setup_training_environment(config):
    """設置訓練環境"""
    logger.info("🔧 設置訓練環境...")
    
    try:
        # 導入必要模組
        from models.config.training_config import TrainingConfig
        from models.model_architecture import TSEAlphaModel, ModelConfig
        from models.trainer import ModelTrainer
        from models.data_loader import TSEDataLoader, DataConfig
        from stock_config import TRAIN_STOCKS, VALIDATION_STOCKS, TEST_STOCKS
        
        # 創建訓練配置
        training_config = TrainingConfig()
        training_config.batch_size = config['batch_size']
        training_config.num_epochs = config['num_epochs']
        training_config.learning_rate = config['learning_rate']
        training_config.early_stopping_patience = config['early_stopping_patience']
        
        # 創建模型配置
        model_config = ModelConfig(
            price_frame_shape=(config['n_stocks'], config['sequence_length'], 27),
            fundamental_dim=43,  # 完整基本面特徵
            n_stocks=config['n_stocks'],
            hidden_dim=512,      # 高配置
            num_heads=16,        # 更多注意力頭
            num_layers=6,        # 更深層數
            dropout=0.1
        )
        
        # 創建資料配置
        symbols = TRAIN_STOCKS + VALIDATION_STOCKS + TEST_STOCKS
        if config['n_stocks'] < 180:
            symbols = symbols[:config['n_stocks']]
        
        data_config = DataConfig(
            symbols=symbols,
            train_start_date=config['train_start_date'],
            train_end_date=config['train_end_date'],
            val_start_date=config['val_start_date'],
            val_end_date=config['val_end_date'],
            test_start_date=config['test_start_date'],
            test_end_date=config['test_end_date'],
            sequence_length=config['sequence_length'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        # 創建模型
        model = TSEAlphaModel(model_config)
        
        # 創建訓練器
        trainer = ModelTrainer(model, training_config, data_config)
        
        # 創建資料載入器
        data_loader = TSEDataLoader(data_config)
        
        logger.info("✅ 訓練環境設置完成")
        
        return {
            'model': model,
            'trainer': trainer,
            'data_loader': data_loader,
            'training_config': training_config,
            'model_config': model_config,
            'data_config': data_config
        }
        
    except Exception as e:
        logger.error(f"❌ 訓練環境設置失敗: {e}")
        raise

def run_supervised_training(components, config):
    """執行監督學習訓練"""
    logger.info("🚀 開始監督學習訓練...")
    
    try:
        trainer = components['trainer']
        data_loader = components['data_loader']
        
        # 準備資料
        data_loader.prepare_data()
        train_loader, val_loader, test_loader = data_loader.get_dataloaders()
        
        logger.info(f"訓練集: {len(train_loader)} 批次")
        logger.info(f"驗證集: {len(val_loader)} 批次")
        logger.info(f"測試集: {len(test_loader)} 批次")
        
        # 開始訓練
        start_time = time.time()
        results = trainer.train_supervised(train_loader, val_loader, test_loader)
        training_time = time.time() - start_time
        
        logger.info(f"✅ 監督學習訓練完成")
        logger.info(f"訓練時間: {training_time/3600:.2f} 小時")
        logger.info(f"最佳驗證 Loss: {results['best_val_loss']:.6f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 監督學習訓練失敗: {e}")
        raise

def run_optuna_optimization(components, config):
    """執行 Optuna 超參數優化"""
    logger.info("🔍 開始 Optuna 超參數優化...")
    
    try:
        import optuna
        from training.optuna_optimizer import OptunaOptimizer
        
        # 創建 Optuna 優化器
        optimizer = OptunaOptimizer(components['training_config'])
        
        # 創建研究
        study = optimizer.create_study()
        
        # 執行優化
        study.optimize(
            optimizer.objective,
            n_trials=config['optuna_trials'],
            timeout=config['optuna_timeout']
        )
        
        logger.info(f"✅ Optuna 優化完成")
        logger.info(f"最佳參數: {study.best_params}")
        logger.info(f"最佳分數: {study.best_value:.6f}")
        
        return study.best_params
        
    except Exception as e:
        logger.error(f"❌ Optuna 優化失敗: {e}")
        raise

def main():
    """主要訓練流程"""
    parser = argparse.ArgumentParser(description='TSE Alpha 完整訓練 - RTX 4090')
    parser.add_argument('--mode', choices=['supervised', 'optuna', 'both'], default='supervised',
                       help='訓練模式')
    parser.add_argument('--epochs', type=int, help='訓練輪數')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--lr', type=float, help='學習率')
    parser.add_argument('--n-stocks', type=int, help='股票數量')
    parser.add_argument('--data-ratio', type=float, help='資料使用比例')
    parser.add_argument('--optuna-trials', type=int, help='Optuna 試驗次數')
    parser.add_argument('--optuna-timeout', type=int, help='Optuna 超時時間(秒)')
    parser.add_argument('--resume', type=str, help='從檢查點恢復')
    parser.add_argument('--force', action='store_true', help='強制執行(跳過 GPU 檢查)')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🚀 TSE Alpha 完整訓練 - RTX 4090")
    logger.info("=" * 60)
    
    # 檢查環境
    if not args.force and not check_rtx4090_environment():
        logger.error("❌ 環境檢查失敗")
        return False
    
    # 創建配置
    config = create_production_config(args)
    
    # 設置訓練環境
    components = setup_training_environment(config)
    
    # 執行訓練
    try:
        if args.mode in ['supervised', 'both']:
            supervised_results = run_supervised_training(components, config)
        
        if args.mode in ['optuna', 'both']:
            optuna_results = run_optuna_optimization(components, config)
        
        logger.info("🎉 完整訓練流程完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 訓練流程失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)