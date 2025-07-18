# models/optimizer.py
"""
TSE Alpha Optuna 超參數優化器
整合 Optuna 進行自動超參數調優
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import logging
import json
from datetime import datetime

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .model_architecture import TSEAlphaModel, ModelConfig
    from .trainer import ModelTrainer, TrainingConfig as BaseTrainingConfig
    from .data_loader import TSEDataLoader, DataConfig
except ImportError as e:
    print(f"警告: 無法導入模組: {e}")

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Optuna 優化配置"""
    
    # 優化目標
    direction: str = "maximize"  # "maximize" 或 "minimize"
    metric_name: str = "val_sharpe_ratio"  # 優化目標指標
    
    # 試驗配置
    n_trials: int = 100  # 試驗次數
    timeout: int = 3600  # 超時時間 (秒)
    n_jobs: int = 1  # 並行作業數
    
    # 早停配置
    pruning_enabled: bool = True
    patience: int = 10  # 早停耐心值
    min_trials: int = 20  # 最少試驗次數
    
    # 搜索空間配置
    search_space: Dict[str, Dict[str, Any]] = None
    
    # 結果保存
    study_name: str = "tse_alpha_optimization"
    storage_url: str = "sqlite:///optuna_studies.db"
    save_best_model: bool = True
    
    def __post_init__(self):
        if self.search_space is None:
            self.search_space = self._get_default_search_space()
    
    def _get_default_search_space(self) -> Dict[str, Dict[str, Any]]:
        """獲取預設搜索空間"""
        return {
            # 模型架構參數
            "conv_channels": {"type": "categorical", "choices": [64, 128, 256]},
            "conv_kernel_size": {"type": "int", "low": 3, "high": 7, "step": 2},
            "n_conv_layers": {"type": "int", "low": 2, "high": 5},
            
            "transformer_dim": {"type": "categorical", "choices": [128, 256, 512]},
            "n_heads": {"type": "categorical", "choices": [4, 8, 16]},
            "n_transformer_layers": {"type": "int", "low": 2, "high": 6},
            "dropout_rate": {"type": "float", "low": 0.1, "high": 0.5},
            
            # 訓練參數
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
            "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
            
            # 資料參數
            "sequence_length": {"type": "categorical", "choices": [32, 64, 128]},
            "prediction_horizon": {"type": "categorical", "choices": [1, 3, 5, 10]},
        }


class OptunaTuner:
    """Optuna 超參數調優器"""
    
    def __init__(self, 
                 data_config: DataConfig,
                 optimization_config: OptimizationConfig):
        self.data_config = data_config
        self.opt_config = optimization_config
        
        # 創建或載入研究
        self.study = optuna.create_study(
            study_name=self.opt_config.study_name,
            storage=self.opt_config.storage_url,
            direction=self.opt_config.direction,
            load_if_exists=True
        )
        
        # 最佳結果追蹤
        self.best_trial = None
        self.best_model = None
        self.best_score = float('-inf') if self.opt_config.direction == "maximize" else float('inf')
        
        logger.info(f"Optuna 調優器初始化完成: {self.opt_config.study_name}")
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """建議超參數"""
        params = {}
        
        for param_name, param_config in self.opt_config.search_space.items():
            param_type = param_config["type"]
            
            if param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config["low"], 
                    param_config["high"],
                    step=param_config.get("step", 1)
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"]
                )
            elif param_type == "loguniform":
                params[param_name] = trial.suggest_loguniform(
                    param_name,
                    param_config["low"],
                    param_config["high"]
                )
        
        return params
    
    def create_model_config(self, params: Dict[str, Any]) -> ModelConfig:
        """根據超參數創建模型配置"""
        return ModelConfig(
            # 輸入維度 (固定)
            n_stocks=self.data_config.symbols.__len__() if hasattr(self.data_config, 'symbols') else 180,
            sequence_length=params.get("sequence_length", 64),
            price_features=21,
            fundamental_features=25,
            account_features=4,
            
            # Conv1D 參數
            conv_channels=params.get("conv_channels", 128),
            conv_kernel_size=params.get("conv_kernel_size", 5),
            n_conv_layers=params.get("n_conv_layers", 3),
            
            # Transformer 參數
            transformer_dim=params.get("transformer_dim", 256),
            n_heads=params.get("n_heads", 8),
            n_transformer_layers=params.get("n_transformer_layers", 4),
            
            # 正則化
            dropout_rate=params.get("dropout_rate", 0.2),
            
            # 輸出維度
            output_dim=1  # 回歸輸出
        )
    
    def create_training_config(self, params: Dict[str, Any]) -> BaseTrainingConfig:
        """根據超參數創建訓練配置"""
        return BaseTrainingConfig(
            # 訓練參數
            learning_rate=params.get("learning_rate", 1e-3),
            batch_size=params.get("batch_size", 32),
            weight_decay=params.get("weight_decay", 1e-4),
            
            # 其他固定參數
            max_epochs=50,
            patience=10,
            min_delta=1e-4
        )
    
    def objective(self, trial: optuna.Trial) -> float:
        """優化目標函數"""
        try:
            # 建議超參數
            params = self.suggest_hyperparameters(trial)
            
            # 創建配置
            model_config = self.create_model_config(params)
            training_config = self.create_training_config(params)
            
            # 更新資料配置
            updated_data_config = DataConfig(
                symbols=self.data_config.symbols,
                train_start_date=self.data_config.train_start_date,
                train_end_date=self.data_config.train_end_date,
                val_start_date=self.data_config.val_start_date,
                val_end_date=self.data_config.val_end_date,
                test_start_date=self.data_config.test_start_date,
                test_end_date=self.data_config.test_end_date,
                sequence_length=params.get("sequence_length", 64),
                prediction_horizon=params.get("prediction_horizon", 5),
                batch_size=params.get("batch_size", 32)
            )
            
            # 創建模型和訓練器
            model = TSEAlphaModel(model_config)
            trainer = ModelTrainer(model, training_config)
            
            # 準備資料
            data_loader = TSEDataLoader(updated_data_config)
            train_loader, val_loader, test_loader = data_loader.get_dataloaders()
            
            # 訓練模型
            trainer.train(train_loader, val_loader)
            
            # 評估模型
            metrics = trainer.evaluate(val_loader)
            
            # 獲取目標指標
            score = metrics.get(self.opt_config.metric_name, 0.0)
            
            # 更新最佳結果
            is_better = (
                (self.opt_config.direction == "maximize" and score > self.best_score) or
                (self.opt_config.direction == "minimize" and score < self.best_score)
            )
            
            if is_better:
                self.best_score = score
                self.best_trial = trial
                if self.opt_config.save_best_model:
                    self.best_model = model.state_dict().copy()
            
            # 記錄試驗結果
            trial.set_user_attr("all_metrics", metrics)
            
            logger.info(f"Trial {trial.number}: {self.opt_config.metric_name} = {score:.4f}")
            
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} 失敗: {e}")
            # 返回最差分數
            return float('-inf') if self.opt_config.direction == "maximize" else float('inf')
    
    def optimize(self) -> Dict[str, Any]:
        """執行優化"""
        logger.info(f"開始 Optuna 優化: {self.opt_config.n_trials} 次試驗")
        
        # 添加剪枝回調 (如果啟用)
        callbacks = []
        if self.opt_config.pruning_enabled:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.opt_config.min_trials,
                n_warmup_steps=10
            )
            self.study.pruner = pruner
        
        # 執行優化
        self.study.optimize(
            self.objective,
            n_trials=self.opt_config.n_trials,
            timeout=self.opt_config.timeout,
            n_jobs=self.opt_config.n_jobs
        )
        
        # 獲取最佳結果
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        logger.info(f"優化完成! 最佳 {self.opt_config.metric_name}: {best_value:.4f}")
        logger.info(f"最佳參數: {best_params}")
        
        # 保存結果
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "best_trial_number": best_trial.number,
            "n_trials": len(self.study.trials),
            "study_name": self.opt_config.study_name
        }
        
        # 保存最佳模型
        if self.opt_config.save_best_model and self.best_model is not None:
            model_path = f"models/checkpoints/best_model_{self.opt_config.study_name}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.best_model, model_path)
            results["best_model_path"] = model_path
        
        return results
    
    def get_optimization_history(self) -> pd.DataFrame:
        """獲取優化歷史"""
        trials_data = []
        
        for trial in self.study.trials:
            trial_data = {
                "trial_number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                "datetime_start": trial.datetime_start,
                "datetime_complete": trial.datetime_complete
            }
            
            # 添加參數
            trial_data.update(trial.params)
            
            # 添加用戶屬性
            if trial.user_attrs:
                trial_data.update(trial.user_attrs)
            
            trials_data.append(trial_data)
        
        return pd.DataFrame(trials_data)
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """繪製優化歷史"""
        try:
            import matplotlib.pyplot as plt
            
            df = self.get_optimization_history()
            completed_trials = df[df['state'] == 'COMPLETE']
            
            if len(completed_trials) == 0:
                logger.warning("沒有完成的試驗可以繪製")
                return
            
            plt.figure(figsize=(12, 8))
            
            # 優化歷史
            plt.subplot(2, 2, 1)
            plt.plot(completed_trials['trial_number'], completed_trials['value'])
            plt.xlabel('Trial Number')
            plt.ylabel(self.opt_config.metric_name)
            plt.title('Optimization History')
            
            # 參數重要性 (如果有足夠的試驗)
            if len(completed_trials) >= 10:
                plt.subplot(2, 2, 2)
                importance = optuna.importance.get_param_importances(self.study)
                params = list(importance.keys())
                values = list(importance.values())
                plt.barh(params, values)
                plt.xlabel('Importance')
                plt.title('Parameter Importance')
            
            # 最佳值歷史
            plt.subplot(2, 2, 3)
            best_values = []
            current_best = float('-inf') if self.opt_config.direction == "maximize" else float('inf')
            
            for value in completed_trials['value']:
                if self.opt_config.direction == "maximize":
                    current_best = max(current_best, value)
                else:
                    current_best = min(current_best, value)
                best_values.append(current_best)
            
            plt.plot(completed_trials['trial_number'], best_values)
            plt.xlabel('Trial Number')
            plt.ylabel(f'Best {self.opt_config.metric_name}')
            plt.title('Best Value History')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"優化歷史圖表已保存: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib 未安裝，無法繪製圖表")


def test_optimizer():
    """測試優化器"""
    print("=== 測試 Optuna 優化器 ===")
    
    try:
        # 創建測試配置
        data_config = DataConfig(
            symbols=['2330', '2317'],
            train_start_date='2024-01-01',
            train_end_date='2024-02-29',
            val_start_date='2024-03-01',
            val_end_date='2024-03-31',
            sequence_length=32,
            batch_size=16
        )
        
        opt_config = OptimizationConfig(
            n_trials=3,  # 少量試驗用於測試
            timeout=300,  # 5分鐘超時
            study_name="test_optimization"
        )
        
        # 創建優化器
        tuner = OptunaTuner(data_config, opt_config)
        print("OK Optuna 優化器創建成功")
        
        # 測試超參數建議
        study = optuna.create_study()
        trial = study.ask()
        params = tuner.suggest_hyperparameters(trial)
        print(f"OK 超參數建議: {params}")
        
        # 測試配置創建
        model_config = tuner.create_model_config(params)
        training_config = tuner.create_training_config(params)
        print("OK 配置創建成功")
        
        print("OK 優化器測試完成")
        
    except Exception as e:
        print(f"ERROR 優化器測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_optimizer()