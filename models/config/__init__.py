# models/config/__init__.py
"""
TSE Alpha 訓練模組配置系統

提供統一的配置管理，包括：
- 訓練配置 (TrainingConfig)
- 模型配置 (ModelConfig)
- 資料處理配置
"""

from .training_config import TrainingConfig
from .model_config import ModelConfig

__all__ = ['TrainingConfig', 'ModelConfig']