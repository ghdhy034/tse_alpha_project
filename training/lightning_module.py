# training/lightning_module.py
"""
TSE Alpha Lightning 模組
整合模型、損失函數、優化器的統一訓練介面
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple
import numpy as np
from omegaconf import DictConfig

from models.model_architecture import TSEAlphaModel, ModelConfig
from models.config.training_config import TrainingConfig


class TSEAlphaLightningModule(pl.LightningModule):
    """TSE Alpha Lightning 訓練模組"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # 創建模型配置
        model_config = ModelConfig(
            price_frame_shape=(config.model.n_stocks, config.model.sequence_length, config.model.price_features),
            fundamental_dim=config.model.fundamental_features,
            account_dim=config.model.account_features,
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            n_stocks=config.model.n_stocks,
            max_position=config.model.max_position
        )
        
        # 創建模型
        self.model = TSEAlphaModel(model_config)
        
        # 損失函數權重
        self.stock_loss_weight = config.training.stock_loss_weight
        self.position_loss_weight = config.training.position_loss_weight
        self.value_loss_weight = config.training.value_loss_weight
        self.risk_loss_weight = config.training.risk_loss_weight
        
        # 訓練模式 (supervised/reinforcement)
        self.training_mode = config.training.mode
        
        # 指標追蹤
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向傳播"""
        return self.model(observation)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """訓練步驟"""
        if self.training_mode == "supervised":
            return self._supervised_training_step(batch, batch_idx)
        elif self.training_mode == "reinforcement":
            return self._reinforcement_training_step(batch, batch_idx)
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
    
    def _supervised_training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """監督學習訓練步驟"""
        observation = {
            'price_frame': batch['price_frame'],
            'fundamental': batch['fundamental'],
            'account': batch['account']
        }
        
        # 前向傳播
        outputs = self.forward(observation)
        
        # 計算損失
        losses = self._compute_supervised_losses(outputs, batch)
        total_loss = self._combine_losses(losses)
        
        # 記錄指標
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, on_step=False, on_epoch=True)
        
        return total_loss
    
    def _reinforcement_training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """強化學習訓練步驟"""
        # TODO: 實作強化學習訓練邏輯
        # 這裡先返回監督學習損失作為佔位符
        return self._supervised_training_step(batch, batch_idx)
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """驗證步驟"""
        observation = {
            'price_frame': batch['price_frame'],
            'fundamental': batch['fundamental'],
            'account': batch['account']
        }
        
        # 前向傳播
        outputs = self.forward(observation)
        
        # 計算損失
        losses = self._compute_supervised_losses(outputs, batch)
        total_loss = self._combine_losses(losses)
        
        # 記錄指標
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True)
        
        # 計算準確率指標
        accuracy_metrics = self._compute_accuracy_metrics(outputs, batch)
        for metric_name, metric_value in accuracy_metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_step=False, on_epoch=True)
        
        return total_loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """測試步驟"""
        observation = {
            'price_frame': batch['price_frame'],
            'fundamental': batch['fundamental'],
            'account': batch['account']
        }
        
        # 前向傳播
        outputs = self.forward(observation)
        
        # 計算損失
        losses = self._compute_supervised_losses(outputs, batch)
        total_loss = self._combine_losses(losses)
        
        # 記錄指標
        self.log('test_loss', total_loss, on_step=False, on_epoch=True)
        for loss_name, loss_value in losses.items():
            self.log(f'test_{loss_name}', loss_value, on_step=False, on_epoch=True)
        
        # 計算準確率指標
        accuracy_metrics = self._compute_accuracy_metrics(outputs, batch)
        for metric_name, metric_value in accuracy_metrics.items():
            self.log(f'test_{metric_name}', metric_value, on_step=False, on_epoch=True)
        
        return total_loss
    
    def _compute_supervised_losses(self, outputs: Dict[str, torch.Tensor], 
                                 batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """計算監督學習損失"""
        losses = {}
        
        # 股票選擇損失 (交叉熵)
        if 'stock_labels' in batch:
            stock_loss = F.cross_entropy(outputs['stock_logits'], batch['stock_labels'])
            losses['stock_loss'] = stock_loss
        
        # 倉位大小損失 (MSE)
        if 'position_labels' in batch:
            position_loss = F.mse_loss(outputs['position_size'], batch['position_labels'])
            losses['position_loss'] = position_loss
        
        # 價值估計損失 (MSE)
        if 'value_labels' in batch:
            value_loss = F.mse_loss(outputs['value'], batch['value_labels'])
            losses['value_loss'] = value_loss
        
        # 風險評估損失 (BCE)
        if 'risk_labels' in batch:
            risk_loss = F.binary_cross_entropy(outputs['risk_score'], batch['risk_labels'])
            losses['risk_loss'] = risk_loss
        
        return losses
    
    def _combine_losses(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """組合多個損失"""
        total_loss = torch.tensor(0.0, device=self.device)
        
        if 'stock_loss' in losses:
            total_loss += self.stock_loss_weight * losses['stock_loss']
        
        if 'position_loss' in losses:
            total_loss += self.position_loss_weight * losses['position_loss']
        
        if 'value_loss' in losses:
            total_loss += self.value_loss_weight * losses['value_loss']
        
        if 'risk_loss' in losses:
            total_loss += self.risk_loss_weight * losses['risk_loss']
        
        return total_loss
    
    def _compute_accuracy_metrics(self, outputs: Dict[str, torch.Tensor], 
                                batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """計算準確率指標"""
        metrics = {}
        
        # 股票選擇準確率
        if 'stock_labels' in batch:
            stock_preds = outputs['stock_logits'].argmax(dim=1)
            stock_accuracy = (stock_preds == batch['stock_labels']).float().mean()
            metrics['stock_accuracy'] = stock_accuracy
        
        # 倉位大小 MAE
        if 'position_labels' in batch:
            position_mae = F.l1_loss(outputs['position_size'], batch['position_labels'])
            metrics['position_mae'] = position_mae
        
        # 價值估計 MAE
        if 'value_labels' in batch:
            value_mae = F.l1_loss(outputs['value'], batch['value_labels'])
            metrics['value_mae'] = value_mae
        
        return metrics
    
    def configure_optimizers(self):
        """配置優化器和學習率調度器"""
        # 優化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 學習率調度器
        if self.config.training.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif self.config.training.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.training.lr_factor,
                patience=self.config.training.lr_patience,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return optimizer
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """預測步驟"""
        observation = {
            'price_frame': batch['price_frame'],
            'fundamental': batch['fundamental'],
            'account': batch['account']
        }
        
        outputs = self.forward(observation)
        
        # 轉換為動作格式
        stock_probs = F.softmax(outputs['stock_logits'], dim=1)
        stock_indices = stock_probs.argmax(dim=1)
        
        return {
            'stock_indices': stock_indices,
            'stock_probs': stock_probs,
            'position_sizes': outputs['position_size'],
            'values': outputs['value'],
            'risk_scores': outputs['risk_score']
        }
    
    def on_train_epoch_end(self):
        """訓練輪次結束"""
        # 記錄學習率
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_epoch=True)
    
    def on_validation_epoch_end(self):
        """驗證輪次結束"""
        pass
    
    def get_model_for_inference(self) -> TSEAlphaModel:
        """獲取用於推理的模型"""
        return self.model