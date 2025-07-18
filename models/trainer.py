# models/trainer.py
"""
TSE Alpha 模型訓練器 - 與 Gym 環境和回測系統完全相容
支援強化學習和監督學習兩種訓練模式
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, date
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import json
from tqdm import tqdm

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .model_architecture import TSEAlphaModel, ModelConfig
    from .data_loader import TSEDataLoader, DataConfig
    from gym_env.env import TSEAlphaEnv
    from backtest.ledger import TradingLedger
except ImportError as e:
    print(f"警告: 無法導入模組: {e}")

logger = logging.getLogger(__name__)


# 使用統一的 TrainingConfig
try:
    from models.config.training_config import TrainingConfig
except ImportError:
    from dataclasses import dataclass
    
    @dataclass
    class TrainingConfig:
        """簡化的訓練配置 (備用)"""
        model_name: str = "tse_alpha_v1"
        training_mode: str = "supervised"
        num_epochs: int = 100
        learning_rate: float = 1e-4
        batch_size: int = 32
        weight_decay: float = 1e-5
        gradient_clip_norm: float = 1.0
        rl_episodes: int = 1000
        rl_steps_per_episode: int = 252
        gamma: float = 0.99
        entropy_coef: float = 0.01
        value_loss_coef: float = 0.5
        validation_freq: int = 10
        early_stopping_patience: int = 20
        save_best_model: bool = True
        checkpoint_dir: str = "checkpoints"
        log_dir: str = "logs"
        device: str = "auto"
        mixed_precision: bool = True
        debug_mode: bool = False
        save_predictions: bool = False


class ModelTrainer:
    """TSE Alpha 模型訓練器"""
    
    def __init__(self, 
                 model: TSEAlphaModel,
                 config: TrainingConfig,
                 data_config: Optional[DataConfig] = None):
        self.model = model
        self.config = config
        self.data_config = data_config or DataConfig()
        
        # 設定設備
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # 設定優化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 設定學習率調度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # 設定損失函數
        self.criterion = self._setup_loss_functions()
        
        # 設定日誌
        self.writer = SummaryWriter(log_dir=f"{config.log_dir}/{config.model_name}")
        
        # 訓練狀態
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = []
        
        # 創建目錄
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        logger.info(f"模型訓練器初始化完成，設備: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """設定計算設備"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"使用 GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("使用 CPU")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _setup_loss_functions(self) -> Dict[str, nn.Module]:
        """設定損失函數"""
        return {
            'regression': nn.MSELoss(),
            'classification': nn.CrossEntropyLoss(),
            'value': nn.MSELoss(),
            'policy': nn.CrossEntropyLoss()
        }
    
    def train_supervised(self, 
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """監督學習訓練"""
        logger.info("開始監督學習訓練...")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # 訓練階段
            train_metrics = self._train_epoch_supervised(train_loader)
            
            # 驗證階段
            if epoch % self.config.validation_freq == 0:
                val_metrics = self._validate_epoch_supervised(val_loader)
                
                # 記錄指標
                self._log_metrics(train_metrics, val_metrics, epoch)
                
                # 檢查早停
                if self._check_early_stopping(val_metrics['loss']):
                    logger.info(f"早停觸發，在第 {epoch} 輪停止訓練")
                    break
                
                # 保存最佳模型
                if self.config.save_best_model and val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self._save_checkpoint(epoch, is_best=True)
        
        # 測試階段
        test_metrics = {}
        if test_loader:
            test_metrics = self._test_supervised(test_loader)
        
        # 保存最終模型
        self._save_checkpoint(self.current_epoch, is_final=True)
        
        return {
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'test_metrics': test_metrics
        }
    
    def _train_epoch_supervised(self, train_loader: DataLoader) -> Dict[str, float]:
        """監督學習訓練一個 epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移動資料到設備
            observation = {k: v.to(self.device) for k, v in batch['observation'].items()}
            labels = batch['labels'].to(self.device)
            
            # 前向傳播
            self.optimizer.zero_grad()
            outputs = self.model(observation)
            
            # 計算損失
            if self.data_config.label_type == 'regression':
                loss = self.criterion['regression'](outputs['value'], labels)
            elif self.data_config.label_type == 'classification':
                loss = self.criterion['classification'](outputs['stock_logits'], labels.long())
            else:
                # 混合損失
                value_loss = self.criterion['value'](outputs['value'], labels)
                policy_loss = self.criterion['policy'](outputs['stock_logits'], labels.long())
                loss = value_loss + 0.1 * policy_loss
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            # 更新參數
            self.optimizer.step()
            
            # 統計
            total_loss += loss.item()
            total_samples += labels.size(0)
            
            # 更新進度條
            progress_bar.set_postfix({'loss': loss.item()})
            
            # 記錄到 TensorBoard
            if batch_idx % 100 == 0:
                step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
        
        avg_loss = total_loss / len(train_loader)
        return {'loss': avg_loss, 'samples': total_samples}
    
    def _validate_epoch_supervised(self, val_loader: DataLoader) -> Dict[str, float]:
        """監督學習驗證一個 epoch"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # 移動資料到設備
                observation = {k: v.to(self.device) for k, v in batch['observation'].items()}
                labels = batch['labels'].to(self.device)
                
                # 前向傳播
                outputs = self.model(observation)
                
                # 計算損失
                if self.data_config.label_type == 'regression':
                    loss = self.criterion['regression'](outputs['value'], labels)
                    predictions.extend(outputs['value'].cpu().numpy())
                elif self.data_config.label_type == 'classification':
                    loss = self.criterion['classification'](outputs['stock_logits'], labels.long())
                    predictions.extend(torch.argmax(outputs['stock_logits'], dim=1).cpu().numpy())
                else:
                    value_loss = self.criterion['value'](outputs['value'], labels)
                    policy_loss = self.criterion['policy'](outputs['stock_logits'], labels.long())
                    loss = value_loss + 0.1 * policy_loss
                    predictions.extend(outputs['value'].cpu().numpy())
                
                targets.extend(labels.cpu().numpy())
                total_loss += loss.item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        
        # 計算額外指標
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        if self.data_config.label_type == 'regression':
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
            
            return {
                'loss': avg_loss,
                'mse': mse,
                'mae': mae,
                'correlation': correlation,
                'samples': total_samples
            }
        else:
            accuracy = np.mean(predictions == targets)
            return {
                'loss': avg_loss,
                'accuracy': accuracy,
                'samples': total_samples
            }
    
    def train_reinforcement(self, 
                           env: TSEAlphaEnv,
                           ledger: Optional[TradingLedger] = None) -> Dict[str, Any]:
        """強化學習訓練"""
        logger.info("開始強化學習訓練...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(self.config.rl_episodes):
            episode_reward, episode_length = self._train_episode_rl(env, episode, ledger)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 記錄指標
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                
                logger.info(f"Episode {episode}: 平均獎勵 = {avg_reward:.4f}, 平均長度 = {avg_length:.1f}")
                
                self.writer.add_scalar('RL/EpisodeReward', episode_reward, episode)
                self.writer.add_scalar('RL/AvgReward', avg_reward, episode)
                self.writer.add_scalar('RL/EpisodeLength', episode_length, episode)
            
            # 保存檢查點
            if episode % 100 == 0:
                self._save_checkpoint(episode, is_rl=True)
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'avg_final_reward': np.mean(episode_rewards[-100:])
        }
    
    def _train_episode_rl(self, 
                         env: TSEAlphaEnv, 
                         episode: int,
                         ledger: Optional[TradingLedger] = None) -> Tuple[float, int]:
        """強化學習訓練一個 episode"""
        self.model.train()
        
        # 重置環境
        observation, info = env.reset()
        
        # 轉換觀測格式
        obs_tensor = self._convert_observation(observation)
        
        episode_reward = 0.0
        episode_length = 0
        log_probs = []
        values = []
        rewards = []
        
        for step in range(self.config.rl_steps_per_episode):
            # 獲取動作
            with torch.no_grad():
                action = self.model.get_action(obs_tensor, deterministic=False)
            
            # 執行動作
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            # 記錄到帳本
            if ledger and info.get('trade_executed', False):
                trade_result = info.get('trade_result', {})
                if trade_result.get('success', False):
                    # 這裡可以記錄到帳本，但需要更多交易信息
                    pass
            
            # 評估動作
            evaluation = self.model.evaluate_action(obs_tensor, action)
            
            log_probs.append(evaluation['stock_log_prob'] + evaluation['position_log_prob'])
            values.append(evaluation['value'])
            rewards.append(reward)
            
            episode_reward += reward
            episode_length += 1
            
            # 更新觀測
            obs_tensor = self._convert_observation(next_observation)
            
            if terminated or truncated:
                break
        
        # 計算優勢和回報
        if len(rewards) > 0:
            returns = self._compute_returns(rewards, values)
            advantages = returns - torch.cat(values)
            
            # 策略梯度更新
            policy_loss = -(torch.cat(log_probs) * advantages.detach()).mean()
            value_loss = nn.MSELoss()(torch.cat(values), returns)
            
            # 總損失
            total_loss = policy_loss + self.config.value_loss_coef * value_loss
            
            # 反向傳播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
        
        return episode_reward, episode_length
    
    def _convert_observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """轉換觀測格式"""
        return {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
            for k, v in observation.items()
        }
    
    def _compute_returns(self, rewards: List[float], values: List[torch.Tensor]) -> torch.Tensor:
        """計算折扣回報"""
        returns = []
        R = 0
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.config.gamma * R
            returns.insert(0, R)
        
        return torch.tensor(returns, dtype=torch.float32).to(self.device)
    
    def _test_supervised(self, test_loader: DataLoader) -> Dict[str, float]:
        """監督學習測試"""
        logger.info("開始測試...")
        return self._validate_epoch_supervised(test_loader)
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """記錄訓練指標"""
        # TensorBoard 記錄
        self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        
        for key, value in val_metrics.items():
            if key != 'loss' and key != 'samples':
                self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # 學習率調度
        self.scheduler.step(val_metrics['loss'])
        
        # 保存歷史
        self.training_history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            **{f'val_{k}': v for k, v in val_metrics.items() if k not in ['loss', 'samples']}
        })
        
        logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.6f}, Val Loss = {val_metrics['loss']:.6f}")
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """檢查早停條件"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False, is_rl: bool = False):
        """保存檢查點"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config,
            'model_config': self.model.config
        }
        
        # 確定檔案名稱
        if is_best:
            filename = f"{self.config.model_name}_best.pth"
        elif is_final:
            filename = f"{self.config.model_name}_final.pth"
        elif is_rl:
            filename = f"{self.config.model_name}_rl_episode_{epoch}.pth"
        else:
            filename = f"{self.config.model_name}_epoch_{epoch}.pth"
        
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        logger.info(f"檢查點已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """載入檢查點"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"檢查點已載入: {filepath}")
        return checkpoint
    
    def evaluate_on_env(self, env: TSEAlphaEnv, num_episodes: int = 10) -> Dict[str, Any]:
        """在環境中評估模型"""
        self.model.eval()
        
        episode_rewards = []
        episode_navs = []
        
        for episode in range(num_episodes):
            observation, info = env.reset()
            obs_tensor = self._convert_observation(observation)
            
            episode_reward = 0.0
            initial_nav = info['nav']
            
            while True:
                with torch.no_grad():
                    action = self.model.get_action(obs_tensor, deterministic=True)
                
                observation, reward, terminated, truncated, info = env.step(action)
                obs_tensor = self._convert_observation(observation)
                
                episode_reward += reward
                
                if terminated or truncated:
                    final_nav = info['nav']
                    break
            
            episode_rewards.append(episode_reward)
            episode_navs.append(final_nav / initial_nav - 1.0)  # 收益率
        
        return {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_return': np.mean(episode_navs),
            'std_return': np.std(episode_navs),
            'sharpe_ratio': np.mean(episode_navs) / (np.std(episode_navs) + 1e-8),
            'episode_rewards': episode_rewards,
            'episode_returns': episode_navs
        }
    
    def close(self):
        """關閉訓練器"""
        if self.writer:
            self.writer.close()


def create_trainer(model: TSEAlphaModel, 
                  config: Optional[TrainingConfig] = None,
                  data_config: Optional[DataConfig] = None) -> ModelTrainer:
    """創建訓練器實例"""
    if config is None:
        config = TrainingConfig()
    
    return ModelTrainer(model, config, data_config)


def test_trainer():
    """測試訓練器"""
    print("=== 測試 TSE Alpha 訓練器 ===")
    
    try:
        from .model_architecture import create_model, ModelConfig
        
        # 創建模型
        model_config = ModelConfig(
            price_frame_shape=(5, 64, 5),
            n_stocks=5,
            hidden_dim=128
        )
        model = create_model(model_config)
        
        # 創建訓練配置
        training_config = TrainingConfig(
            num_epochs=2,
            batch_size=4,
            debug_mode=True
        )
        
        # 創建訓練器
        trainer = create_trainer(model, training_config)
        
        print(f"✅ 訓練器創建成功，設備: {trainer.device}")
        print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 測試檢查點保存
        trainer._save_checkpoint(0, is_best=True)
        print("✅ 檢查點保存測試完成")
        
        trainer.close()
        
    except Exception as e:
        print(f"❌ 訓練器測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_trainer()