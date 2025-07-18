# models/config/model_config.py
"""
TSE Alpha 模型架構配置

管理模型架構相關的配置參數，與 model_architecture.py 中的 ModelConfig 保持一致。
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch


@dataclass
class ModelConfig:
    """模型架構配置 - 與現有 model_architecture.py 保持一致"""
    
    # ==================== 輸入維度配置 ====================
    # 與 Gym 環境觀測空間對應
    price_frame_shape: Tuple[int, int, int] = (180, 64, 21)  # (n_stocks, seq_len, features)
    fundamental_dim: int = 25                # 基本面特徵維度
    account_dim: int = 4                     # 帳戶狀態維度
    
    # ==================== 模型架構參數 ====================
    n_stocks: int = 180                      # 股票數量
    hidden_dim: int = 256                    # 隱藏層維度
    num_heads: int = 8                       # 注意力頭數
    num_layers: int = 4                      # Transformer 層數
    dropout: float = 0.1                     # Dropout 比率
    
    # ==================== Conv1D 配置 ====================
    conv_channels: Tuple[int, ...] = (64, 128, 256)  # 卷積通道數
    conv_kernel_sizes: Tuple[int, ...] = (3, 3, 3)   # 卷積核大小
    conv_stride: int = 1                     # 卷積步長
    conv_padding: int = 1                    # 卷積填充
    
    # ==================== Transformer 配置 ====================
    transformer_dim_feedforward: int = 1024  # 前饋網絡維度
    transformer_activation: str = "relu"     # 激活函數
    transformer_norm_first: bool = False     # 是否先進行歸一化
    
    # ==================== 輸出配置 ====================
    num_actions: int = 3                     # 動作數量 (Buy/Sell/Hold)
    action_dim: int = 2                      # 動作維度 (股票索引, 倉位)
    
    # ==================== 正則化配置 ====================
    layer_norm: bool = True                  # 是否使用 Layer Normalization
    batch_norm: bool = False                 # 是否使用 Batch Normalization
    weight_init: str = "xavier_uniform"      # 權重初始化方法
    
    # ==================== 位置編碼配置 ====================
    use_positional_encoding: bool = True     # 是否使用位置編碼
    max_position_embeddings: int = 5000      # 最大位置編碼長度
    
    def __post_init__(self):
        """初始化後的驗證"""
        self._validate_config()
    
    def _validate_config(self):
        """驗證配置的合理性"""
        # 驗證輸入維度
        n_stocks, seq_len, features = self.price_frame_shape
        if n_stocks != self.n_stocks:
            raise ValueError(f"price_frame_shape 中的股票數量 ({n_stocks}) 與 n_stocks ({self.n_stocks}) 不匹配")
        
        # 驗證隱藏層維度能被注意力頭數整除
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) 必須能被 num_heads ({self.num_heads}) 整除")
        
        # 驗證卷積配置
        if len(self.conv_channels) != len(self.conv_kernel_sizes):
            raise ValueError("conv_channels 和 conv_kernel_sizes 長度必須相同")
        
        # 驗證 dropout 範圍
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout ({self.dropout}) 必須在 [0, 1] 範圍內")
    
    def get_model_summary(self) -> dict:
        """獲取模型摘要信息"""
        return {
            'input_shapes': {
                'price_frame': self.price_frame_shape,
                'fundamental': (self.fundamental_dim,),
                'account': (self.account_dim,)
            },
            'architecture': {
                'hidden_dim': self.hidden_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'conv_channels': self.conv_channels
            },
            'output_shapes': {
                'actions': (self.num_actions,),
                'action_dim': self.action_dim
            },
            'parameters': {
                'dropout': self.dropout,
                'layer_norm': self.layer_norm,
                'positional_encoding': self.use_positional_encoding
            }
        }
    
    def estimate_model_size(self) -> dict:
        """估算模型大小"""
        # 簡化的參數數量估算
        conv_params = sum(
            in_ch * out_ch * kernel_size + out_ch  # 權重 + 偏置
            for in_ch, out_ch, kernel_size in zip(
                [self.price_frame_shape[2]] + list(self.conv_channels[:-1]),
                self.conv_channels,
                self.conv_kernel_sizes
            )
        )
        
        # Transformer 參數估算 (簡化)
        transformer_params = (
            self.num_layers * (
                4 * self.hidden_dim * self.hidden_dim +  # 注意力權重
                2 * self.hidden_dim * self.transformer_dim_feedforward +  # 前饋網絡
                4 * self.hidden_dim  # 偏置和歸一化
            )
        )
        
        # 輸出層參數
        output_params = self.hidden_dim * self.num_actions + self.num_actions
        
        total_params = conv_params + transformer_params + output_params
        
        return {
            'conv_parameters': conv_params,
            'transformer_parameters': transformer_params,
            'output_parameters': output_params,
            'total_parameters': total_params,
            'estimated_memory_mb': total_params * 4 / (1024 * 1024)  # 假設 float32
        }


def create_default_model_config() -> ModelConfig:
    """創建默認模型配置"""
    return ModelConfig()


def create_small_model_config() -> ModelConfig:
    """創建小型模型配置 (用於調試)"""
    return ModelConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        conv_channels=(32, 64, 128),
        transformer_dim_feedforward=512
    )


def create_large_model_config() -> ModelConfig:
    """創建大型模型配置 (用於生產)"""
    return ModelConfig(
        hidden_dim=512,
        num_heads=16,
        num_layers=6,
        conv_channels=(128, 256, 512),
        transformer_dim_feedforward=2048
    )


if __name__ == "__main__":
    # 測試模型配置
    print("=== TSE Alpha 模型配置系統測試 ===")
    
    # 測試默認配置
    config = create_default_model_config()
    print("✅ 默認模型配置創建成功")
    
    # 顯示模型摘要
    print("\n📊 模型摘要:")
    summary = config.get_model_summary()
    for section, details in summary.items():
        print(f"  {section}:")
        for key, value in details.items():
            print(f"    {key}: {value}")
    
    # 顯示模型大小估算
    print("\n📐 模型大小估算:")
    size_info = config.estimate_model_size()
    for key, value in size_info.items():
        if 'parameters' in key:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value:.2f}")
    
    # 測試小型和大型配置
    small_config = create_small_model_config()
    large_config = create_large_model_config()
    
    print(f"\n📊 配置對比:")
    print(f"  小型模型參數: {small_config.estimate_model_size()['total_parameters']:,}")
    print(f"  默認模型參數: {config.estimate_model_size()['total_parameters']:,}")
    print(f"  大型模型參數: {large_config.estimate_model_size()['total_parameters']:,}")
    
    print("\n🎯 模型配置系統測試完成！")