# models/model_architecture.py
"""
TSE Alpha 模型架構 - Conv1D + Transformer
與 Gym 環境完全相容的模型設計
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class ModelConfig:
    """模型配置"""
    # 輸入維度 (與 Gym 環境觀測空間對應，基於66維特徵配置)
    price_frame_shape: Tuple[int, int, int] = (10, 64, 51)  # (n_stocks, seq_len, other_features)
    fundamental_dim: int = 15  # 基本面特徵 (15個，66維配置)
    account_dim: int = 4       # 帳戶狀態特徵
    
    # 模型架構參數
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    
    # Conv1D 參數
    conv_channels: list = None
    conv_kernel_sizes: list = None
    
    # 輸出維度 (與 Gym 環境動作空間對應)
    n_stocks: int = 10
    max_position: int = 300
    
    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [64, 128, 256]
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [3, 3, 3]


class PriceFrameEncoder(nn.Module):
    """價格框架編碼器 - Conv1D + Transformer"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Conv1D 特徵提取層
        conv_layers = []
        in_channels = config.price_frame_shape[2]  # 53 (其他特徵: 價量+技術+籌碼+估值+日內結構)
        
        for out_channels, kernel_size in zip(config.conv_channels, config.conv_kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 投影到 hidden_dim
        self.projection = nn.Linear(config.conv_channels[-1], config.hidden_dim)
        
        # Transformer 編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # 位置編碼 - 使用足夠大的 max_len 來處理 n_stocks * seq_len
        max_seq_len = config.price_frame_shape[0] * config.price_frame_shape[1]  # n_stocks * seq_len
        self.pos_encoding = PositionalEncoding(config.hidden_dim, max_seq_len)
    
    def forward(self, price_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            price_frame: (batch_size, n_stocks, seq_len, features)
        Returns:
            encoded: (batch_size, n_stocks, hidden_dim)
        """
        batch_size, n_stocks, seq_len, features = price_frame.shape
        
        # 重塑為 (batch_size * n_stocks, features, seq_len) 用於 Conv1D
        x = price_frame.view(batch_size * n_stocks, seq_len, features)
        x = x.transpose(1, 2)  # (batch_size * n_stocks, features, seq_len)
        
        # Conv1D 特徵提取
        x = self.conv_layers(x)  # (batch_size * n_stocks, conv_channels[-1], seq_len)
        
        # 轉回序列格式
        x = x.transpose(1, 2)  # (batch_size * n_stocks, seq_len, conv_channels[-1])
        x = self.projection(x)  # (batch_size * n_stocks, seq_len, hidden_dim)
        
        # 重塑為 (batch_size, n_stocks * seq_len, hidden_dim)
        x = x.view(batch_size, n_stocks * seq_len, self.config.hidden_dim)
        
        # 位置編碼
        x = self.pos_encoding(x)
        
        # Transformer 編碼
        x = self.transformer(x)  # (batch_size, n_stocks * seq_len, hidden_dim)
        
        # 池化到每檔股票一個向量
        x = x.view(batch_size, n_stocks, seq_len, self.config.hidden_dim)
        x = x.mean(dim=2)  # (batch_size, n_stocks, hidden_dim)
        
        return x


class PositionalEncoding(nn.Module):
    """位置編碼"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 創建位置編碼矩陣: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 直接註冊為 (max_len, d_model) 形狀，不做額外變換
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()
        
        # pe shape: (max_len, d_model)
        # 確保位置編碼維度匹配
        pe_hidden_dim = self.pe.size(1)
        
        # 確保序列長度不超過最大長度
        actual_seq_len = min(seq_len, self.pe.size(0))
        
        if hidden_dim != pe_hidden_dim:
            # 如果維度不匹配，截取或填充
            if hidden_dim <= pe_hidden_dim:
                pe_slice = self.pe[:actual_seq_len, :hidden_dim]
            else:
                # 如果需要更多維度，用零填充
                pe_base = self.pe[:actual_seq_len, :]
                pe_slice = torch.zeros(actual_seq_len, hidden_dim, device=self.pe.device, dtype=self.pe.dtype)
                pe_slice[:, :pe_hidden_dim] = pe_base
        else:
            pe_slice = self.pe[:actual_seq_len, :hidden_dim]
        
        # 如果實際序列長度小於需要的長度，用零填充
        if actual_seq_len < seq_len:
            pe_full = torch.zeros(seq_len, hidden_dim, device=self.pe.device, dtype=self.pe.dtype)
            pe_full[:actual_seq_len, :] = pe_slice
            pe_slice = pe_full
        
        # 添加批次維度: (1, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        pe_expanded = pe_slice.unsqueeze(0).expand(batch_size, -1, -1)
        
        return x + pe_expanded


class AttentionPooling(nn.Module):
    """注意力池化層"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_stocks, hidden_dim)
        Returns:
            pooled: (batch_size, hidden_dim)
        """
        weights = F.softmax(self.attention(x), dim=1)  # (batch_size, n_stocks, 1)
        return (x * weights).sum(dim=1)  # (batch_size, hidden_dim)


class TSEAlphaModel(nn.Module):
    """TSE Alpha 主模型 - 與 Gym 環境完全相容"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 價格框架編碼器
        self.price_encoder = PriceFrameEncoder(config)
        
        # 基本面特徵編碼器
        self.fundamental_encoder = nn.Sequential(
            nn.Linear(config.fundamental_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2)
        )
        
        # 帳戶狀態編碼器
        self.account_encoder = nn.Sequential(
            nn.Linear(config.account_dim, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 4)
        )
        
        # 跨股票注意力
        self.cross_stock_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 特徵融合層
        fusion_input_dim = config.hidden_dim + config.hidden_dim // 2 + config.hidden_dim // 4
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 決策頭 - 股票選擇 (與 Gym 動作空間對應)
        self.stock_selector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.n_stocks)
        )
        
        # 決策頭 - 倉位大小
        self.position_sizer = nn.Sequential(
            nn.Linear(config.hidden_dim + config.n_stocks, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Tanh()  # 輸出 [-1, 1]
        )
        
        # 價值估計頭 (用於強化學習)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # 風險評估頭
        self.risk_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()  # 風險分數 [0, 1]
        )
    
    def forward(self, observation: Dict[str, torch.Tensor], 
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向傳播 - 與 Gym 環境觀測格式完全對應
        
        Args:
            observation: {
                'price_frame': (batch_size, n_stocks, seq_len, 5),
                'fundamental': (batch_size, 10),
                'account': (batch_size, 4)
            }
        Returns:
            outputs: {
                'stock_logits': (batch_size, n_stocks),
                'position_size': (batch_size, 1), 
                'value': (batch_size, 1),
                'risk_score': (batch_size, 1)
            }
        """
        # 編碼價格框架
        price_features = self.price_encoder(observation['price_frame'])  # (batch_size, n_stocks, hidden_dim)
        
        # 跨股票注意力
        price_features_attended, attention_weights = self.cross_stock_attention(
            price_features, price_features, price_features
        )
        price_features = price_features + price_features_attended  # 殘差連接
        
        # 全局價格特徵 (所有股票的聚合)
        global_price_features = price_features.mean(dim=1)  # (batch_size, hidden_dim)
        
        # 編碼基本面特徵
        fundamental_features = self.fundamental_encoder(observation['fundamental'])  # (batch_size, hidden_dim//2)
        
        # 編碼帳戶狀態
        account_features = self.account_encoder(observation['account'])  # (batch_size, hidden_dim//4)
        
        # 特徵融合
        fused_features = torch.cat([
            global_price_features, 
            fundamental_features, 
            account_features
        ], dim=1)
        fused_features = self.feature_fusion(fused_features)  # (batch_size, hidden_dim)
        
        # 股票選擇決策
        stock_logits = self.stock_selector(fused_features)  # (batch_size, n_stocks)
        
        # 倉位大小決策 (結合股票選擇信息)
        position_input = torch.cat([fused_features, stock_logits], dim=1)
        position_size = self.position_sizer(position_input)  # (batch_size, 1)
        
        # 價值估計
        value = self.value_head(fused_features)  # (batch_size, 1)
        
        # 風險評估
        risk_score = self.risk_head(fused_features)  # (batch_size, 1)
        
        outputs = {
            'stock_logits': stock_logits,
            'position_size': position_size,
            'value': value,
            'risk_score': risk_score
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            outputs['price_features'] = price_features
        
        return outputs
    
    def get_action(self, observation: Dict[str, torch.Tensor], 
                   deterministic: bool = False) -> Tuple[int, np.ndarray]:
        """
        獲取與 Gym 環境相容的動作
        
        Returns:
            action: (stock_idx, position_array) 符合 Gym 動作空間格式
        """
        with torch.no_grad():
            outputs = self.forward(observation)
            
            # 股票選擇 (只處理第一個樣本)
            stock_logits = outputs['stock_logits']
            if deterministic:
                stock_idx = stock_logits.argmax(dim=1)[0].item()  # 取第一個樣本
            else:
                stock_probs = F.softmax(stock_logits, dim=1)
                stock_idx = torch.multinomial(stock_probs[0:1], 1).item()  # 取第一個樣本
            
            # 倉位大小 (轉換為整數，只處理第一個樣本)
            position_size = outputs['position_size'][0].item()  # 取第一個樣本
            position_qty = int(position_size * self.config.max_position)
            
            return stock_idx, np.array([position_qty], dtype=np.int16)
    
    def get_value(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """獲取狀態價值 (用於強化學習)"""
        with torch.no_grad():
            outputs = self.forward(observation)
            return outputs['value']
    
    def evaluate_action(self, observation: Dict[str, torch.Tensor], 
                       action: Tuple[int, np.ndarray]) -> Dict[str, torch.Tensor]:
        """評估動作 (用於策略梯度)"""
        outputs = self.forward(observation)
        
        stock_idx, position_array = action
        position_qty = position_array[0] / self.config.max_position  # 標準化到 [-1, 1]
        
        # 計算動作概率
        stock_logits = outputs['stock_logits']
        stock_log_probs = F.log_softmax(stock_logits, dim=1)
        stock_log_prob = stock_log_probs[0, stock_idx]
        
        # 倉位大小的概率密度 (假設正態分佈)
        position_mean = outputs['position_size']
        position_std = 0.1  # 可調參數
        position_log_prob = -0.5 * ((position_qty - position_mean) / position_std) ** 2
        
        return {
            'stock_log_prob': stock_log_prob,
            'position_log_prob': position_log_prob,
            'value': outputs['value'],
            'risk_score': outputs['risk_score']
        }


def create_model(config: Optional[ModelConfig] = None) -> TSEAlphaModel:
    """創建模型實例"""
    if config is None:
        config = ModelConfig()
    return TSEAlphaModel(config)


def test_model():
    """測試模型架構"""
    print("=== 測試 TSE Alpha 模型架構 ===")
    
    # 創建模型
    config = ModelConfig(
        price_frame_shape=(10, 64, 5),
        n_stocks=10,
        hidden_dim=128
    )
    model = create_model(config)
    
    # 創建測試輸入 (模擬 Gym 環境觀測)
    batch_size = 2
    observation = {
        'price_frame': torch.randn(batch_size, 10, 64, 5),
        'fundamental': torch.randn(batch_size, 10),
        'account': torch.randn(batch_size, 4)
    }
    
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 前向傳播測試
    outputs = model(observation, return_attention=True)
    
    print("輸出形狀:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # 動作生成測試
    action = model.get_action(observation, deterministic=True)
    print(f"生成動作: 股票索引={action[0]}, 倉位={action[1]}")
    
    # 動作評估測試
    evaluation = model.evaluate_action(observation, action)
    print("動作評估:")
    for key, value in evaluation.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("✅ 模型架構測試完成")


if __name__ == "__main__":
    test_model()