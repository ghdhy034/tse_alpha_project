# models/data_loader.py
"""
TSE Alpha 資料載入器 - 與特徵工程和 Gym 環境相容
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass
from datetime import datetime, date
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "data_pipeline"))
sys.path.append(str(Path(__file__).parent.parent / "market_data_collector"))

try:
    from data_pipeline.features import FeatureEngine
    from market_data_collector.utils.db import query_df
    from market_data_collector.utils.config import STOCK_IDS
except ImportError as e:
    print(f"警告: 無法導入模組: {e}")
    STOCK_IDS = ['2330', '2317', '2603']

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """資料配置"""
    # 資料範圍
    symbols: List[str] = None
    train_start_date: str = '2020-03-02'
    train_end_date: str = '2023-12-31'
    val_start_date: str = '2024-01-01'
    val_end_date: str = '2024-06-30'
    test_start_date: str = '2024-07-01'
    test_end_date: str = '2024-12-31'
    
    # 序列參數
    sequence_length: int = 64
    prediction_horizon: int = 5
    
    # 批次參數
    batch_size: int = 32
    num_workers: int = 4
    
    # 特徵參數
    normalize_features: bool = True
    include_chip_features: bool = True
    
    # 標籤類型
    label_type: str = 'regression'  # 'regression', 'classification', 'ranking'
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = STOCK_IDS[:20]  # 預設使用前20檔股票


class TSEDataset(Dataset):
    """TSE Alpha 資料集 - 與 Gym 環境觀測格式相容"""
    
    def __init__(self, 
                 features_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
                 config: DataConfig,
                 mode: str = 'train'):
        """
        Args:
            features_dict: {symbol: (features, labels, prices)}
            config: 資料配置
            mode: 'train', 'val', 'test'
        """
        self.features_dict = features_dict
        self.config = config
        self.mode = mode
        self.symbols = list(features_dict.keys())
        
        # 建立序列索引
        self.sequence_indices = self._build_sequence_indices()
        
        logger.info(f"{mode} 資料集: {len(self.sequence_indices)} 個序列, {len(self.symbols)} 檔股票")
    
    def _build_sequence_indices(self) -> List[Tuple[str, int]]:
        """建立序列索引 (symbol, start_idx)"""
        indices = []
        
        for symbol in self.symbols:
            if symbol not in self.features_dict:
                continue
                
            features, labels, prices = self.features_dict[symbol]
            
            # 確保有足夠的資料建立序列
            min_length = self.config.sequence_length + self.config.prediction_horizon
            if len(features) < min_length:
                continue
            
            # 為每個可能的起始位置建立索引
            max_start_idx = len(features) - min_length
            for start_idx in range(max_start_idx + 1):
                indices.append((symbol, start_idx))
        
        return indices
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        獲取單個樣本 - 格式與 Gym 環境觀測相容
        
        Returns:
            sample: {
                'observation': {
                    'price_frame': (n_stocks, seq_len, 5),
                    'fundamental': (10,),
                    'account': (4,)
                },
                'labels': (prediction_horizon,),
                'metadata': {...}
            }
        """
        symbol, start_idx = self.sequence_indices[idx]
        features, labels, prices = self.features_dict[symbol]
        
        # 提取序列
        end_idx = start_idx + self.config.sequence_length
        label_idx = end_idx + self.config.prediction_horizon - 1
        
        # 價格框架資料 (OHLCV)
        price_sequence = prices.iloc[start_idx:end_idx]
        price_frame = self._extract_price_frame(price_sequence)
        
        # 基本面特徵 (使用序列最後一天的值)
        feature_row = features.iloc[end_idx - 1]
        fundamental_features = self._extract_fundamental_features(feature_row)
        
        # 帳戶狀態 (模擬)
        account_features = self._simulate_account_state(symbol, end_idx)
        
        # 標籤
        if label_idx < len(labels):
            label_row = labels.iloc[label_idx]
            target_labels = self._extract_labels(label_row)
        else:
            target_labels = torch.zeros(1)  # 預設標籤
        
        # 構建觀測 (與 Gym 環境格式相容)
        observation = {
            'price_frame': price_frame,
            'fundamental': fundamental_features,
            'account': account_features
        }
        
        # 元資料 (修復 Timestamp 序列化問題)
        metadata = {
            'symbol': symbol,
            'start_date': str(price_sequence.index[0]),  # 轉換為字符串
            'end_date': str(price_sequence.index[-1]),   # 轉換為字符串
            'current_price': float(price_sequence.iloc[-1]['close'])
        }
        
        return {
            'observation': observation,
            'labels': target_labels,
            'metadata': metadata
        }
    
    def _extract_price_frame(self, price_sequence: pd.DataFrame) -> torch.Tensor:
        """提取價格框架 - 基於66維特徵配置的其他特徵部分"""
        # 獲取實際配置的其他特徵數量 (51個: 價量+技術+籌碼+估值+日內結構)
        try:
            from models.config.training_config import TrainingConfig
            config = TrainingConfig()
            other_features_count = config.other_features  # 51個其他特徵
        except:
            other_features_count = 51  # 預設值 (66維配置)
        
        # 使用特徵引擎計算完整的其他特徵 (51個，66維配置)
        from data_pipeline.features import FeatureEngine
        feature_engine = FeatureEngine()
        
        # 計算所有其他特徵 (51個: 價量+技術+籌碼+估值+日內結構，66維配置)
        # 這裡需要完整的特徵處理，暫時使用技術特徵作為基礎
        tech_features = feature_engine.calculate_technical_features(price_sequence)
        
        if tech_features.empty or tech_features.shape[1] < 27:
            # 如果特徵計算失敗，使用零填充
            other_array = np.zeros((len(price_sequence), other_features_count))
        else:
            # 創建51個其他特徵的數組
            other_array = np.zeros((len(price_sequence), other_features_count))
            tech_array = tech_features.to_numpy()  # (seq_len, 27)
            
            # 填充前27個特徵（技術特徵）
            other_array[:, :min(27, other_features_count)] = tech_array[:, :min(27, other_features_count)]
            
            # 填充剩餘的24個特徵（籌碼+估值+日內結構+其他）
            remaining_features = other_features_count - 27
            if remaining_features > 0 and len(price_sequence) > 0:
                # 添加一些基於價格的簡單特徵作為填充
                close_prices = price_sequence['close'].values
                if len(close_prices) > 1:
                    # 計算價格變化率
                    price_returns = np.diff(close_prices) / (close_prices[:-1] + 1e-8)
                    price_returns = np.concatenate([[0], price_returns])  # 第一個值設為0
                    
                    # 計算移動平均偏差
                    if len(close_prices) >= 5:
                        ma5 = np.convolve(close_prices, np.ones(5)/5, mode='same')
                        ma_deviation = (close_prices - ma5) / (ma5 + 1e-8)
                    else:
                        ma_deviation = np.zeros_like(close_prices)
                    
                    # 填充剩餘特徵
                    for i in range(remaining_features):
                        if i % 2 == 0 and i//2 < len(price_returns):
                            other_array[:, 27 + i] = price_returns[min(i//2, len(price_returns)-1)]
                        elif i % 2 == 1 and i//2 < len(ma_deviation):
                            other_array[:, 27 + i] = ma_deviation[min(i//2, len(ma_deviation)-1)]
        
        # 擴展到多股票格式 (為了與模型相容)
        # 使用實際的symbols數量而不是config.symbols
        actual_symbols_count = len(self.symbols) if hasattr(self, 'symbols') else 1
        price_frame = np.zeros((actual_symbols_count, self.config.sequence_length, other_features_count))
        
        # 找到當前股票在符號列表中的位置
        symbol_idx = 0  # 預設使用第一個位置
        if hasattr(self, 'symbols') and len(self.symbols) > 0:
            # 從sequence_indices中獲取當前symbol
            current_symbol = self.sequence_indices[0][0] if self.sequence_indices else self.symbols[0]
            if current_symbol in self.symbols:
                symbol_idx = self.symbols.index(current_symbol)
        
        # 填充當前股票的資料 (確保索引安全)
        seq_len = min(len(other_array), self.config.sequence_length)
        if symbol_idx < price_frame.shape[0]:  # 安全檢查
            price_frame[symbol_idx, -seq_len:, :] = other_array[-seq_len:]
        
        return torch.tensor(price_frame, dtype=torch.float32)
    
    def _extract_fundamental_features(self, feature_row: pd.Series) -> torch.Tensor:
        """提取基本面特徵"""
        # 獲取實際配置的基本面特徵數量 (15個: 月營收+財報，66維配置)
        try:
            from models.config.training_config import TrainingConfig
            config = TrainingConfig()
            fundamental_dim = config.fundamental_features  # 15個基本面特徵 (66維配置)
        except:
            fundamental_dim = 15  # 預設值 (66維配置)
        
        # 從特徵行中提取基本面相關特徵
        fundamental_cols = [col for col in feature_row.index if 'fundamental' in col or 'market_cap' in col or 'liquidity' in col]
        
        if len(fundamental_cols) >= fundamental_dim:
            fundamental_values = feature_row[fundamental_cols[:fundamental_dim]].values
        else:
            # 如果基本面特徵不足，用其他特徵補充
            all_values = feature_row.values
            fundamental_values = np.zeros(fundamental_dim)
            fundamental_values[:min(len(all_values), fundamental_dim)] = all_values[:min(len(all_values), fundamental_dim)]
        
        return torch.tensor(fundamental_values, dtype=torch.float32)
    
    def _simulate_account_state(self, symbol: str, current_idx: int) -> torch.Tensor:
        """模擬帳戶狀態"""
        # 在實際應用中，這應該從回測系統或實際帳戶狀態獲取
        # 這裡提供模擬值
        
        # NAV 變化 (標準化)
        nav_change = np.random.normal(0, 0.01)
        
        # 持倉比例
        position_ratio = np.random.uniform(0, 0.8)
        
        # 未實現損益百分比
        unrealized_pnl = np.random.normal(0, 0.02)
        
        # 風險緩衝
        risk_buffer = np.random.uniform(0.2, 1.0)
        
        account_state = np.array([nav_change, position_ratio, unrealized_pnl, risk_buffer])
        return torch.tensor(account_state, dtype=torch.float32)
    
    def _extract_labels(self, label_row: pd.Series) -> torch.Tensor:
        """提取標籤"""
        if self.config.label_type == 'regression':
            # 回歸標籤 (未來收益率)
            return_col = f'return_{self.config.prediction_horizon}d'
            if return_col in label_row.index:
                return torch.tensor([label_row[return_col]], dtype=torch.float32)
            else:
                return torch.tensor([0.0], dtype=torch.float32)
        
        elif self.config.label_type == 'classification':
            # 分類標籤 (漲跌)
            up_col = f'up_{self.config.prediction_horizon}d'
            if up_col in label_row.index:
                return torch.tensor([label_row[up_col]], dtype=torch.long)
            else:
                return torch.tensor([0], dtype=torch.long)
        
        elif self.config.label_type == 'ranking':
            # 排序標籤 (相對表現)
            return_col = f'return_{self.config.prediction_horizon}d'
            if return_col in label_row.index:
                return torch.tensor([label_row[return_col]], dtype=torch.float32)
            else:
                return torch.tensor([0.0], dtype=torch.float32)
        
        return torch.tensor([0.0], dtype=torch.float32)


class MultiStockDataset(Dataset):
    """多股票資料集 - 同時處理多檔股票"""
    
    def __init__(self,
                 features_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
                 config: DataConfig,
                 mode: str = 'train'):
        self.features_dict = features_dict
        self.config = config
        self.mode = mode
        self.symbols = list(features_dict.keys())[:config.symbols.__len__()]  # 限制股票數量
        
        # 對齊所有股票的日期
        self.aligned_data = self._align_stock_data()
        self.date_indices = list(range(len(self.aligned_data)))
        
        logger.info(f"{mode} 多股票資料集: {len(self.date_indices)} 個時間點, {len(self.symbols)} 檔股票")
    
    def _align_stock_data(self) -> pd.DataFrame:
        """對齊所有股票的資料到相同的日期索引"""
        all_dates = set()
        
        # 收集所有日期
        for symbol in self.symbols:
            if symbol in self.features_dict:
                features, _, _ = self.features_dict[symbol]
                all_dates.update(features.index)
        
        # 創建統一的日期索引
        common_dates = sorted(all_dates)
        return pd.DataFrame(index=common_dates)
    
    def __len__(self) -> int:
        # 修復：確保有足夠的資料點來建立序列
        available_dates = len(self.date_indices)
        min_required = self.config.sequence_length + self.config.prediction_horizon
        
        if available_dates < min_required:
            print(f"⚠️ 資料不足: 可用日期{available_dates}天 < 最少需要{min_required}天")
            return 0
        
        # 可用的序列數量 = 總日期數 - 序列長度 - 預測範圍 + 1
        sequence_count = available_dates - min_required + 1
        print(f"📊 MultiStockDataset: {available_dates}個日期 → {sequence_count}個序列")
        return max(0, sequence_count)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        獲取多股票樣本
        
        Returns:
            sample: {
                'observation': {
                    'price_frame': (n_stocks, seq_len, 5),
                    'fundamental': (10,),
                    'account': (4,)
                },
                'labels': (n_stocks,),
                'metadata': {...}
            }
        """
        start_idx = idx
        end_idx = start_idx + self.config.sequence_length
        label_idx = end_idx + self.config.prediction_horizon - 1
        
        # 獲取日期範圍
        date_range = self.aligned_data.index[start_idx:end_idx]
        target_date = self.aligned_data.index[label_idx] if label_idx < len(self.aligned_data) else None
        
        # 獲取實際配置的其他特徵數量 (66維配置中的51個其他特徵)
        try:
            from models.config.training_config import TrainingConfig
            config = TrainingConfig()
            price_features_count = config.other_features  # 51個其他特徵
        except:
            price_features_count = 51  # 預設值 (66維配置)
        
        # 構建多股票價格框架 - 使用51個其他特徵
        price_frame = np.zeros((len(self.symbols), self.config.sequence_length, price_features_count))
        
        # 獲取實際配置的基本面特徵數量 (15個，66維配置)
        try:
            from models.config.training_config import TrainingConfig
            config = TrainingConfig()
            fundamental_dim = config.fundamental_features  # 15個基本面特徵 (66維配置)
        except:
            fundamental_dim = 15  # 預設值 (66維配置)
        
        fundamental_features = np.zeros(fundamental_dim)
        labels = np.zeros(len(self.symbols))
        
        for i, symbol in enumerate(self.symbols):
            # 確保索引不超出price_frame的邊界
            if i >= len(self.symbols):
                break
                
            if symbol not in self.features_dict:
                continue
            
            features, stock_labels, prices = self.features_dict[symbol]
            
            # 提取該股票在日期範圍內的價格資料
            stock_prices = prices.reindex(date_range, method='ffill').dropna()
            
            if len(stock_prices) > 0:
                # 使用特徵引擎計算完整的51個其他特徵
                from data_pipeline.features import FeatureEngine
                feature_engine = FeatureEngine()
                
                # 計算技術指標特徵 (返回27個特徵: 5個OHLCV + 22個技術指標)
                tech_features = feature_engine.calculate_technical_features(stock_prices)
                
                if not tech_features.empty and tech_features.shape[1] >= 27:
                    # 創建51個其他特徵的數組
                    price_data = np.zeros((len(stock_prices), price_features_count))
                    tech_array = tech_features.to_numpy()  # (seq_len, 27)
                    
                    # 填充前27個特徵（技術特徵）
                    price_data[:, :min(27, price_features_count)] = tech_array[:, :min(27, price_features_count)]
                    
                    # 填充剩餘的24個特徵
                    remaining_features = price_features_count - 27
                    if remaining_features > 0 and len(stock_prices) > 0:
                        close_prices = stock_prices['close'].values
                        if len(close_prices) > 1:
                            # 計算價格變化率
                            price_returns = np.diff(close_prices) / (close_prices[:-1] + 1e-8)
                            price_returns = np.concatenate([[0], price_returns])
                            
                            # 計算移動平均偏差
                            if len(close_prices) >= 5:
                                ma5 = np.convolve(close_prices, np.ones(5)/5, mode='same')
                                ma_deviation = (close_prices - ma5) / (ma5 + 1e-8)
                            else:
                                ma_deviation = np.zeros_like(close_prices)
                            
                            # 填充剩餘特徵 (修復變數名衝突)
                            for j in range(remaining_features):
                                if j % 2 == 0 and j//2 < len(price_returns):
                                    price_data[:, 27 + j] = price_returns[min(j//2, len(price_returns)-1)]
                                elif j % 2 == 1 and j//2 < len(ma_deviation):
                                    price_data[:, 27 + j] = ma_deviation[min(j//2, len(ma_deviation)-1)]
                else:
                    # 如果特徵計算失敗，使用零填充
                    price_data = np.zeros((len(stock_prices), price_features_count))
                
                # 填充到固定長度 (確保索引安全)
                seq_len = min(len(price_data), self.config.sequence_length)
                if i < price_frame.shape[0]:  # 額外的安全檢查
                    price_frame[i, -seq_len:, :] = price_data[-seq_len:]
            
            # 提取標籤
            if target_date and target_date in stock_labels.index:
                label_row = stock_labels.loc[target_date]
                return_col = f'return_{self.config.prediction_horizon}d'
                if return_col in label_row.index:
                    labels[i] = label_row[return_col]
        
        # 聚合基本面特徵 (使用所有股票的平均)
        for symbol in self.symbols:
            if symbol in self.features_dict and end_idx - 1 < len(self.features_dict[symbol][0]):
                features, _, _ = self.features_dict[symbol]
                if end_idx - 1 < len(features):
                    feature_row = features.iloc[end_idx - 1]
                    fundamental_cols = [col for col in feature_row.index if 'fundamental' in col][:fundamental_dim]
                    if fundamental_cols:
                        fundamental_features[:len(fundamental_cols)] += feature_row[fundamental_cols].values
        
        fundamental_features /= len(self.symbols)  # 平均化
        
        # 模擬帳戶狀態
        account_features = np.array([
            np.random.normal(0, 0.01),  # NAV 變化
            np.random.uniform(0, 0.8),  # 持倉比例
            np.random.normal(0, 0.02),  # 未實現損益
            np.random.uniform(0.2, 1.0)  # 風險緩衝
        ])
        
        observation = {
            'price_frame': torch.tensor(price_frame, dtype=torch.float32),
            'fundamental': torch.tensor(fundamental_features, dtype=torch.float32),
            'account': torch.tensor(account_features, dtype=torch.float32)
        }
        
        metadata = {
            'symbols': self.symbols,
            'start_date': str(date_range[0]) if len(date_range) > 0 else None,
            'end_date': str(date_range[-1]) if len(date_range) > 0 else None,
            'target_date': str(target_date) if target_date is not None else None
        }
        
        return {
            'observation': observation,
            'labels': torch.tensor(labels, dtype=torch.float32),
            'metadata': metadata
        }


class TSEDataLoader:
    """TSE Alpha 資料載入器主類"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_engine = FeatureEngine(symbols=config.symbols)
        self.features_dict = {}
        
    def prepare_data(self) -> None:
        """準備訓練資料"""
        logger.info("開始準備訓練資料...")
        
        # 合併所有日期範圍
        all_start_date = min(self.config.train_start_date, self.config.val_start_date, self.config.test_start_date)
        all_end_date = max(self.config.train_end_date, self.config.val_end_date, self.config.test_end_date)
        
        # 處理特徵工程
        self.features_dict = self.feature_engine.process_multiple_symbols(
            symbols=self.config.symbols,
            start_date=all_start_date,
            end_date=all_end_date,
            normalize=self.config.normalize_features
        )
        
        logger.info(f"特徵工程完成，處理了 {len(self.features_dict)} 檔股票")
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """獲取訓練、驗證、測試資料載入器"""
        if not self.features_dict:
            self.prepare_data()
        
        # 按日期分割資料
        train_features = self._filter_by_date(self.config.train_start_date, self.config.train_end_date)
        val_features = self._filter_by_date(self.config.val_start_date, self.config.val_end_date)
        test_features = self._filter_by_date(self.config.test_start_date, self.config.test_end_date)
        
        # 創建資料集
        train_dataset = MultiStockDataset(train_features, self.config, 'train')
        val_dataset = MultiStockDataset(val_features, self.config, 'val')
        test_dataset = MultiStockDataset(test_features, self.config, 'test')
        
        # 創建資料載入器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # 設為0避免多進程問題
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # 設為0避免多進程問題
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # 設為0避免多進程問題
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def _filter_by_date(self, start_date: str, end_date: str) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """按日期範圍過濾資料"""
        filtered_dict = {}
        
        for symbol, (features, labels, prices) in self.features_dict.items():
            # 過濾日期範圍
            mask = (features.index >= start_date) & (features.index <= end_date)
            
            filtered_features = features[mask]
            filtered_labels = labels[mask]
            filtered_prices = prices[mask]
            
            if len(filtered_features) > 0:
                filtered_dict[symbol] = (filtered_features, filtered_labels, filtered_prices)
        
        return filtered_dict
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """獲取特徵統計信息"""
        if not self.features_dict:
            return {}
        
        stats = {
            'n_symbols': len(self.features_dict),
            'symbols': list(self.features_dict.keys()),
            'feature_dims': {},
            'date_ranges': {}
        }
        
        for symbol, (features, labels, prices) in self.features_dict.items():
            stats['feature_dims'][symbol] = features.shape[1]
            stats['date_ranges'][symbol] = {
                'start': features.index.min(),
                'end': features.index.max(),
                'count': len(features)
            }
        
        return stats
    
    def load_data(self, symbols: List[str], start_date: str, end_date: str, split: str = 'train'):
        """
        載入指定股票和日期範圍的資料 (為了相容性添加的方法)
        
        Args:
            symbols: 股票代碼列表
            start_date: 開始日期
            end_date: 結束日期
            split: 資料分割類型 ('train', 'val', 'test')
        
        Returns:
            資料集對象
        """
        # 更新配置中的股票清單
        self.config.symbols = symbols
        
        # 更新特徵引擎的股票清單
        self.feature_engine = FeatureEngine(symbols=symbols)
        
        # 處理特徵工程
        self.features_dict = self.feature_engine.process_multiple_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            normalize=self.config.normalize_features
        )
        
        # 按日期過濾資料
        filtered_features = self._filter_by_date(start_date, end_date)
        
        # 創建資料集
        if split == 'train':
            dataset = MultiStockDataset(filtered_features, self.config, 'train')
        elif split == 'val':
            dataset = MultiStockDataset(filtered_features, self.config, 'val')
        else:
            dataset = MultiStockDataset(filtered_features, self.config, 'test')
        
        return dataset


def test_data_loader():
    """測試資料載入器"""
    print("=== 測試 TSE Alpha 資料載入器 ===")
    
    # 創建配置
    config = DataConfig(
        symbols=['2330', '2317'],
        train_start_date='2024-01-01',
        train_end_date='2024-03-31',
        val_start_date='2024-04-01',
        val_end_date='2024-05-31',
        test_start_date='2024-06-01',
        test_end_date='2024-06-30',
        sequence_length=20,
        batch_size=4
    )
    
    # 創建資料載入器
    data_loader = TSEDataLoader(config)
    
    try:
        # 準備資料
        data_loader.prepare_data()
        
        # 獲取統計信息
        stats = data_loader.get_feature_stats()
        print(f"特徵統計: {stats}")
        
        # 獲取資料載入器
        train_loader, val_loader, test_loader = data_loader.get_dataloaders()
        
        print(f"訓練集: {len(train_loader)} 批次")
        print(f"驗證集: {len(val_loader)} 批次")
        print(f"測試集: {len(test_loader)} 批次")
        
        # 測試一個批次
        for batch in train_loader:
            print("批次樣本:")
            print(f"  觀測形狀:")
            for key, value in batch['observation'].items():
                print(f"    {key}: {value.shape}")
            print(f"  標籤形狀: {batch['labels'].shape}")
            break
        
        print("✅ 資料載入器測試完成")
        
    except Exception as e:
        print(f"❌ 資料載入器測試失敗: {e}")
        import traceback
        traceback.print_exc()


# 為了向後相容，提供別名
TSEAlphaDataLoader = TSEDataLoader

if __name__ == "__main__":
    test_data_loader()