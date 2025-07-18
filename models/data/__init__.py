# models/data/__init__.py
"""
TSE Alpha 資料處理模組

提供完整的資料處理功能，包括：
- 資料庫連接與查詢 (DatabaseManager)
- 資料載入與批次生成 (TSEDataLoader)
- 特徵工程處理 (FeatureEngineer)
- 標籤生成系統 (LabelGenerator)
- 資料品質驗證 (DataValidator)
"""

from .database_manager import DatabaseManager
from .data_loader import TSEDataLoader

__all__ = ['DatabaseManager', 'TSEDataLoader']