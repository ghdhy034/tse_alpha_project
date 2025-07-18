# backtest/__init__.py
"""
回測引擎模組 - 完整的 Walk-forward 回測系統
實作 References.txt 建議的所有功能
"""
__version__ = "1.0.0"

# 安全導入，避免循環依賴
try:
    from .config import BacktestConfig, Period, CostConfig, create_default_config, create_smoke_test_config
    from .time_manager import TimeSeriesManager
    from .data_manager import DataManager
    from .strategy_executor import StrategyExecutor, SessionResult, SessionConfig
    from .metrics_engine import MetricsEngine, PerformanceMetrics, RiskMetrics, TradingMetrics
    from .engine import BacktestEngine, BacktestResult, WalkForwardResult, create_dummy_model
    
    _import_success = True
except ImportError as e:
    print(f"警告: 回測模組導入失敗: {e}")
    _import_success = False

# 只有在導入成功時才定義 __all__
if _import_success:
    __all__ = [
        # 配置類別
        'BacktestConfig', 'Period', 'CostConfig',
        'create_default_config', 'create_smoke_test_config',
        
        # 核心組件
        'TimeSeriesManager', 'DataManager', 'StrategyExecutor', 'MetricsEngine',
        
        # 主引擎
        'BacktestEngine',
        
        # 結果類別
        'BacktestResult', 'WalkForwardResult', 'SessionResult', 'SessionConfig',
        'PerformanceMetrics', 'RiskMetrics', 'TradingMetrics',
        
        # 工具函數
        'create_dummy_model'
    ]
else:
    __all__ = []