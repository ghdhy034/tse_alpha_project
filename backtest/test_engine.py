# backtest/test_engine.py
"""
回測引擎測試套件 - 實作 References.txt 建議
包含 pytest-benchmark 性能測試和 Smoke Test
"""
import pytest
import sys
from pathlib import Path
from datetime import date, datetime
import time
import numpy as np
import pandas as pd

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))

from config import BacktestConfig, create_smoke_test_config
from engine import BacktestEngine, create_dummy_model
from time_manager import TimeSeriesManager
from data_manager import DataManager
from strategy_executor import StrategyExecutor
from metrics_engine import MetricsEngine


class TestBacktestEngine:
    """回測引擎測試類別"""
    
    @pytest.fixture
    def smoke_config(self):
        """Smoke Test 配置"""
        return create_smoke_test_config()
    
    @pytest.fixture
    def engine(self, smoke_config):
        """回測引擎實例"""
        return BacktestEngine(smoke_config)
    
    @pytest.fixture
    def dummy_model(self):
        """虛擬模型"""
        return create_dummy_model()
    
    @pytest.mark.smoke
    @pytest.mark.backtest
    def test_smoke_test_performance(self, engine, dummy_model, benchmark):
        """
        Smoke Test 性能測試 - References.txt 建議
        目標：單 period (3 m) random agent ≤ 60 s on CPU
        """
        def run_smoke_test():
            return engine.smoke_test(dummy_model)
        
        # 使用 pytest-benchmark 測量性能
        result = benchmark(run_smoke_test)
        
        # 驗證性能要求
        assert result['success'], f"Smoke Test 失敗: {result.get('errors', [])}"
        assert result['performance_ok'], f"性能未達標: {result['execution_time']:.2f}s > {result['target_time']}s"
        assert result['execution_time'] <= 60, "執行時間超過 60 秒限制"
    
    @pytest.mark.smoke
    def test_engine_initialization(self, smoke_config):
        """測試引擎初始化"""
        engine = BacktestEngine(smoke_config)
        
        assert engine.config == smoke_config
        assert isinstance(engine.time_manager, TimeSeriesManager)
        assert isinstance(engine.data_manager, DataManager)
        assert isinstance(engine.strategy_executor, StrategyExecutor)
        assert isinstance(engine.metrics_engine, MetricsEngine)
    
    @pytest.mark.smoke
    def test_basic_backtest(self, engine, dummy_model):
        """測試基本回測功能"""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)  # 1 個月
        symbols = ['2330', '2317']
        
        result = engine.run_backtest(
            model=dummy_model,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols
        )
        
        # 基本驗證
        assert len(result.errors) == 0, f"回測有錯誤: {result.errors}"
        assert result.total_periods > 0, "沒有生成有效週期"
        assert len(result.session_results) > 0, "沒有會話結果"
        assert result.execution_time > 0, "執行時間無效"
        
        # 績效指標驗證
        metrics = result.performance_metrics
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert metrics.max_drawdown >= 0, "最大回撤不能為負數"
    
    def test_walk_forward_validation(self, engine):
        """測試 Walk-forward 驗證"""
        def model_factory():
            return create_dummy_model()
        
        start_date = date(2024, 1, 1)
        end_date = date(2024, 2, 29)
        
        wf_result = engine.walk_forward_validation(
            model_factory=model_factory,
            start_date=start_date,
            end_date=end_date
        )
        
        # 驗證結果結構
        assert wf_result.backtest_result is not None
        assert len(wf_result.period_performance) > 0
        assert 'return_std' in wf_result.performance_stability
        assert 'avg_return' in wf_result.out_of_sample_metrics
        assert wf_result.retrain_count > 0
    
    def test_error_handling(self, smoke_config):
        """測試錯誤處理"""
        engine = BacktestEngine(smoke_config)
        
        # 測試無效日期範圍
        start_date = date(2024, 12, 31)
        end_date = date(2024, 1, 1)  # 結束日期早於開始日期
        
        result = engine.run_backtest(
            model=create_dummy_model(),
            start_date=start_date,
            end_date=end_date
        )
        
        # 應該有錯誤
        assert len(result.errors) > 0, "應該檢測到日期範圍錯誤"
    
    @pytest.mark.benchmark
    def test_performance_benchmark(self, engine, dummy_model, benchmark):
        """性能基準測試"""
        def run_backtest():
            return engine.run_backtest(
                model=dummy_model,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                symbols=['2330']
            )
        
        result = benchmark(run_backtest)
        
        # 驗證結果有效性
        assert len(result.errors) == 0
        assert result.total_periods > 0


class TestTimeSeriesManager:
    """時間序列管理器測試"""
    
    @pytest.fixture
    def time_manager(self):
        config = create_smoke_test_config()
        return TimeSeriesManager(config)
    
    def test_period_generation(self, time_manager):
        """測試週期生成"""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 6, 30)
        
        periods = time_manager.generate_walk_forward_periods(start_date, end_date)
        
        assert len(periods) > 0, "應該生成至少一個週期"
        
        # 驗證週期結構
        for period in periods:
            assert period.train_start < period.train_end
            assert period.test_start < period.test_end
            assert period.train_end < period.test_start
    
    def test_period_validation(self, time_manager):
        """測試週期驗證 - References.txt 建議的標籤外溢檢查"""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)
        
        periods = time_manager.generate_walk_forward_periods(start_date, end_date)
        is_valid = time_manager.validate_periods(periods)
        
        assert is_valid, "週期驗證應該通過"
        
        # 檢查標籤外溢保護
        for period in periods:
            label_horizon = time_manager.config.horizon_days
            last_sample_date = period.train_end
            last_label_date = last_sample_date + pd.Timedelta(days=label_horizon)
            
            assert last_label_date <= period.test_start, f"週期 {period.period_id} 存在標籤外溢"


class TestDataManager:
    """資料管理器測試"""
    
    @pytest.fixture
    def data_manager(self):
        config = create_smoke_test_config()
        config.cache_mode = 'none'  # 測試時不使用快取
        return DataManager(config)
    
    def test_data_loading(self, data_manager):
        """測試資料載入"""
        symbol = '2330'
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        # 測試日線資料載入
        daily_data = data_manager.get_stock_data(symbol, start_date, end_date, 'daily')
        
        # 如果有資料，驗證格式
        if daily_data is not None:
            assert isinstance(daily_data, pd.DataFrame)
            assert not daily_data.empty
            expected_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in expected_columns:
                assert col in daily_data.columns, f"缺少欄位: {col}"
    
    def test_cache_functionality(self):
        """測試快取功能"""
        config = create_smoke_test_config()
        config.cache_mode = 'duckdb'
        data_manager = DataManager(config)
        
        # 測試快取統計
        stats = data_manager.get_cache_stats()
        
        assert 'cache_mode' in stats
        assert 'hits' in stats
        assert 'misses' in stats
        assert stats['cache_mode'] == 'duckdb'


class TestMetricsEngine:
    """績效計算引擎測試"""
    
    @pytest.fixture
    def metrics_engine(self):
        config = create_smoke_test_config()
        return MetricsEngine(config)
    
    def test_annualization_factors(self, metrics_engine):
        """測試年化因子 - References.txt 建議"""
        expected_factors = {
            'D': 252,
            'W': 52,
            'M': 12,
            'Q': 4,
            'Y': 1
        }
        
        for freq, expected_factor in expected_factors.items():
            actual_factor = metrics_engine.annualization_factors[freq]
            assert actual_factor == expected_factor, f"{freq} 頻率年化因子錯誤"
    
    def test_max_drawdown_calculation(self, metrics_engine):
        """測試最大回撤計算 - 使用已知數值驗證"""
        # 創建已知最大回撤的序列
        test_nav = pd.Series([100, 110, 120, 90, 80, 100, 110], 
                           index=pd.date_range('2024-01-01', periods=7, freq='D'))
        
        # 手算最大回撤：從 120 跌到 80，回撤 = (120-80)/120 = 1/3
        expected_max_dd = (120 - 80) / 120
        calculated_max_dd = metrics_engine._calculate_max_drawdown_simple(test_nav)
        
        # 使用 pandas.testing 進行精確比較
        import pandas.testing as pdt
        
        expected_series = pd.Series([expected_max_dd], index=['max_drawdown'])
        calculated_series = pd.Series([calculated_max_dd], index=['max_drawdown'])
        
        pdt.assert_series_equal(expected_series, calculated_series, rtol=1e-6)
    
    def test_performance_metrics_calculation(self, metrics_engine):
        """測試績效指標計算"""
        # 創建虛擬會話結果
        from strategy_executor import SessionResult
        
        session = SessionResult(
            period_id='test',
            start_time=datetime.now(),
            end_time=datetime.now(),
            initial_nav=1000000,
            final_nav=1100000,
            total_return=0.1,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            total_trades=10,
            win_trades=6,
            lose_trades=4,
            avg_holding_days=5.0,
            nav_series=[1000000, 1050000, 1100000],
            trade_records=[],
            position_records=[],
            execution_time=60.0,
            steps_executed=100,
            errors=[]
        )
        
        metrics = metrics_engine.calculate_performance_metrics([session])
        
        # 驗證指標類型和合理性
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert metrics.max_drawdown >= 0
        assert metrics.total_trades >= 0


# 性能測試標記
pytestmark = pytest.mark.backtest


def test_integration_smoke():
    """整合 Smoke Test - 可獨立執行"""
    config = create_smoke_test_config()
    engine = BacktestEngine(config)
    model = create_dummy_model()
    
    start_time = time.time()
    smoke_result = engine.smoke_test(model)
    execution_time = time.time() - start_time
    
    print(f"\n=== 整合 Smoke Test 結果 ===")
    print(f"執行時間: {execution_time:.2f}s")
    print(f"目標時間: {smoke_result['target_time']}s")
    print(f"成功: {smoke_result['success']}")
    print(f"性能達標: {smoke_result['performance_ok']}")
    
    if smoke_result.get('errors'):
        print("錯誤:")
        for error in smoke_result['errors']:
            print(f"  - {error}")
    
    assert smoke_result['success'], "Smoke Test 應該成功"
    assert smoke_result['performance_ok'], "性能應該達標"


if __name__ == "__main__":
    # 直接執行整合測試
    test_integration_smoke()
    print("✅ 所有測試通過")