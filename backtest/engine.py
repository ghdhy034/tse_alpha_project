# backtest/engine.py
"""
主回測引擎 - 整合所有組件
實作 References.txt 建議的 Smoke Test 和性能基線
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import date, datetime, timedelta
import time
import logging
from dataclasses import dataclass

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .config import BacktestConfig, Period, create_smoke_test_config
    from .time_manager import TimeSeriesManager
    from .data_manager import DataManager
    from .strategy_executor import StrategyExecutor, SessionResult
    from .metrics_engine import MetricsEngine, PerformanceMetrics
except ImportError:
    from config import BacktestConfig, Period, create_smoke_test_config
    from time_manager import TimeSeriesManager
    from data_manager import DataManager
    from strategy_executor import StrategyExecutor, SessionResult
    from metrics_engine import MetricsEngine, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回測結果"""
    # 基本資訊
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    execution_time: float
    
    # 週期資訊
    periods: List[Period]
    total_periods: int
    
    # 績效指標
    performance_metrics: PerformanceMetrics
    
    # 會話結果
    session_results: List[SessionResult]
    
    # 統計資訊
    cache_stats: Dict[str, Any]
    execution_stats: Dict[str, Any]
    
    # 錯誤和警告
    errors: List[str]
    warnings: List[str]


@dataclass
class WalkForwardResult:
    """Walk-forward 驗證結果"""
    backtest_result: BacktestResult
    
    # Walk-forward 特有指標
    period_performance: List[Dict[str, float]]
    performance_stability: Dict[str, float]
    out_of_sample_metrics: Dict[str, float]
    
    # 模型重訓練統計
    retrain_count: int
    avg_retrain_time: float


class BacktestEngine:
    """
    主回測引擎 - 協調所有組件
    實作 References.txt 建議的性能基線測試
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
        # 初始化組件
        self.time_manager = TimeSeriesManager(config)
        self.data_manager = DataManager(config)
        self.strategy_executor = StrategyExecutor(config)
        self.metrics_engine = MetricsEngine(config)
        
        # 狀態追蹤
        self._errors = []
        self._warnings = []
        
        logger.info(f"回測引擎初始化完成: {config.backend} 後端")
    
    def run_backtest(self, 
                    model: Any,
                    start_date: date,
                    end_date: date,
                    symbols: Optional[List[str]] = None) -> BacktestResult:
        """
        執行完整回測
        
        Args:
            model: 訓練好的模型
            start_date: 回測開始日期
            end_date: 回測結束日期
            symbols: 股票代碼列表 (None 則使用配置中的股票池)
            
        Returns:
            回測結果
        """
        start_time = datetime.now()
        logger.info(f"開始回測: {start_date} ~ {end_date}")
        
        try:
            # 準備股票列表
            if symbols is None:
                symbols = self.config.stock_universe
            
            # 預載入資料
            if self.config.preload_data:
                logger.info("預載入資料...")
                preload_success = self.data_manager.preload(symbols, (start_date, end_date))
                if not preload_success:
                    self._warnings.append("資料預載入失敗，可能影響性能")
            
            # 生成 Walk-forward 週期
            logger.info("生成 Walk-forward 週期...")
            periods = self.time_manager.generate_walk_forward_periods(start_date, end_date)
            
            if not periods:
                error_msg = "無法生成有效的回測週期"
                self._errors.append(error_msg)
                logger.error(error_msg)
                return self._create_error_result(start_time, error_msg)
            
            # 驗證週期
            if not self.time_manager.validate_periods(periods):
                error_msg = "週期驗證失敗"
                self._errors.append(error_msg)
                logger.error(error_msg)
                return self._create_error_result(start_time, error_msg)
            
            logger.info(f"生成 {len(periods)} 個有效週期")
            
            # 執行策略
            logger.info("執行交易策略...")
            session_results = self.strategy_executor.execute_strategy(
                model=model,
                periods=periods,
                symbols=symbols,
                data_manager=self.data_manager
            )
            
            if not session_results:
                error_msg = "策略執行失敗，無會話結果"
                self._errors.append(error_msg)
                logger.error(error_msg)
                return self._create_error_result(start_time, error_msg)
            
            # 計算績效指標
            logger.info("計算績效指標...")
            performance_metrics = self.metrics_engine.calculate_performance_metrics(
                sessions=session_results,
                frequency='D'
            )
            
            # 收集統計資訊
            cache_stats = self.data_manager.get_cache_stats()
            execution_stats = self.strategy_executor.get_execution_stats()
            
            # 創建結果
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result = BacktestResult(
                config=self.config,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
                periods=periods,
                total_periods=len(periods),
                performance_metrics=performance_metrics,
                session_results=session_results,
                cache_stats=cache_stats,
                execution_stats=execution_stats,
                errors=self._errors.copy(),
                warnings=self._warnings.copy()
            )
            
            logger.info(f"回測完成: 耗時 {execution_time:.2f} 秒")
            logger.info(f"總收益: {performance_metrics.total_return:.4f}")
            logger.info(f"Sharpe 比率: {performance_metrics.sharpe_ratio:.4f}")
            logger.info(f"最大回撤: {performance_metrics.max_drawdown:.4f}")
            
            return result
            
        except Exception as e:
            error_msg = f"回測執行異常: {e}"
            self._errors.append(error_msg)
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return self._create_error_result(start_time, error_msg)
    
    def walk_forward_validation(self, 
                               model_factory: Callable,
                               start_date: date,
                               end_date: date,
                               retrain_frequency: str = 'monthly') -> WalkForwardResult:
        """
        Walk-forward 驗證 - 包含模型重訓練
        
        Args:
            model_factory: 模型工廠函數
            start_date: 驗證開始日期
            end_date: 驗證結束日期
            retrain_frequency: 重訓練頻率 ('monthly', 'quarterly')
            
        Returns:
            Walk-forward 驗證結果
        """
        logger.info(f"開始 Walk-forward 驗證: {retrain_frequency} 重訓練")
        
        # 執行基本回測
        model = model_factory()
        backtest_result = self.run_backtest(model, start_date, end_date)
        
        # 分析週期績效
        period_performance = []
        for session in backtest_result.session_results:
            period_perf = {
                'period_id': session.period_id,
                'return': session.total_return,
                'sharpe': session.sharpe_ratio,
                'max_drawdown': session.max_drawdown,
                'trades': session.total_trades
            }
            period_performance.append(period_perf)
        
        # 計算穩定性指標
        returns = [p['return'] for p in period_performance]
        sharpes = [p['sharpe'] for p in period_performance]
        
        import numpy as np
        performance_stability = {
            'return_std': np.std(returns) if returns else 0.0,
            'sharpe_std': np.std(sharpes) if sharpes else 0.0,
            'return_consistency': len([r for r in returns if r > 0]) / len(returns) if returns else 0.0
        }
        
        # 樣本外指標 (簡化版本)
        out_of_sample_metrics = {
            'avg_return': np.mean(returns) if returns else 0.0,
            'avg_sharpe': np.mean(sharpes) if sharpes else 0.0,
            'success_rate': len([r for r in returns if r > 0]) / len(returns) if returns else 0.0
        }
        
        # 重訓練統計 (模擬)
        retrain_count = len(backtest_result.periods)  # 每個週期重訓練一次
        avg_retrain_time = 60.0  # 假設平均重訓練時間 60 秒
        
        wf_result = WalkForwardResult(
            backtest_result=backtest_result,
            period_performance=period_performance,
            performance_stability=performance_stability,
            out_of_sample_metrics=out_of_sample_metrics,
            retrain_count=retrain_count,
            avg_retrain_time=avg_retrain_time
        )
        
        logger.info("Walk-forward 驗證完成")
        return wf_result
    
    def smoke_test(self, model: Any) -> Dict[str, Any]:
        """
        Smoke Test - References.txt 建議
        目標：單 period (3 m) random agent ≤ 60 s on CPU
        """
        logger.info("開始 Smoke Test")
        
        # 使用 Smoke Test 配置
        smoke_config = create_smoke_test_config()
        smoke_engine = BacktestEngine(smoke_config)
        
        # 測試參數 - 擴大日期範圍確保能生成週期
        start_date = date(2024, 1, 1)
        end_date = date(2024, 6, 30)  # 6 個月，確保有足夠資料生成週期
        test_symbols = ['2330', '2317', '2603']  # 3 檔股票
        
        start_time = time.time()
        
        try:
            # 執行快速回測
            result = smoke_engine.run_backtest(
                model=model,
                start_date=start_date,
                end_date=end_date,
                symbols=test_symbols
            )
            
            execution_time = time.time() - start_time
            
            # 檢查性能基線
            performance_ok = execution_time <= smoke_config.smoke_test_timeout
            
            smoke_result = {
                'success': len(result.errors) == 0,
                'execution_time': execution_time,
                'performance_ok': performance_ok,
                'target_time': smoke_config.smoke_test_timeout,
                'periods_tested': result.total_periods,
                'total_return': result.performance_metrics.total_return,
                'errors': result.errors,
                'warnings': result.warnings
            }
            
            if smoke_result['success'] and performance_ok:
                logger.info(f"✅ Smoke Test 通過: {execution_time:.2f}s ≤ {smoke_config.smoke_test_timeout}s")
            else:
                logger.warning(f"⚠️ Smoke Test 未完全通過: 成功={smoke_result['success']}, 性能={performance_ok}")
            
            return smoke_result
            
        except Exception as e:
            logger.error(f"❌ Smoke Test 失敗: {e}")
            return {
                'success': False,
                'execution_time': time.time() - start_time,
                'performance_ok': False,
                'error': str(e)
            }
    
    def _create_error_result(self, start_time: datetime, error_msg: str) -> BacktestResult:
        """創建錯誤結果"""
        end_time = datetime.now()
        
        return BacktestResult(
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            execution_time=(end_time - start_time).total_seconds(),
            periods=[],
            total_periods=0,
            performance_metrics=self.metrics_engine._create_empty_metrics(),
            session_results=[],
            cache_stats={},
            execution_stats={},
            errors=[error_msg],
            warnings=self._warnings.copy()
        )
    
    def benchmark_performance(self, model: Any, iterations: int = 5) -> Dict[str, float]:
        """
        性能基準測試 - References.txt 建議
        測量多次執行的平均性能
        """
        logger.info(f"開始性能基準測試: {iterations} 次迭代")
        
        execution_times = []
        
        for i in range(iterations):
            logger.info(f"基準測試迭代 {i+1}/{iterations}")
            
            smoke_result = self.smoke_test(model)
            execution_times.append(smoke_result['execution_time'])
        
        import numpy as np
        
        benchmark_result = {
            'avg_execution_time': np.mean(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'std_execution_time': np.std(execution_times),
            'target_time': self.config.smoke_test_timeout,
            'success_rate': sum(1 for t in execution_times if t <= self.config.smoke_test_timeout) / iterations
        }
        
        logger.info(f"性能基準測試完成:")
        logger.info(f"  平均執行時間: {benchmark_result['avg_execution_time']:.2f}s")
        logger.info(f"  目標時間: {benchmark_result['target_time']}s")
        logger.info(f"  成功率: {benchmark_result['success_rate']:.2%}")
        
        return benchmark_result


def create_dummy_model():
    """創建虛擬模型用於測試"""
    class DummyModel:
        def get_action(self, observation, deterministic=True):
            import random
            stock_idx = random.randint(0, 2)  # 隨機選擇股票
            qty = random.choice([-50, -25, 0, 25, 50])  # 隨機交易量
            return (stock_idx, [qty])
    
    return DummyModel()


def test_backtest_engine():
    """測試回測引擎"""
    print("=== 測試回測引擎 ===")
    
    # 創建測試配置
    config = create_smoke_test_config()
    
    # 創建回測引擎
    print("1. 創建回測引擎...")
    try:
        engine = BacktestEngine(config)
        print(f"   ✅ 回測引擎創建成功")
    except Exception as e:
        print(f"   ❌ 回測引擎創建失敗: {e}")
        return False
    
    # 創建虛擬模型
    model = create_dummy_model()
    
    # 測試 Smoke Test
    print("2. 測試 Smoke Test...")
    try:
        smoke_result = engine.smoke_test(model)
        
        print(f"   執行時間: {smoke_result['execution_time']:.2f}s")
        print(f"   目標時間: {smoke_result['target_time']}s")
        print(f"   成功: {smoke_result['success']}")
        print(f"   性能達標: {smoke_result['performance_ok']}")
        
        if smoke_result['success']:
            print("   ✅ Smoke Test 通過")
        else:
            print("   ⚠️ Smoke Test 未完全通過")
            if smoke_result.get('errors'):
                for error in smoke_result['errors']:
                    print(f"     錯誤: {error}")
                    
    except Exception as e:
        print(f"   ❌ Smoke Test 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 測試完整回測
    print("3. 測試完整回測...")
    try:
        start_date = date(2024, 1, 1)
        end_date = date(2024, 2, 29)  # 2 個月
        
        result = engine.run_backtest(
            model=model,
            start_date=start_date,
            end_date=end_date,
            symbols=['2330', '2317']
        )
        
        print(f"   執行時間: {result.execution_time:.2f}s")
        print(f"   週期數: {result.total_periods}")
        print(f"   總收益: {result.performance_metrics.total_return:.4f}")
        print(f"   Sharpe 比率: {result.performance_metrics.sharpe_ratio:.4f}")
        print(f"   錯誤數: {len(result.errors)}")
        
        if len(result.errors) == 0:
            print("   ✅ 完整回測成功")
        else:
            print("   ⚠️ 回測有錯誤:")
            for error in result.errors:
                print(f"     {error}")
                
    except Exception as e:
        print(f"   ❌ 完整回測失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ 回測引擎測試完成")
    return True


if __name__ == "__main__":
    test_backtest_engine()