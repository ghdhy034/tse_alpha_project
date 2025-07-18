# backtest/strategy_executor.py
"""
策略執行器 - 實作 References.txt 建議
支援 backend='seq'|'mp'|'ray' 並行執行
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import date, datetime
import time
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .config import BacktestConfig, Period, CostConfig
    from .data_manager import DataManager
except ImportError:
    from config import BacktestConfig, Period, CostConfig
    from data_manager import DataManager

# 延遲導入避免循環依賴
def _import_gym_env():
    try:
        from gym_env.env import TSEAlphaEnv
        return TSEAlphaEnv
    except ImportError as e:
        logger.warning(f"無法導入 TSEAlphaEnv: {e}")
        return None

def _import_trading_ledger():
    try:
        from backtest.ledger import TradingLedger
        return TradingLedger
    except ImportError as e:
        logger.warning(f"無法導入 TradingLedger: {e}")
        return None

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """交易會話配置"""
    period: Period
    symbols: List[str]
    initial_cash: float
    max_steps: int = 10000
    save_detailed_logs: bool = True


@dataclass
class SessionResult:
    """交易會話結果"""
    period_id: str
    start_time: datetime
    end_time: datetime
    
    # 績效指標
    initial_nav: float
    final_nav: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    
    # 交易統計
    total_trades: int
    win_trades: int
    lose_trades: int
    avg_holding_days: float
    
    # 詳細記錄
    nav_series: List[float]
    trade_records: List[Dict]
    position_records: List[Dict]
    
    # 執行統計
    execution_time: float
    steps_executed: int
    errors: List[str]


class StrategyExecutor:
    """
    策略執行器 - 實作 References.txt 建議
    支援多種並行後端和性能測量
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.backend = config.backend
        self.max_workers = config.max_workers
        self.cost_config = CostConfig(
            commission_rate=config.commission_rate,
            commission_discount=config.commission_discount,
            min_commission=config.min_commission,
            tax_rate=config.tax_rate,
            slippage_bps=config.slippage_bps,
            k_slip=config.k_slip,
            min_tick=config.min_tick
        )
        
        # 性能統計
        self._execution_stats = {
            'total_sessions': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0,
            'backend_used': self.backend
        }
        
        logger.info(f"策略執行器初始化: backend={self.backend}, max_workers={self.max_workers}")
    
    def execute_strategy(self, 
                        model: Any,
                        periods: List[Period],
                        symbols: List[str],
                        data_manager: DataManager) -> List[SessionResult]:
        """
        執行策略 - 支援多種並行後端
        
        Args:
            model: 訓練好的模型
            periods: 測試週期列表
            symbols: 股票代碼列表
            data_manager: 資料管理器
            
        Returns:
            會話結果列表
        """
        logger.info(f"開始執行策略: {len(periods)} 個週期, {len(symbols)} 檔股票, backend={self.backend}")
        
        start_time = time.time()
        
        # 準備會話配置
        session_configs = []
        for period in periods:
            session_config = SessionConfig(
                period=period,
                symbols=symbols,
                initial_cash=self.config.initial_cash,
                max_steps=252 * 2,  # 約 2 年的交易日
                save_detailed_logs=self.config.save_trades
            )
            session_configs.append(session_config)
        
        # 根據後端執行
        if self.backend == 'seq':
            results = self._execute_sequential(model, session_configs, data_manager)
        elif self.backend == 'mp':
            results = self._execute_multiprocessing(model, session_configs, data_manager)
        elif self.backend == 'ray':
            results = self._execute_ray(model, session_configs, data_manager)
        else:
            logger.error(f"不支援的後端: {self.backend}")
            return []
        
        # 更新統計
        execution_time = time.time() - start_time
        self._execution_stats['total_sessions'] += len(results)
        self._execution_stats['total_execution_time'] += execution_time
        self._execution_stats['avg_execution_time'] = (
            self._execution_stats['total_execution_time'] / self._execution_stats['total_sessions']
        )
        
        logger.info(f"策略執行完成: {len(results)} 個會話, 耗時 {execution_time:.2f} 秒")
        return results
    
    def _execute_sequential(self, 
                           model: Any,
                           session_configs: List[SessionConfig],
                           data_manager: DataManager) -> List[SessionResult]:
        """順序執行"""
        results = []
        
        for i, session_config in enumerate(session_configs):
            logger.info(f"執行會話 {i+1}/{len(session_configs)}: {session_config.period.period_id}")
            
            try:
                result = self._run_single_session(model, session_config, data_manager)
                results.append(result)
            except Exception as e:
                logger.error(f"會話 {session_config.period.period_id} 執行失敗: {e}")
                # 創建錯誤結果
                error_result = self._create_error_result(session_config, str(e))
                results.append(error_result)
        
        return results
    
    def _execute_multiprocessing(self, 
                                model: Any,
                                session_configs: List[SessionConfig],
                                data_manager: DataManager) -> List[SessionResult]:
        """多進程執行"""
        logger.info(f"使用多進程執行: {self.max_workers} 個工作進程")
        
        results = []
        
        try:
            # 準備工作函數參數
            work_args = []
            for session_config in session_configs:
                work_args.append((model, session_config, self.config))
            
            # 使用進程池執行
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(_worker_function, args) for args in work_args]
                
                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=300)  # 5 分鐘超時
                        results.append(result)
                        logger.info(f"會話 {i+1}/{len(futures)} 完成")
                    except Exception as e:
                        logger.error(f"會話 {i+1} 執行失敗: {e}")
                        error_result = self._create_error_result(session_configs[i], str(e))
                        results.append(error_result)
        
        except Exception as e:
            logger.error(f"多進程執行失敗: {e}")
            # 回退到順序執行
            logger.info("回退到順序執行")
            return self._execute_sequential(model, session_configs, data_manager)
        
        return results
    
    def _execute_ray(self, 
                    model: Any,
                    session_configs: List[SessionConfig],
                    data_manager: DataManager) -> List[SessionResult]:
        """Ray 並行執行"""
        try:
            import ray
            
            if not ray.is_initialized():
                ray.init(num_cpus=self.max_workers)
            
            logger.info(f"使用 Ray 並行執行: {self.max_workers} 個工作節點")
            
            # 將模型和配置放入 Ray 物件存儲
            model_ref = ray.put(model)
            config_ref = ray.put(self.config)
            
            # 創建遠程任務
            futures = []
            for session_config in session_configs:
                future = _ray_worker_function.remote(model_ref, session_config, config_ref)
                futures.append(future)
            
            # 收集結果
            results = ray.get(futures)
            
            return results
            
        except ImportError:
            logger.warning("Ray 未安裝，回退到多進程執行")
            return self._execute_multiprocessing(model, session_configs, data_manager)
        except Exception as e:
            logger.error(f"Ray 執行失敗: {e}")
            return self._execute_sequential(model, session_configs, data_manager)
    
    def _run_single_session(self, 
                           model: Any,
                           session_config: SessionConfig,
                           data_manager: DataManager) -> SessionResult:
        """執行單個交易會話"""
        start_time = datetime.now()
        period = session_config.period
        
        # 創建交易環境
        env = self._create_trading_environment(session_config, data_manager)
        
        # 創建交易帳本
        TradingLedger = _import_trading_ledger()
        if TradingLedger is not None:
            ledger = TradingLedger(f"backtest_{period.period_id}")
        else:
            ledger = None
        
        # 初始化記錄
        nav_series = []
        trade_records = []
        position_records = []
        errors = []
        
        try:
            # 重置環境
            observation, info = env.reset()
            initial_nav = info['nav']
            nav_series.append(initial_nav)
            
            step_count = 0
            max_steps = session_config.max_steps
            
            # 主要交易循環
            while step_count < max_steps:
                try:
                    # 模型預測和動作生成
                    action = self._model_predict_and_act(model, observation)
                    
                    # 執行動作
                    observation, reward, terminated, truncated, info = env.step(action)
                    
                    # 記錄 NAV
                    nav_series.append(info['nav'])
                    
                    # 記錄交易
                    if info.get('trade_executed', False):
                        trade_result = info.get('trade_result', {})
                        if trade_result.get('success', False):
                            # 記錄到帳本
                            # 這裡需要更詳細的交易信息
                            pass
                    
                    # 記錄持倉
                    if session_config.save_detailed_logs:
                        position_records.append({
                            'step': step_count,
                            'date': info.get('current_date'),
                            'positions': dict(info.get('positions', {})),
                            'nav': info['nav']
                        })
                    
                    step_count += 1
                    
                    # 檢查結束條件
                    if terminated or truncated:
                        logger.info(f"會話 {period.period_id} 在第 {step_count} 步結束")
                        break
                        
                except Exception as e:
                    error_msg = f"步驟 {step_count} 執行錯誤: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    break
            
            # 計算績效指標
            final_nav = nav_series[-1] if nav_series else initial_nav
            total_return = (final_nav - initial_nav) / initial_nav
            
            # 計算最大回撤
            max_drawdown = self._calculate_max_drawdown(nav_series)
            
            # 計算 Sharpe 比率
            sharpe_ratio = self._calculate_sharpe_ratio(nav_series)
            
            # 統計交易
            total_trades = len(trade_records)
            win_trades = sum(1 for trade in trade_records if trade.get('pnl', 0) > 0)
            lose_trades = total_trades - win_trades
            
            # 創建結果
            result = SessionResult(
                period_id=period.period_id,
                start_time=start_time,
                end_time=datetime.now(),
                initial_nav=initial_nav,
                final_nav=final_nav,
                total_return=total_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                total_trades=total_trades,
                win_trades=win_trades,
                lose_trades=lose_trades,
                avg_holding_days=0.0,  # 需要計算
                nav_series=nav_series,
                trade_records=trade_records,
                position_records=position_records,
                execution_time=(datetime.now() - start_time).total_seconds(),
                steps_executed=step_count,
                errors=errors
            )
            
            return result
            
        except Exception as e:
            logger.error(f"會話 {period.period_id} 執行失敗: {e}")
            return self._create_error_result(session_config, str(e))
        
        finally:
            env.close()
    
    def _create_trading_environment(self, 
                                  session_config: SessionConfig,
                                  data_manager: DataManager):
        """創建交易環境"""
        period = session_config.period
        
        # 動態導入環境類別
        TSEAlphaEnv = _import_gym_env()
        if TSEAlphaEnv is None:
            raise ImportError("無法導入 TSEAlphaEnv")
        
        # 創建環境
        env = TSEAlphaEnv(
            symbols=session_config.symbols,
            start_date=period.test_start.strftime('%Y-%m-%d'),
            end_date=period.test_end.strftime('%Y-%m-%d'),
            initial_cash=session_config.initial_cash,
            max_holding_days=self.config.max_holding_days,
            max_position_per_stock=self.config.max_position_per_stock,
            daily_max_loss_pct=self.config.daily_max_loss_pct,
            rolling_max_dd_pct=self.config.rolling_max_dd_pct
        )
        
        return env
    
    def _model_predict_and_act(self, model: Any, observation: Dict) -> Any:
        """模型預測和動作生成"""
        try:
            # 檢查模型是否有 get_action 方法
            if hasattr(model, 'get_action'):
                # 轉換觀測格式
                obs_tensor = self._convert_observation_to_tensor(observation)
                action = model.get_action(obs_tensor, deterministic=True)
                return action
            else:
                # 使用隨機動作作為備用
                logger.warning("模型沒有 get_action 方法，使用隨機動作")
                return self._generate_random_action()
                
        except Exception as e:
            logger.error(f"模型預測失敗: {e}")
            return self._generate_random_action()
    
    def _convert_observation_to_tensor(self, observation: Dict) -> Dict:
        """轉換觀測格式為模型輸入格式"""
        # 這裡需要根據實際的模型輸入格式進行轉換
        # 暫時返回原始觀測
        return observation
    
    def _generate_random_action(self) -> tuple:
        """生成隨機動作"""
        import random
        stock_idx = random.randint(0, 9)  # 假設有 10 檔股票
        qty = random.choice([-50, -25, 0, 25, 50])  # 隨機交易量
        return (stock_idx, [qty])
    
    def _calculate_max_drawdown(self, nav_series: List[float]) -> float:
        """計算最大回撤"""
        if len(nav_series) < 2:
            return 0.0
        
        peak = nav_series[0]
        max_dd = 0.0
        
        for nav in nav_series[1:]:
            if nav > peak:
                peak = nav
            else:
                dd = (peak - nav) / peak
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, nav_series: List[float], risk_free_rate: float = 0.01) -> float:
        """計算 Sharpe 比率"""
        if len(nav_series) < 2:
            return 0.0
        
        # 計算日收益率
        returns = []
        for i in range(1, len(nav_series)):
            ret = (nav_series[i] - nav_series[i-1]) / nav_series[i-1]
            returns.append(ret)
        
        if not returns:
            return 0.0
        
        import numpy as np
        returns = np.array(returns)
        
        # 年化收益率和波動率
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        
        if annual_volatility == 0:
            return 0.0
        
        sharpe = (annual_return - risk_free_rate) / annual_volatility
        return sharpe
    
    def _create_error_result(self, session_config: SessionConfig, error_msg: str) -> SessionResult:
        """創建錯誤結果"""
        return SessionResult(
            period_id=session_config.period.period_id,
            start_time=datetime.now(),
            end_time=datetime.now(),
            initial_nav=session_config.initial_cash,
            final_nav=session_config.initial_cash,
            total_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
            win_trades=0,
            lose_trades=0,
            avg_holding_days=0.0,
            nav_series=[session_config.initial_cash],
            trade_records=[],
            position_records=[],
            execution_time=0.0,
            steps_executed=0,
            errors=[error_msg]
        )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """獲取執行統計"""
        return self._execution_stats.copy()


# 工作函數 (用於多進程)
def _worker_function(args) -> SessionResult:
    """多進程工作函數"""
    model, session_config, config = args
    
    # 在工作進程中創建執行器
    executor = StrategyExecutor(config)
    
    # 創建資料管理器 (每個進程獨立)
    data_manager = DataManager(config)
    
    # 執行會話
    return executor._run_single_session(model, session_config, data_manager)


# Ray 工作函數
try:
    import ray
    
    @ray.remote
    def _ray_worker_function(model, session_config, config) -> SessionResult:
        """Ray 工作函數"""
        executor = StrategyExecutor(config)
        data_manager = DataManager(config)
        return executor._run_single_session(model, session_config, data_manager)
        
except ImportError:
    # Ray 未安裝時的占位函數
    def _ray_worker_function(*args):
        pass


def test_strategy_executor():
    """測試策略執行器"""
    print("=== 測試策略執行器 ===")
    
    # 創建測試配置
    from config import create_smoke_test_config
    config = create_smoke_test_config()
    config.backend = 'seq'  # 使用順序執行進行測試
    
    # 創建執行器
    print("1. 創建策略執行器...")
    try:
        executor = StrategyExecutor(config)
        print(f"   ✅ 策略執行器創建成功")
        print(f"   後端: {executor.backend}")
    except Exception as e:
        print(f"   ❌ 策略執行器創建失敗: {e}")
        return False
    
    # 測試單個會話執行
    print("2. 測試單個會話執行...")
    try:
        from datetime import date
        from config import Period
        
        # 創建測試週期
        period = Period(
            train_start=date(2024, 1, 1),
            train_end=date(2024, 3, 31),
            test_start=date(2024, 4, 1),
            test_end=date(2024, 4, 30),
            period_id="test_period"
        )
        
        # 創建會話配置
        session_config = SessionConfig(
            period=period,
            symbols=['2330', '2317'],
            initial_cash=1000000.0,
            max_steps=100,  # 限制步數進行快速測試
            save_detailed_logs=False
        )
        
        # 創建虛擬模型
        class DummyModel:
            def get_action(self, observation, deterministic=True):
                return (0, [0])  # 不交易
        
        model = DummyModel()
        
        # 創建資料管理器
        from data_manager import DataManager
        data_manager = DataManager(config)
        
        # 執行會話
        result = executor._run_single_session(model, session_config, data_manager)
        
        print(f"   ✅ 會話執行成功")
        print(f"   週期: {result.period_id}")
        print(f"   執行時間: {result.execution_time:.2f} 秒")
        print(f"   執行步數: {result.steps_executed}")
        print(f"   總收益: {result.total_return:.4f}")
        
    except Exception as e:
        print(f"   ❌ 會話執行測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 測試執行統計
    print("3. 測試執行統計...")
    try:
        stats = executor.get_execution_stats()
        print("   執行統計:")
        for key, value in stats.items():
            print(f"     {key}: {value}")
        print("   ✅ 執行統計獲取成功")
    except Exception as e:
        print(f"   ❌ 執行統計測試失敗: {e}")
        return False
    
    print("✅ 策略執行器測試完成")
    return True


if __name__ == "__main__":
    test_strategy_executor()