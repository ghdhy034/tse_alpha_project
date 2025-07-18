# backtest/metrics_engine.py
"""
績效計算引擎 - 實作 References.txt 建議
年化因子由 freq 動態決定，包含 pandas.testing 單元測試
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .config import BacktestConfig
    from .strategy_executor import SessionResult
except ImportError:
    from config import BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """績效指標"""
    # 收益指標
    total_return: float
    annual_return: float
    cumulative_return: float
    
    # 風險指標
    annual_volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    expected_shortfall: float
    
    # 風險調整收益指標
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    information_ratio: float
    
    # 交易統計
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_holding_days: float
    
    # 穩定性指標
    best_month: float
    worst_month: float
    positive_months: int
    negative_months: int
    
    # 時間相關
    start_date: date
    end_date: date
    total_days: int
    trading_days: int
    frequency: str


@dataclass
class RiskMetrics:
    """風險指標"""
    max_drawdown: float
    max_drawdown_start: date
    max_drawdown_end: date
    max_drawdown_duration: int
    
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    
    downside_deviation: float
    upside_deviation: float
    
    beta: Optional[float] = None
    alpha: Optional[float] = None
    tracking_error: Optional[float] = None


@dataclass
class TradingMetrics:
    """交易統計指標"""
    total_trades: int
    long_trades: int
    short_trades: int
    
    win_trades: int
    lose_trades: int
    win_rate: float
    
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    profit_factor: float
    expectancy: float
    
    avg_holding_days: float
    max_holding_days: int
    min_holding_days: int
    
    avg_trade_size: float
    turnover_rate: float


class MetricsEngine:
    """
    績效計算引擎 - 實作 References.txt 建議
    年化因子由 freq 動態決定
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
        # 年化因子映射 - References.txt 建議
        self.annualization_factors = {
            'D': 252,    # 日頻率
            'W': 52,     # 週頻率  
            'M': 12,     # 月頻率
            'Q': 4,      # 季頻率
            'Y': 1       # 年頻率
        }
        
        logger.info("績效計算引擎初始化完成")
    
    def calculate_performance_metrics(self, 
                                    sessions: List[SessionResult],
                                    benchmark_returns: Optional[pd.Series] = None,
                                    frequency: str = 'D') -> PerformanceMetrics:
        """
        計算完整的績效指標
        
        Args:
            sessions: 交易會話結果列表
            benchmark_returns: 基準收益率序列
            frequency: 資料頻率 ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            績效指標
        """
        if not sessions:
            logger.warning("無會話結果，返回空指標")
            return self._create_empty_metrics()
        
        # 合併所有會話的 NAV 序列
        nav_series = self._combine_nav_series(sessions)
        
        if nav_series.empty:
            logger.warning("無有效 NAV 資料，返回空指標")
            return self._create_empty_metrics()
        
        # 計算收益率序列
        returns = nav_series.pct_change().dropna()
        
        # 獲取年化因子
        annualization_factor = self.annualization_factors.get(frequency, 252)
        
        # 計算各類指標
        return_metrics = self._calculate_return_metrics(nav_series, returns, annualization_factor)
        risk_metrics = self._calculate_risk_metrics(nav_series, returns, annualization_factor)
        trading_metrics = self._calculate_trading_metrics(sessions)
        stability_metrics = self._calculate_stability_metrics(returns, frequency)
        
        # 計算基準相關指標
        benchmark_metrics = {}
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_returns, annualization_factor)
        
        # 組合所有指標
        metrics = PerformanceMetrics(
            # 收益指標
            total_return=return_metrics['total_return'],
            annual_return=return_metrics['annual_return'],
            cumulative_return=return_metrics['cumulative_return'],
            
            # 風險指標
            annual_volatility=risk_metrics['annual_volatility'],
            max_drawdown=risk_metrics['max_drawdown'],
            max_drawdown_duration=risk_metrics['max_drawdown_duration'],
            var_95=risk_metrics['var_95'],
            expected_shortfall=risk_metrics['expected_shortfall'],
            
            # 風險調整收益指標
            sharpe_ratio=return_metrics['sharpe_ratio'],
            calmar_ratio=return_metrics['calmar_ratio'],
            sortino_ratio=return_metrics['sortino_ratio'],
            information_ratio=benchmark_metrics.get('information_ratio', 0.0),
            
            # 交易統計
            total_trades=trading_metrics['total_trades'],
            win_rate=trading_metrics['win_rate'],
            profit_factor=trading_metrics['profit_factor'],
            avg_win=trading_metrics['avg_win'],
            avg_loss=trading_metrics['avg_loss'],
            avg_holding_days=trading_metrics['avg_holding_days'],
            
            # 穩定性指標
            best_month=stability_metrics['best_period'],
            worst_month=stability_metrics['worst_period'],
            positive_months=stability_metrics['positive_periods'],
            negative_months=stability_metrics['negative_periods'],
            
            # 時間相關
            start_date=nav_series.index[0] if not nav_series.empty else date.today(),
            end_date=nav_series.index[-1] if not nav_series.empty else date.today(),
            total_days=len(nav_series),
            trading_days=len(returns),
            frequency=frequency
        )
        
        return metrics
    
    def _combine_nav_series(self, sessions: List[SessionResult]) -> pd.Series:
        """合併會話 NAV 序列"""
        all_nav_data = []
        
        for session in sessions:
            if session.nav_series and len(session.nav_series) > 0:
                # 創建日期索引 (簡化處理)
                start_date = session.start_time.date()
                dates = pd.date_range(start=start_date, periods=len(session.nav_series), freq='D')
                
                session_series = pd.Series(session.nav_series, index=dates)
                all_nav_data.append(session_series)
        
        if not all_nav_data:
            return pd.Series(dtype=float)
        
        # 合併所有序列
        combined_series = pd.concat(all_nav_data).sort_index()
        
        # 去除重複日期，保留最後一個值
        combined_series = combined_series[~combined_series.index.duplicated(keep='last')]
        
        return combined_series
    
    def _calculate_return_metrics(self, 
                                nav_series: pd.Series, 
                                returns: pd.Series, 
                                annualization_factor: int) -> Dict[str, float]:
        """計算收益指標"""
        if nav_series.empty or returns.empty:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'cumulative_return': 0.0,
                'sharpe_ratio': 0.0,
                'calmar_ratio': 0.0,
                'sortino_ratio': 0.0
            }
        
        # 總收益率
        total_return = (nav_series.iloc[-1] - nav_series.iloc[0]) / nav_series.iloc[0]
        
        # 年化收益率
        total_days = len(nav_series)
        years = total_days / annualization_factor
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        
        # 累積收益率
        cumulative_return = total_return
        
        # 年化波動率
        annual_volatility = returns.std() * np.sqrt(annualization_factor)
        
        # Sharpe 比率
        risk_free_rate = 0.01  # 假設無風險利率 1%
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown_simple(nav_series)
        
        # Calmar 比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Sortino 比率
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(annualization_factor) if len(downside_returns) > 0 else 0.0
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def _calculate_risk_metrics(self, 
                              nav_series: pd.Series, 
                              returns: pd.Series, 
                              annualization_factor: int) -> Dict[str, float]:
        """計算風險指標"""
        if returns.empty:
            return {
                'annual_volatility': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'var_95': 0.0,
                'expected_shortfall': 0.0
            }
        
        # 年化波動率
        annual_volatility = returns.std() * np.sqrt(annualization_factor)
        
        # 最大回撤和持續時間
        max_drawdown, max_dd_duration = self._calculate_max_drawdown_detailed(nav_series)
        
        # VaR (95%)
        var_95 = np.percentile(returns, 5)  # 5% 分位數
        
        # 期望損失 (Expected Shortfall)
        tail_returns = returns[returns <= var_95]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else 0.0
        
        return {
            'annual_volatility': annual_volatility,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'var_95': var_95,
            'expected_shortfall': expected_shortfall
        }
    
    def _calculate_trading_metrics(self, sessions: List[SessionResult]) -> Dict[str, float]:
        """計算交易統計指標"""
        all_trades = []
        
        # 收集所有交易記錄
        for session in sessions:
            all_trades.extend(session.trade_records)
        
        if not all_trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_holding_days': 0.0
            }
        
        # 分析交易
        total_trades = len(all_trades)
        profitable_trades = [t for t in all_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in all_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0.0
        
        avg_win = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0.0
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0.0
        
        total_profit = sum(t['pnl'] for t in profitable_trades)
        total_loss = sum(abs(t['pnl']) for t in losing_trades)
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        # 平均持倉天數 (簡化計算)
        avg_holding_days = np.mean([t.get('holding_days', 1) for t in all_trades])
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_holding_days': avg_holding_days
        }
    
    def _calculate_stability_metrics(self, returns: pd.Series, frequency: str) -> Dict[str, Any]:
        """計算穩定性指標"""
        if returns.empty:
            return {
                'best_period': 0.0,
                'worst_period': 0.0,
                'positive_periods': 0,
                'negative_periods': 0
            }
        
        # 根據頻率重新採樣
        if frequency == 'D':
            # 按月重新採樣
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        elif frequency == 'W':
            # 按季重新採樣
            monthly_returns = returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
        else:
            monthly_returns = returns
        
        if monthly_returns.empty:
            return {
                'best_period': 0.0,
                'worst_period': 0.0,
                'positive_periods': 0,
                'negative_periods': 0
            }
        
        best_period = monthly_returns.max()
        worst_period = monthly_returns.min()
        positive_periods = (monthly_returns > 0).sum()
        negative_periods = (monthly_returns < 0).sum()
        
        return {
            'best_period': best_period,
            'worst_period': worst_period,
            'positive_periods': positive_periods,
            'negative_periods': negative_periods
        }
    
    def _calculate_benchmark_metrics(self, 
                                   returns: pd.Series, 
                                   benchmark_returns: pd.Series, 
                                   annualization_factor: int) -> Dict[str, float]:
        """計算相對基準的指標"""
        # 對齊時間序列
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if aligned_returns.empty:
            return {'information_ratio': 0.0}
        
        # 超額收益
        excess_returns = aligned_returns - aligned_benchmark
        
        # 信息比率
        tracking_error = excess_returns.std() * np.sqrt(annualization_factor)
        information_ratio = excess_returns.mean() * annualization_factor / tracking_error if tracking_error > 0 else 0.0
        
        return {
            'information_ratio': information_ratio,
            'tracking_error': tracking_error
        }
    
    def _calculate_max_drawdown_simple(self, nav_series: pd.Series) -> float:
        """計算最大回撤 (簡化版本)"""
        if nav_series.empty:
            return 0.0
        
        peak = nav_series.expanding().max()
        drawdown = (nav_series - peak) / peak
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown)
    
    def _calculate_max_drawdown_detailed(self, nav_series: pd.Series) -> Tuple[float, int]:
        """計算最大回撤和持續時間"""
        if nav_series.empty:
            return 0.0, 0
        
        peak = nav_series.iloc[0]
        max_drawdown = 0.0
        max_duration = 0
        current_duration = 0
        
        for nav in nav_series:
            if nav > peak:
                peak = nav
                current_duration = 0
            else:
                drawdown = (peak - nav) / peak
                max_drawdown = max(max_drawdown, drawdown)
                current_duration += 1
                max_duration = max(max_duration, current_duration)
        
        return max_drawdown, max_duration
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """創建空的績效指標"""
        return PerformanceMetrics(
            total_return=0.0,
            annual_return=0.0,
            cumulative_return=0.0,
            annual_volatility=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            var_95=0.0,
            expected_shortfall=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            information_ratio=0.0,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_holding_days=0.0,
            best_month=0.0,
            worst_month=0.0,
            positive_months=0,
            negative_months=0,
            start_date=date.today(),
            end_date=date.today(),
            total_days=0,
            trading_days=0,
            frequency='D'
        )


def test_metrics_engine():
    """測試績效計算引擎 - References.txt 建議的 pandas.testing 單元測試"""
    print("=== 測試績效計算引擎 ===")
    
    # 創建測試配置
    from config import create_smoke_test_config
    config = create_smoke_test_config()
    
    # 創建績效引擎
    print("1. 創建績效引擎...")
    try:
        engine = MetricsEngine(config)
        print(f"   ✅ 績效引擎創建成功")
        print(f"   年化因子: {engine.annualization_factors}")
    except Exception as e:
        print(f"   ❌ 績效引擎創建失敗: {e}")
        return False
    
    # 測試基本指標計算 - 使用手算值驗證
    print("2. 測試基本指標計算...")
    try:
        # 創建測試 NAV 序列
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # 模擬 NAV 序列：初始 100萬，每日隨機變動
        np.random.seed(42)  # 固定隨機種子確保可重現
        daily_returns = np.random.normal(0.001, 0.02, 100)  # 日均收益 0.1%，波動 2%
        nav_values = [1000000]
        
        for ret in daily_returns:
            nav_values.append(nav_values[-1] * (1 + ret))
        
        nav_series = pd.Series(nav_values[1:], index=dates)
        returns = nav_series.pct_change().dropna()
        
        # 計算指標
        return_metrics = engine._calculate_return_metrics(nav_series, returns, 252)
        
        # 手算驗證 - 總收益率
        expected_total_return = (nav_series.iloc[-1] - nav_series.iloc[0]) / nav_series.iloc[0]
        calculated_total_return = return_metrics['total_return']
        
        # 使用 pandas.testing 進行精確比較
        import pandas.testing as pdt
        
        # 轉換為 Series 進行比較
        expected_series = pd.Series([expected_total_return], index=['total_return'])
        calculated_series = pd.Series([calculated_total_return], index=['total_return'])
        
        try:
            pdt.assert_series_equal(expected_series, calculated_series, rtol=1e-10)
            print(f"   ✅ 總收益率計算正確: {calculated_total_return:.6f}")
        except AssertionError:
            print(f"   ❌ 總收益率計算錯誤: 期望 {expected_total_return:.6f}, 實際 {calculated_total_return:.6f}")
            return False
        
        # 驗證年化波動率
        expected_volatility = returns.std() * np.sqrt(252)
        calculated_volatility = return_metrics.get('annual_volatility', 
                                                  engine._calculate_risk_metrics(nav_series, returns, 252)['annual_volatility'])
        
        if abs(expected_volatility - calculated_volatility) < 1e-10:
            print(f"   ✅ 年化波動率計算正確: {calculated_volatility:.6f}")
        else:
            print(f"   ❌ 年化波動率計算錯誤: 期望 {expected_volatility:.6f}, 實際 {calculated_volatility:.6f}")
            return False
        
    except Exception as e:
        print(f"   ❌ 基本指標計算測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 測試最大回撤計算
    print("3. 測試最大回撤計算...")
    try:
        # 創建已知最大回撤的序列
        test_nav = pd.Series([100, 110, 120, 90, 80, 100, 110], 
                           index=pd.date_range('2024-01-01', periods=7, freq='D'))
        
        # 手算最大回撤：從 120 跌到 80，回撤 = (120-80)/120 = 1/3 ≈ 0.3333
        expected_max_dd = (120 - 80) / 120
        calculated_max_dd = engine._calculate_max_drawdown_simple(test_nav)
        
        if abs(expected_max_dd - calculated_max_dd) < 1e-6:
            print(f"   ✅ 最大回撤計算正確: {calculated_max_dd:.6f}")
        else:
            print(f"   ❌ 最大回撤計算錯誤: 期望 {expected_max_dd:.6f}, 實際 {calculated_max_dd:.6f}")
            return False
            
    except Exception as e:
        print(f"   ❌ 最大回撤計算測試失敗: {e}")
        return False
    
    # 測試年化因子動態決定
    print("4. 測試年化因子動態決定...")
    try:
        test_frequencies = ['D', 'W', 'M', 'Q', 'Y']
        expected_factors = [252, 52, 12, 4, 1]
        
        for freq, expected_factor in zip(test_frequencies, expected_factors):
            actual_factor = engine.annualization_factors.get(freq)
            if actual_factor == expected_factor:
                print(f"   ✅ {freq} 頻率年化因子正確: {actual_factor}")
            else:
                print(f"   ❌ {freq} 頻率年化因子錯誤: 期望 {expected_factor}, 實際 {actual_factor}")
                return False
                
    except Exception as e:
        print(f"   ❌ 年化因子測試失敗: {e}")
        return False
    
    print("✅ 績效計算引擎測試完成")
    return True


if __name__ == "__main__":
    test_metrics_engine()