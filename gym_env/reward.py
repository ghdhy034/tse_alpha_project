# gym_env/reward.py
"""
獎勵計算模組 - 實作 ΔNAV - cost - κ·timeout 獎勵機制
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RewardCalculator:
    """獎勵計算器"""
    
    def __init__(self, 
                 initial_nav: float,
                 timeout_penalty_factor: float = 0.01,
                 risk_penalty_factor: float = 0.05,
                 transaction_cost_factor: float = 0.001425):
        """
        初始化獎勵計算器
        
        Args:
            initial_nav: 初始 NAV
            timeout_penalty_factor: 超時罰款係數 κ
            risk_penalty_factor: 風險限制罰款係數
            transaction_cost_factor: 交易成本係數
        """
        self.initial_nav = initial_nav
        self.timeout_penalty_factor = timeout_penalty_factor
        self.risk_penalty_factor = risk_penalty_factor
        self.transaction_cost_factor = transaction_cost_factor
        
        # 歷史記錄
        self.nav_history = []
        self.cost_history = []
        self.penalty_history = []
        
    def calculate_reward(self,
                        current_nav: float,
                        transaction_cost: float = 0.0,
                        timeout_count: int = 0,
                        risk_violation: bool = False) -> Dict[str, float]:
        """
        計算當前步驟的獎勵
        
        Args:
            current_nav: 當前 NAV
            transaction_cost: 交易成本
            timeout_count: 超時持倉數量
            risk_violation: 是否違反風險限制
            
        Returns:
            包含獎勵分解的字典
        """
        # 計算 ΔNAV
        if self.nav_history:
            delta_nav = current_nav - self.nav_history[-1]
        else:
            delta_nav = current_nav - self.initial_nav
        
        # 標準化 ΔNAV (相對於初始 NAV)
        delta_nav_normalized = delta_nav / self.initial_nav
        
        # 計算成本罰款
        cost_penalty = transaction_cost / self.initial_nav
        
        # 計算超時罰款
        timeout_penalty = timeout_count * self.timeout_penalty_factor
        
        # 計算風險違規罰款
        risk_penalty = self.risk_penalty_factor if risk_violation else 0.0
        
        # 總獎勵 = ΔNAV - cost - κ·timeout - risk_penalty
        total_reward = delta_nav_normalized - cost_penalty - timeout_penalty - risk_penalty
        
        # 記錄歷史
        self.nav_history.append(current_nav)
        self.cost_history.append(cost_penalty)
        self.penalty_history.append(timeout_penalty + risk_penalty)
        
        reward_breakdown = {
            'total_reward': total_reward,
            'delta_nav': delta_nav_normalized,
            'cost_penalty': cost_penalty,
            'timeout_penalty': timeout_penalty,
            'risk_penalty': risk_penalty,
            'raw_delta_nav': delta_nav
        }
        
        logger.debug(f"獎勵計算: {reward_breakdown}")
        return reward_breakdown
    
    def get_cumulative_reward(self) -> float:
        """獲取累積獎勵"""
        if not self.nav_history:
            return 0.0
        
        # 總 NAV 變化
        total_nav_change = (self.nav_history[-1] - self.initial_nav) / self.initial_nav
        
        # 總成本
        total_cost = sum(self.cost_history)
        
        # 總罰款
        total_penalty = sum(self.penalty_history)
        
        return total_nav_change - total_cost - total_penalty
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.01) -> float:
        """計算 Sharpe 比率"""
        if len(self.nav_history) < 2:
            return 0.0
        
        # 計算日收益率
        returns = []
        for i in range(1, len(self.nav_history)):
            daily_return = (self.nav_history[i] - self.nav_history[i-1]) / self.nav_history[i-1]
            returns.append(daily_return)
        
        if not returns:
            return 0.0
        
        returns = np.array(returns)
        
        # 年化收益率和波動率 (假設 252 個交易日)
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        
        if annual_volatility == 0:
            return 0.0
        
        sharpe = (annual_return - risk_free_rate) / annual_volatility
        return sharpe
    
    def get_max_drawdown(self) -> float:
        """計算最大回撤"""
        if len(self.nav_history) < 2:
            return 0.0
        
        navs = np.array(self.nav_history)
        peak = np.maximum.accumulate(navs)
        drawdown = (navs - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return abs(max_drawdown)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """獲取完整的績效指標"""
        if not self.nav_history:
            return {}
        
        current_nav = self.nav_history[-1]
        total_return = (current_nav - self.initial_nav) / self.initial_nav
        
        metrics = {
            'total_return': total_return,
            'cumulative_reward': self.get_cumulative_reward(),
            'sharpe_ratio': self.get_sharpe_ratio(),
            'max_drawdown': self.get_max_drawdown(),
            'final_nav': current_nav,
            'total_trades': len(self.cost_history),
            'total_cost': sum(self.cost_history),
            'total_penalty': sum(self.penalty_history)
        }
        
        return metrics
    
    def reset(self):
        """重置獎勵計算器"""
        self.nav_history.clear()
        self.cost_history.clear()
        self.penalty_history.clear()


class AdaptiveRewardCalculator(RewardCalculator):
    """自適應獎勵計算器 - 根據市場狀況調整獎勵"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.market_volatility_window = 20
        self.base_volatility = 0.02  # 基準波動率
        
    def calculate_market_adjusted_reward(self,
                                       current_nav: float,
                                       market_prices: Dict[str, float],
                                       transaction_cost: float = 0.0,
                                       timeout_count: int = 0,
                                       risk_violation: bool = False) -> Dict[str, float]:
        """
        計算市場調整後的獎勵
        
        Args:
            current_nav: 當前 NAV
            market_prices: 市場價格字典
            transaction_cost: 交易成本
            timeout_count: 超時持倉數量
            risk_violation: 是否違反風險限制
            
        Returns:
            包含市場調整獎勵的字典
        """
        # 基本獎勵計算
        basic_reward = self.calculate_reward(
            current_nav, transaction_cost, timeout_count, risk_violation
        )
        
        # 計算市場波動率調整
        volatility_adjustment = self._calculate_volatility_adjustment(market_prices)
        
        # 調整獎勵
        adjusted_reward = basic_reward['total_reward'] * volatility_adjustment
        
        result = basic_reward.copy()
        result['volatility_adjustment'] = volatility_adjustment
        result['market_adjusted_reward'] = adjusted_reward
        
        return result
    
    def _calculate_volatility_adjustment(self, market_prices: Dict[str, float]) -> float:
        """計算波動率調整係數"""
        if len(self.nav_history) < self.market_volatility_window:
            return 1.0
        
        # 計算最近的 NAV 波動率
        recent_navs = self.nav_history[-self.market_volatility_window:]
        returns = []
        
        for i in range(1, len(recent_navs)):
            ret = (recent_navs[i] - recent_navs[i-1]) / recent_navs[i-1]
            returns.append(ret)
        
        if not returns:
            return 1.0
        
        current_volatility = np.std(returns)
        
        # 波動率調整：高波動時降低獎勵敏感度，低波動時提高
        if current_volatility > 0:
            adjustment = self.base_volatility / current_volatility
            # 限制調整範圍在 0.5 到 2.0 之間
            adjustment = np.clip(adjustment, 0.5, 2.0)
        else:
            adjustment = 1.0
        
        return adjustment


def test_reward_calculator():
    """測試獎勵計算器"""
    print("=== 測試獎勵計算器 ===")
    
    # 基本獎勵計算器測試
    calc = RewardCalculator(initial_nav=1000000.0)
    
    # 模擬幾個交易日
    test_scenarios = [
        {'nav': 1010000, 'cost': 1000, 'timeout': 0, 'risk': False},  # 獲利
        {'nav': 1005000, 'cost': 500, 'timeout': 1, 'risk': False},   # 獲利但有超時
        {'nav': 995000, 'cost': 800, 'timeout': 0, 'risk': True},     # 虧損且風險違規
        {'nav': 1020000, 'cost': 1200, 'timeout': 0, 'risk': False},  # 大幅獲利
    ]
    
    print("基本獎勵計算測試:")
    for i, scenario in enumerate(test_scenarios):
        reward = calc.calculate_reward(
            current_nav=scenario['nav'],
            transaction_cost=scenario['cost'],
            timeout_count=scenario['timeout'],
            risk_violation=scenario['risk']
        )
        
        print(f"第 {i+1} 天:")
        print(f"  NAV: {scenario['nav']:,}")
        print(f"  總獎勵: {reward['total_reward']:.6f}")
        print(f"  ΔNAV: {reward['delta_nav']:.6f}")
        print(f"  成本罰款: {reward['cost_penalty']:.6f}")
        print(f"  超時罰款: {reward['timeout_penalty']:.6f}")
        print(f"  風險罰款: {reward['risk_penalty']:.6f}")
        print()
    
    # 績效指標測試
    print("績效指標:")
    metrics = calc.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n=== 測試自適應獎勵計算器 ===")
    
    # 自適應獎勵計算器測試
    adaptive_calc = AdaptiveRewardCalculator(initial_nav=1000000.0)
    
    # 模擬市場價格
    market_prices = {'2330': 580.0, '2317': 110.0}
    
    reward = adaptive_calc.calculate_market_adjusted_reward(
        current_nav=1010000,
        market_prices=market_prices,
        transaction_cost=1000,
        timeout_count=0,
        risk_violation=False
    )
    
    print("自適應獎勵計算:")
    for key, value in reward.items():
        print(f"  {key}: {value:.6f}")
    
    print("✅ 獎勵計算器測試完成")


if __name__ == "__main__":
    test_reward_calculator()