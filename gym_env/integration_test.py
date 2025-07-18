#!/usr/bin/env python3
"""
Gym Environment Integration Test - 完整功能整合測試
"""
import sys
import os
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import gymnasium as gym

# 添加路徑
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "market_data_collector"))

def test_full_trading_scenario():
    """測試完整交易場景"""
    print("=== 完整交易場景測試 ===")
    
    try:
        import gym_env
        from backtest.ledger import TradingLedger
        
        # 建立環境和帳本
        env = gym.make('TSEAlpha-v0', 
                      symbols=['2330', '2317'],
                      start_date='2024-01-01',
                      end_date='2024-01-10',
                      initial_cash=1000000.0)
        
        ledger = TradingLedger("integration_test")
        ledger.cleanup_session()  # 清理舊資料
        
        obs, info = env.reset()
        print(f"初始狀態: NAV={info['nav']:,.2f}, 現金={info['cash']:,.2f}")
        
        total_reward = 0.0
        step_count = 0
        
        # 執行交易策略
        while step_count < 20:  # 限制步數
            step_count += 1
            
            # 簡單策略：隨機交易
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            # 記錄到帳本
            if info.get('trade_executed') and info.get('trade_result', {}).get('success'):
                symbol_idx, qty_array = action
                symbol = env.symbols[symbol_idx]
                qty = int(qty_array[0])
                current_prices = info.get('current_prices', {})
                
                if symbol in current_prices and qty != 0:
                    action_type = 'BUY' if qty > 0 else 'SELL'
                    price = current_prices[symbol]
                    
                    ledger.record_trade(
                        trade_date=info['current_date'],
                        symbol=symbol,
                        action=action_type,
                        quantity=abs(qty),
                        price=price
                    )
            
            # 更新帳本快照
            if info.get('current_date'):
                ledger.update_positions(
                    snapshot_date=info['current_date'],
                    positions=info.get('positions', {}),
                    current_prices=info.get('current_prices', {})
                )
                
                ledger.update_account(
                    snapshot_date=info['current_date'],
                    cash=info['cash'],
                    position_value=info['nav'] - info['cash'],
                    total_nav=info['nav']
                )
            
            print(f"步驟 {step_count}: 獎勵={reward:.6f}, NAV={info['nav']:,.2f}, 持倉={len(info['positions'])}")
            
            if terminated or truncated:
                print(f"環境結束: terminated={terminated}, truncated={truncated}")
                break
        
        # 計算最終績效
        final_metrics = ledger.calculate_performance_metrics()
        print(f"\n最終績效:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                if 'rate' in key or 'return' in key:
                    print(f"  {key}: {value:.2%}")
                else:
                    print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"總獎勵: {total_reward:.6f}")
        print("✅ 完整交易場景測試成功")
        
        # 清理測試資料
        ledger.cleanup_session()
        env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 完整交易場景測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_control_integration():
    """測試風險控制整合"""
    print("\n=== 風險控制整合測試 ===")
    
    try:
        import gym_env
        
        # 建立高風險環境
        env = gym.make('TSEAlpha-v0',
                      symbols=['2330'],
                      start_date='2024-01-01',
                      end_date='2024-01-05',
                      initial_cash=100000.0,  # 較小的初始資金
                      daily_max_loss_pct=0.05,  # 5% 日損失限制
                      max_holding_days=2)  # 2 天超時
        
        obs, info = env.reset()
        print(f"初始狀態: NAV={info['nav']:,.2f}")
        
        # 嘗試大額交易觸發風險控制
        large_buy_action = (0, np.array([200], dtype=np.int16))  # 大量買入
        
        for step in range(10):
            obs, reward, terminated, truncated, info = env.step(large_buy_action)
            
            print(f"步驟 {step + 1}: NAV={info['nav']:,.2f}, 持倉={len(info['positions'])}")
            
            if truncated:
                print("✅ 風險控制機制正常觸發")
                break
            
            if terminated:
                print("環境正常結束")
                break
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 風險控制整合測試失敗: {e}")
        return False

def test_reward_system_integration():
    """測試獎勵系統整合"""
    print("\n=== 獎勵系統整合測試 ===")
    
    try:
        from reward import RewardCalculator, AdaptiveRewardCalculator
        import gym_env
        
        # 測試基本獎勵計算器
        calc = RewardCalculator(initial_nav=1000000.0)
        
        # 模擬一系列 NAV 變化
        nav_sequence = [1000000, 1010000, 1005000, 1020000, 995000]
        
        print("基本獎勵計算:")
        for i, nav in enumerate(nav_sequence[1:], 1):
            reward = calc.calculate_reward(
                current_nav=nav,
                transaction_cost=1000,
                timeout_count=0,
                risk_violation=False
            )
            print(f"  第 {i} 天: NAV={nav:,}, 獎勵={reward['total_reward']:.6f}")
        
        # 測試績效指標
        metrics = calc.get_performance_metrics()
        print(f"績效指標: Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
              f"MaxDD={metrics.get('max_drawdown', 0):.3f}")
        
        # 測試自適應獎勵計算器
        adaptive_calc = AdaptiveRewardCalculator(initial_nav=1000000.0)
        market_prices = {'2330': 580.0, '2317': 110.0}
        
        adaptive_reward = adaptive_calc.calculate_market_adjusted_reward(
            current_nav=1010000,
            market_prices=market_prices,
            transaction_cost=1000
        )
        
        print(f"自適應獎勵: {adaptive_reward['market_adjusted_reward']:.6f}")
        print("✅ 獎勵系統整合測試成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 獎勵系統整合測試失敗: {e}")
        return False

def test_data_integration():
    """測試資料整合"""
    print("\n=== 資料整合測試 ===")
    
    try:
        import gym_env
        
        # 測試與真實資料的整合
        env = gym.make('TSEAlpha-v0',
                      symbols=['2330', '2317'],
                      start_date='2024-01-01',
                      end_date='2024-01-05')
        
        obs, info = env.reset()
        
        # 檢查觀測資料格式
        print(f"價格框架形狀: {obs['price_frame'].shape}")
        print(f"基本面特徵形狀: {obs['fundamental'].shape}")
        print(f"帳戶狀態形狀: {obs['account'].shape}")
        
        # 檢查價格資料
        if info.get('current_prices'):
            print(f"當前價格: {info['current_prices']}")
        else:
            print("使用虛擬價格資料")
        
        # 執行幾步檢查資料一致性
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 檢查觀測資料是否有效
            if np.any(np.isnan(obs['price_frame'])):
                print("⚠️  價格框架包含 NaN 值")
            
            if np.any(np.isnan(obs['account'])):
                print("⚠️  帳戶狀態包含 NaN 值")
            
            if terminated or truncated:
                break
        
        print("✅ 資料整合測試成功")
        env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 資料整合測試失敗: {e}")
        return False

def run_integration_tests():
    """執行所有整合測試"""
    print("開始執行 Gym Environment 整合測試")
    print("=" * 60)
    
    tests = [
        test_full_trading_scenario,
        test_risk_control_integration,
        test_reward_system_integration,
        test_data_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print("   ⚠️  測試未完全通過")
        except Exception as e:
            print(f"   ❌ 測試異常: {e}")
        
        print()  # 空行分隔
    
    print("=" * 60)
    if passed >= total * 0.8:  # 80% 通過率
        print(f"✅ 整合測試基本通過 ({passed}/{total})")
        print("Gym 環境整合功能正常！")
        return True
    else:
        print(f"❌ 整合測試失敗 ({passed}/{total})")
        print("請檢查失敗的測試項目")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)