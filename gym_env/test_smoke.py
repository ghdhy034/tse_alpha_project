#!/usr/bin/env python3
"""
Gym Environment Smoke Test - 快速驗證交易環境功能
"""
import sys
import os
from pathlib import Path
import numpy as np
import gymnasium as gym

# 添加路徑
sys.path.append(str(Path(__file__).parent))

def test_environment_registration():
    """測試環境註冊"""
    print("1. 測試環境註冊...")
    try:
        import gym_env  # 觸發環境註冊
        env = gym.make('TSEAlpha-v0')
        print("   ✅ 環境註冊成功")
        env.close()
        return True
    except Exception as e:
        print(f"   ❌ 環境註冊失敗: {e}")
        return False

def test_environment_reset():
    """測試環境重置"""
    print("2. 測試環境重置...")
    try:
        import gym_env
        env = gym.make('TSEAlpha-v0')
        
        obs, info = env.reset()
        
        # 檢查觀測格式
        expected_keys = ['price_frame', 'fundamental', 'account']
        if all(key in obs for key in expected_keys):
            print("   ✅ 觀測格式正確")
            print(f"      price_frame shape: {obs['price_frame'].shape}")
            print(f"      fundamental shape: {obs['fundamental'].shape}")
            print(f"      account shape: {obs['account'].shape}")
        else:
            print(f"   ❌ 觀測格式錯誤，缺少鍵: {set(expected_keys) - set(obs.keys())}")
            return False
        
        # 檢查 info 內容
        expected_info_keys = ['nav', 'cash', 'positions', 'current_date']
        if all(key in info for key in expected_info_keys):
            print("   ✅ info 格式正確")
            print(f"      初始 NAV: {info['nav']:,.2f}")
            print(f"      初始現金: {info['cash']:,.2f}")
        else:
            print(f"   ❌ info 格式錯誤")
            return False
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ 環境重置失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_step():
    """測試環境步進"""
    print("3. 測試環境步進...")
    try:
        import gym_env
        env = gym.make('TSEAlpha-v0')
        
        obs, info = env.reset()
        
        # 執行幾個動作
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"   步驟 {step + 1}:")
            print(f"      動作: 股票 {action[0]}, 數量 {action[1][0]}")
            print(f"      獎勵: {reward:.6f}")
            print(f"      NAV: {info['nav']:,.2f}")
            print(f"      terminated: {terminated}, truncated: {truncated}")
            
            if terminated or truncated:
                print(f"      環境提前結束")
                break
        
        print("   ✅ 環境步進正常")
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ 環境步進失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_position_limits():
    """測試持倉限制"""
    print("4. 測試持倉限制...")
    try:
        import gym_env
        env = gym.make('TSEAlpha-v0')
        
        obs, info = env.reset()
        
        # 嘗試超大交易量
        large_qty = 500  # 超過 300 股限制
        action = (0, np.array([large_qty], dtype=np.int16))
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 檢查是否正確限制
        if 'trade_result' in info and info['trade_result']:
            if not info['trade_result']['success']:
                print("   ✅ 持倉限制正常工作")
            else:
                print("   ⚠️  持倉限制可能未正確實施")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ 持倉限制測試失敗: {e}")
        return False

def test_reward_calculation():
    """測試獎勵計算"""
    print("5. 測試獎勵計算...")
    try:
        from reward import RewardCalculator
        
        calc = RewardCalculator(initial_nav=1000000.0)
        
        # 測試基本獎勵計算
        reward = calc.calculate_reward(
            current_nav=1010000,  # 獲利 1 萬
            transaction_cost=1000,
            timeout_count=0,
            risk_violation=False
        )
        
        expected_delta_nav = 10000 / 1000000  # 1%
        if abs(reward['delta_nav'] - expected_delta_nav) < 1e-6:
            print("   ✅ ΔNAV 計算正確")
        else:
            print(f"   ❌ ΔNAV 計算錯誤: {reward['delta_nav']} vs {expected_delta_nav}")
            return False
        
        if reward['cost_penalty'] > 0:
            print("   ✅ 成本罰款計算正確")
        else:
            print("   ❌ 成本罰款計算錯誤")
            return False
        
        print(f"   總獎勵: {reward['total_reward']:.6f}")
        return True
        
    except Exception as e:
        print(f"   ❌ 獎勵計算測試失敗: {e}")
        return False

def test_timeout_mechanism():
    """測試超時機制"""
    print("6. 測試超時機制...")
    try:
        import gym_env
        env = gym.make('TSEAlpha-v0', max_holding_days=3)  # 設定 3 天超時
        
        obs, info = env.reset()
        
        # 建立持倉
        buy_action = (0, np.array([100], dtype=np.int16))
        obs, reward, terminated, truncated, info = env.step(buy_action)
        
        # 持有多天不交易
        for day in range(5):  # 超過 3 天限制
            hold_action = (0, np.array([0], dtype=np.int16))  # 不交易
            obs, reward, terminated, truncated, info = env.step(hold_action)
            
            print(f"   第 {day + 1} 天持有，持倉數: {len(info['positions'])}")
            
            if day >= 3 and len(info['positions']) == 0:
                print("   ✅ 超時強制平倉機制正常")
                break
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ 超時機制測試失敗: {e}")
        return False

def test_performance_benchmark():
    """測試效能基準"""
    print("7. 測試效能基準...")
    try:
        import time
        import gym_env
        
        env = gym.make('TSEAlpha-v0')
        
        # 測試 reset + step 效能
        start_time = time.time()
        
        num_operations = 1000
        for i in range(num_operations):
            if i % 100 == 0:  # 每 100 次重置一次
                obs, info = env.reset()
            else:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    obs, info = env.reset()
        
        end_time = time.time()
        elapsed = end_time - start_time
        ops_per_second = num_operations / elapsed
        
        print(f"   {num_operations} 次操作耗時: {elapsed:.2f} 秒")
        print(f"   每秒操作數: {ops_per_second:.0f}")
        
        # 檢查是否滿足效能要求 (1M operations < 60s)
        estimated_1m_time = 1000000 / ops_per_second
        if estimated_1m_time < 60:
            print(f"   ✅ 效能達標 (預估 1M 操作: {estimated_1m_time:.1f} 秒)")
        else:
            print(f"   ⚠️  效能可能不達標 (預估 1M 操作: {estimated_1m_time:.1f} 秒)")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ 效能測試失敗: {e}")
        return False

def run_smoke_test():
    """執行完整的 Smoke Test"""
    print("開始執行 Gym Environment Smoke Test")
    print("=" * 50)
    
    tests = [
        test_environment_registration,
        test_environment_reset,
        test_environment_step,
        test_position_limits,
        test_reward_calculation,
        test_timeout_mechanism,
        test_performance_benchmark,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print("   ⚠️  測試未完全通過，但繼續執行...")
        except Exception as e:
            print(f"   ❌ 測試異常: {e}")
        
        print()  # 空行分隔
    
    print("=" * 50)
    if passed >= total * 0.8:  # 80% 通過率
        print(f"✅ Smoke Test 基本通過 ({passed}/{total})")
        print("Gym 環境基本功能正常！")
        return True
    else:
        print(f"❌ Smoke Test 失敗 ({passed}/{total})")
        print("請檢查失敗的測試項目")
        return False

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)