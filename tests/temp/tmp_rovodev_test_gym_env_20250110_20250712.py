#!/usr/bin/env python3
"""
快速測試 Gym 環境模組完整性
"""
import sys
import os
from pathlib import Path

# 添加路徑
sys.path.append(str(Path(__file__).parent))

def test_gym_env_completeness():
    """測試 Gym 環境模組完整性"""
    print("=== 測試 Gym 環境模組完整性 ===")
    
    # 1. 測試模組導入
    print("1. 測試模組導入...")
    try:
        from gym_env.env import TSEAlphaEnv, AccountManager, DataProvider
        from gym_env.reward import RewardCalculator, AdaptiveRewardCalculator
        import gym_env  # 觸發環境註冊
        print("   ✅ 所有核心類別導入成功")
    except Exception as e:
        print(f"   ❌ 模組導入失敗: {e}")
        return False
    
    # 2. 測試環境註冊
    print("2. 測試環境註冊...")
    try:
        import gymnasium as gym
        env = gym.make('TSEAlpha-v0')
        print("   ✅ 環境註冊成功")
        env.close()
    except Exception as e:
        print(f"   ❌ 環境註冊失敗: {e}")
        return False
    
    # 3. 測試環境基本功能
    print("3. 測試環境基本功能...")
    try:
        env = gym.make('TSEAlpha-v0', 
                      symbols=['2330', '2317'], 
                      start_date='2024-01-01',
                      end_date='2024-01-31')
        
        # 重置測試
        obs, info = env.reset()
        print(f"   觀測空間: {list(obs.keys())}")
        print(f"   動作空間: {env.action_space}")
        
        # 步進測試
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   步進成功，獎勵: {reward:.6f}")
        
        env.close()
        print("   ✅ 環境基本功能正常")
    except Exception as e:
        print(f"   ❌ 環境基本功能測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 測試獎勵計算
    print("4. 測試獎勵計算...")
    try:
        calc = RewardCalculator(initial_nav=1000000.0)
        reward = calc.calculate_reward(
            current_nav=1010000,
            transaction_cost=1000,
            timeout_count=0,
            risk_violation=False
        )
        print(f"   獎勵計算成功: {reward['total_reward']:.6f}")
        print("   ✅ 獎勵計算功能正常")
    except Exception as e:
        print(f"   ❌ 獎勵計算測試失敗: {e}")
        return False
    
    # 5. 測試帳戶管理
    print("5. 測試帳戶管理...")
    try:
        account = AccountManager(initial_cash=1000000.0)
        
        # 測試交易
        result = account.execute_trade('2330', 100, 580.0)
        print(f"   交易執行: {result}")
        
        # 測試 NAV 計算
        nav = account.get_nav({'2330': 590.0})
        print(f"   NAV 計算: {nav:,.2f}")
        
        print("   ✅ 帳戶管理功能正常")
    except Exception as e:
        print(f"   ❌ 帳戶管理測試失敗: {e}")
        return False
    
    # 6. 測試資料提供器
    print("6. 測試資料提供器...")
    try:
        data_provider = DataProvider(['2330'], '2024-01-01', '2024-01-31')
        current_date = data_provider.get_current_date()
        prices = data_provider.get_current_prices()
        obs_data = data_provider.get_observation_data()
        
        print(f"   當前日期: {current_date}")
        print(f"   價格數據: {len(prices)} 檔股票")
        print(f"   觀測數據: {list(obs_data.keys())}")
        print("   ✅ 資料提供器功能正常")
    except Exception as e:
        print(f"   ❌ 資料提供器測試失敗: {e}")
        return False
    
    return True

def test_gym_env_integration():
    """測試 Gym 環境整合功能"""
    print("\n=== 測試 Gym 環境整合功能 ===")
    
    try:
        import gymnasium as gym
        
        # 創建環境
        env = gym.make('TSEAlpha-v0',
                      symbols=['2330', '2317', '2603'],
                      start_date='2024-01-01', 
                      end_date='2024-01-10',
                      initial_cash=1000000.0)
        
        print("1. 執行完整交易場景...")
        
        # 重置環境
        obs, info = env.reset()
        print(f"   初始 NAV: {info['nav']:,.2f}")
        
        total_reward = 0.0
        episode_length = 0
        
        # 執行交易序列
        for step in range(10):
            # 簡單策略：隨機選股票，小量交易
            stock_idx = step % len(env.symbols)
            qty = 50 if step % 2 == 0 else -25  # 買入或賣出
            
            action = (stock_idx, [qty])
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            episode_length += 1
            
            if step % 3 == 0:
                print(f"   步驟 {step}: 獎勵={reward:.6f}, NAV={info['nav']:,.2f}, 持倉={len(info['positions'])}")
            
            if terminated or truncated:
                print(f"   環境在第 {step} 步結束")
                break
        
        print(f"   總獎勵: {total_reward:.6f}")
        print(f"   最終 NAV: {info['nav']:,.2f}")
        print(f"   交易步數: {episode_length}")
        
        env.close()
        print("   ✅ 整合功能測試完成")
        return True
        
    except Exception as e:
        print(f"   ❌ 整合功能測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("開始測試 Gym 環境模組...")
    
    # 基本完整性測試
    basic_ok = test_gym_env_completeness()
    
    # 整合功能測試
    integration_ok = test_gym_env_integration()
    
    print("\n" + "="*50)
    if basic_ok and integration_ok:
        print("✅ Gym 環境模組測試全部通過")
        print("✅ 環境模組開發完成，可以進行回測開發")
        return True
    else:
        print("❌ Gym 環境模組測試失敗")
        print("❌ 需要修復問題後再進行回測開發")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)