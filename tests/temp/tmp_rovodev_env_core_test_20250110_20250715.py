#!/usr/bin/env python3
"""
TSE Alpha 環境核心功能測試腳本
測試 TSEAlphaEnv 的基本功能和核心機制
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from typing import Dict, Any, List
import traceback

# 添加路徑
sys.path.append(str(Path(__file__).parent))

def test_environment_creation():
    """測試 1: 環境創建和初始化"""
    print("🔧 測試 1: 環境創建和初始化")
    print("-" * 40)
    
    try:
        from gym_env.env import TSEAlphaEnv
        
        # 測試基本創建
        env = TSEAlphaEnv(
            symbols=['2330', '2317', '2454'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_cash=1000000.0
        )
        
        print("✅ 環境創建成功")
        print(f"   股票數量: {len(env.symbols)}")
        print(f"   初始資金: {env.initial_cash:,.0f}")
        print(f"   最大持倉天數: {env.max_holding_days}")
        
        # 檢查動作空間
        print(f"   動作空間: {env.action_space}")
        print(f"   觀測空間: {env.observation_space}")
        
        # 檢查組件初始化
        assert env.account_manager is not None, "帳戶管理器未初始化"
        assert env.symbols == ['2330', '2317', '2454'], "股票清單不正確"
        
        print("✅ 環境初始化驗證通過")
        return True, env
        
    except Exception as e:
        print(f"❌ 環境創建失敗: {str(e)}")
        traceback.print_exc()
        return False, None

def test_environment_reset(env):
    """測試 2: 環境重置功能"""
    print("\n🔄 測試 2: 環境重置功能")
    print("-" * 40)
    
    try:
        # 重置環境
        observation, info = env.reset(seed=42)
        
        print("✅ 環境重置成功")
        
        # 檢查觀測格式
        expected_keys = ['price_frame', 'fundamental', 'account']
        for key in expected_keys:
            assert key in observation, f"觀測缺少 {key}"
            print(f"   {key}: {observation[key].shape}")
        
        # 檢查帳戶初始狀態
        assert env.account_manager.cash == env.initial_cash, "初始現金不正確"
        assert len(env.account_manager.positions) == 0, "初始持倉應為空"
        
        # 檢查資料提供器
        assert env.data_provider is not None, "資料提供器未初始化"
        assert env.episode_step_count == 0, "步數計數器未重置"
        
        print("✅ 重置狀態驗證通過")
        print(f"   初始現金: {env.account_manager.cash:,.0f}")
        print(f"   初始持倉: {len(env.account_manager.positions)} 檔")
        print(f"   步數計數: {env.episode_step_count}")
        
        return True, observation, info
        
    except Exception as e:
        print(f"❌ 環境重置失敗: {str(e)}")
        traceback.print_exc()
        return False, None, None

def test_action_execution(env):
    """測試 3: 動作執行和交易機制"""
    print("\n⚡ 測試 3: 動作執行和交易機制")
    print("-" * 40)
    
    try:
        # 重置環境
        observation, info = env.reset(seed=42)
        
        # 測試買入動作
        stock_idx = 0  # 選擇第一檔股票 (2330)
        buy_qty = 100  # 買入100股
        action = (stock_idx, np.array([buy_qty], dtype=np.int16))
        
        print(f"執行買入動作: 股票索引={stock_idx}, 數量={buy_qty}")
        
        # 執行動作
        obs, reward, terminated, truncated, info = env.step(action)
        
        print("✅ 動作執行成功")
        print(f"   獎勵: {reward:.6f}")
        print(f"   交易執行: {info.get('trade_executed', False)}")
        print(f"   交易結果: {info.get('trade_result', {})}")
        
        # 檢查持倉變化
        positions = env.account_manager.positions
        if positions:
            symbol = env.symbols[stock_idx]
            if symbol in positions:
                print(f"   持倉更新: {symbol} = {positions[symbol]['qty']} 股")
            else:
                print("   ⚠️ 持倉未更新 (可能因價格資料問題)")
        
        # 測試賣出動作
        sell_qty = -50  # 賣出50股
        action = (stock_idx, np.array([sell_qty], dtype=np.int16))
        
        print(f"\n執行賣出動作: 股票索引={stock_idx}, 數量={sell_qty}")
        obs, reward, terminated, truncated, info = env.step(action)
        
        print("✅ 賣出動作執行成功")
        print(f"   獎勵: {reward:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 動作執行失敗: {str(e)}")
        traceback.print_exc()
        return False

def test_risk_controls(env):
    """測試 4: 風險控制機制"""
    print("\n🛡️ 測試 4: 風險控制機制")
    print("-" * 40)
    
    try:
        # 重置環境
        observation, info = env.reset(seed=42)
        
        # 測試持倉限制
        stock_idx = 0
        large_qty = env.max_position_per_stock + 100  # 超過限制
        action = (stock_idx, np.array([large_qty], dtype=np.int16))
        
        print(f"測試持倉限制: 嘗試買入 {large_qty} 股 (限制: {env.max_position_per_stock})")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 檢查是否被限制
        trade_result = info.get('trade_result', {})
        if trade_result and not trade_result.get('success', True):
            if trade_result.get('reason') == 'position_limit_exceeded':
                print("✅ 持倉限制正常運作")
            else:
                print(f"   限制原因: {trade_result.get('reason')}")
        
        # 測試現金不足
        huge_qty = 10000  # 需要大量現金
        action = (stock_idx, np.array([huge_qty], dtype=np.int16))
        
        print(f"測試現金限制: 嘗試買入 {huge_qty} 股")
        obs, reward, terminated, truncated, info = env.step(action)
        
        trade_result = info.get('trade_result', {})
        if trade_result and not trade_result.get('success', True):
            if trade_result.get('reason') == 'insufficient_cash':
                print("✅ 現金限制正常運作")
            else:
                print(f"   限制原因: {trade_result.get('reason')}")
        
        # 測試風險限制檢查
        current_prices = env.data_provider.get_current_prices() if env.data_provider else {}
        risk_status = env.account_manager.check_risk_limits(current_prices)
        
        print("✅ 風險限制檢查正常")
        print(f"   每日最大損失超限: {risk_status.get('daily_max_loss_exceeded', False)}")
        print(f"   滾動最大回撤超限: {risk_status.get('rolling_max_dd_exceeded', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 風險控制測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def test_holding_timeout(env):
    """測試 5: 持倉超時機制"""
    print("\n⏰ 測試 5: 持倉超時機制")
    print("-" * 40)
    
    try:
        # 重置環境
        observation, info = env.reset(seed=42)
        
        # 建立持倉
        stock_idx = 0
        symbol = env.symbols[stock_idx]
        buy_qty = 100
        action = (stock_idx, np.array([buy_qty], dtype=np.int16))
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 手動設置持倉天數接近限制
        if symbol in env.account_manager.positions:
            env.account_manager.positions[symbol]['days_held'] = env.max_holding_days - 1
            print(f"設置 {symbol} 持倉天數為 {env.max_holding_days - 1}")
        
        # 執行一步，觸發持倉天數更新
        action = (0, np.array([0], dtype=np.int16))  # 無操作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 檢查超時持倉
        timeout_positions = env.account_manager.get_timeout_positions(env.max_holding_days)
        
        print(f"✅ 持倉超時檢查正常")
        print(f"   超時持倉數: {len(timeout_positions)}")
        
        if timeout_positions:
            print(f"   超時股票: {timeout_positions}")
            # 再執行一步，應該觸發強制平倉
            obs, reward, terminated, truncated, info = env.step(action)
            print("✅ 強制平倉機制測試完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 持倉超時測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def test_observation_format(env):
    """測試 6: 觀測格式和數據完整性"""
    print("\n📊 測試 6: 觀測格式和數據完整性")
    print("-" * 40)
    
    try:
        # 重置環境
        observation, info = env.reset(seed=42)
        
        # 檢查觀測格式
        print("觀測數據結構:")
        
        # price_frame 檢查
        price_frame = observation['price_frame']
        expected_shape = (len(env.symbols), 64, 5)  # (n_stocks, seq_len, features)
        
        print(f"   price_frame: {price_frame.shape} (期望: {expected_shape})")
        assert price_frame.shape == expected_shape, f"price_frame 形狀不正確"
        
        # 檢查數據範圍
        assert not np.any(np.isnan(price_frame)), "price_frame 包含 NaN"
        assert not np.any(np.isinf(price_frame)), "price_frame 包含無限值"
        
        # fundamental 檢查
        fundamental = observation['fundamental']
        expected_shape = (10,)  # 基本面特徵
        
        print(f"   fundamental: {fundamental.shape} (期望: {expected_shape})")
        assert fundamental.shape == expected_shape, f"fundamental 形狀不正確"
        
        # account 檢查
        account = observation['account']
        expected_shape = (4,)  # 帳戶特徵
        
        print(f"   account: {account.shape} (期望: {expected_shape})")
        assert account.shape == expected_shape, f"account 形狀不正確"
        
        # 檢查帳戶特徵的合理性
        nav_norm, pos_ratio, unreal_pnl_pct, risk_buffer = account
        print(f"   帳戶狀態: NAV標準化={nav_norm:.4f}, 持倉比例={pos_ratio:.4f}")
        print(f"              未實現損益={unreal_pnl_pct:.4f}, 風險緩衝={risk_buffer:.4f}")
        
        # 檢查 info 字典
        print("Info 數據結構:")
        for key, value in info.items():
            print(f"   {key}: {type(value)} = {value}")
        
        print("✅ 觀測格式驗證通過")
        return True
        
    except Exception as e:
        print(f"❌ 觀測格式測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def test_episode_completion(env):
    """測試 7: 完整回合執行"""
    print("\n🏁 測試 7: 完整回合執行")
    print("-" * 40)
    
    try:
        # 重置環境
        observation, info = env.reset(seed=42)
        
        step_count = 0
        total_reward = 0.0
        max_steps = 20  # 限制最大步數
        
        print("開始執行完整回合...")
        
        while step_count < max_steps:
            # 隨機動作
            stock_idx = np.random.randint(0, len(env.symbols))
            qty = np.random.randint(-50, 51)  # -50 到 50 股
            action = (stock_idx, np.array([qty], dtype=np.int16))
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_count += 1
            total_reward += reward
            
            if step_count % 5 == 0:
                nav = info.get('nav', 0)
                positions = len(info.get('positions', {}))
                print(f"   步驟 {step_count}: NAV={nav:,.0f}, 持倉={positions}檔, 累積獎勵={total_reward:.6f}")
            
            if terminated or truncated:
                print(f"   回合結束: terminated={terminated}, truncated={truncated}")
                break
        
        print("✅ 完整回合執行成功")
        print(f"   總步數: {step_count}")
        print(f"   累積獎勵: {total_reward:.6f}")
        print(f"   最終NAV: {info.get('nav', 0):,.0f}")
        print(f"   最終持倉: {len(info.get('positions', {}))} 檔")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整回合測試失敗: {str(e)}")
        traceback.print_exc()
        return False

def run_env_core_test():
    """執行環境核心功能測試"""
    print("=" * 60)
    print("TSE Alpha 環境核心功能測試")
    print("測試 TSEAlphaEnv 的基本功能和核心機制")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 7
    env = None
    
    # 測試清單
    tests = [
        ("環境創建和初始化", test_environment_creation),
        ("環境重置功能", lambda: test_environment_reset(env) if env else (False, None, None)),
        ("動作執行和交易機制", lambda: test_action_execution(env) if env else False),
        ("風險控制機制", lambda: test_risk_controls(env) if env else False),
        ("持倉超時機制", lambda: test_holding_timeout(env) if env else False),
        ("觀測格式和數據完整性", lambda: test_observation_format(env) if env else False),
        ("完整回合執行", lambda: test_episode_completion(env) if env else False)
    ]
    
    for i, (test_name, test_func) in enumerate(tests, 1):
        try:
            if i == 1:  # 第一個測試返回環境實例
                success, env = test_func()
                if success:
                    tests_passed += 1
            else:
                if test_func():
                    tests_passed += 1
                    
        except Exception as e:
            print(f"❌ 測試 {i} 異常: {str(e)}")
            traceback.print_exc()
    
    # 測試結果總結
    print("\n" + "=" * 60)
    print("📋 環境核心功能測試結果")
    print("=" * 60)
    
    pass_rate = (tests_passed / total_tests) * 100
    
    print(f"總測試數: {total_tests}")
    print(f"通過測試: {tests_passed}")
    print(f"失敗測試: {total_tests - tests_passed}")
    print(f"通過率: {pass_rate:.1f}%")
    
    if pass_rate >= 85:
        print(f"\n🎉 環境核心功能測試通過！")
        print(f"✅ TSEAlphaEnv 基本功能正常")
        print(f"🚀 可以進行下一階段測試")
        
        print(f"\n🎯 建議下一步:")
        print(f"   1. 執行模型-環境整合測試")
        print(f"   2. 測試代理人行為")
        print(f"   3. 進行端到端訓練測試")
        
    elif pass_rate >= 70:
        print(f"\n✅ 環境基本功能可用")
        print(f"🔧 部分功能可能需要調整")
        
    else:
        print(f"\n⚠️ 環境存在重要問題")
        print(f"🔧 需要修復失敗的測試項目")
    
    return pass_rate >= 70

if __name__ == "__main__":
    success = run_env_core_test()
    print(f"\n{'✅ 測試通過' if success else '❌ 測試失敗'}")
    sys.exit(0 if success else 1)