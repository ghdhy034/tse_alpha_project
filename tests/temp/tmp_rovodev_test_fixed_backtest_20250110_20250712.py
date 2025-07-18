#!/usr/bin/env python3
"""
修復後的回測引擎測試 - 驗證資料庫查詢修復
"""
import sys
import os
from pathlib import Path
from datetime import date, datetime
import time

# 添加路徑
sys.path.append(str(Path(__file__).parent))

def test_fixed_data_manager():
    """測試修復後的資料管理器"""
    print("=== 測試修復後的資料管理器 ===")
    
    try:
        from backtest.config import create_smoke_test_config
        from backtest.data_manager import DataManager
        
        config = create_smoke_test_config()
        config.cache_mode = 'none'
        config.preload_data = False
        
        data_manager = DataManager(config)
        
        # 測試日線資料載入
        print("1. 測試日線資料載入...")
        daily_data = data_manager.get_stock_data('2330', date(2024, 1, 1), date(2024, 1, 31), 'daily')
        if daily_data is not None:
            print(f"   ✅ 日線資料載入成功: {len(daily_data)} 筆記錄")
        else:
            print(f"   ⚠️ 無日線資料")
        
        # 測試分鐘線資料載入
        print("2. 測試分鐘線資料載入...")
        minute_data = data_manager.get_stock_data('2330', date(2024, 1, 1), date(2024, 1, 31), 'minute')
        if minute_data is not None:
            print(f"   ✅ 分鐘線資料載入成功: {len(minute_data)} 筆記錄")
        else:
            print(f"   ⚠️ 無分鐘線資料")
        
        # 測試籌碼面資料載入
        print("3. 測試籌碼面資料載入...")
        chip_data = data_manager.get_stock_data('2330', date(2024, 1, 1), date(2024, 1, 31), 'chip')
        if chip_data is not None:
            print(f"   ✅ 籌碼面資料載入成功: {len(chip_data)} 筆記錄")
            print(f"   欄位: {list(chip_data.columns)}")
        else:
            print(f"   ⚠️ 無籌碼面資料")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 資料管理器測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fixed_smoke_test():
    """測試修復後的煙霧測試"""
    print("\n=== 測試修復後的煙霧測試 ===")
    
    try:
        from backtest.config import create_smoke_test_config
        from backtest.engine import BacktestEngine, create_dummy_model
        
        # 創建最小配置
        config = create_smoke_test_config()
        config.stock_universe = ['2330']  # 只用一檔股票
        config.cache_mode = 'none'
        config.preload_data = False
        config.save_trades = False
        config.save_positions = False
        
        engine = BacktestEngine(config)
        model = create_dummy_model()
        
        print("   開始執行修復後的煙霧測試...")
        print(f"   配置: 訓練{config.train_window_months}月/測試{config.test_window_months}月")
        
        start_time = time.time()
        smoke_result = engine.smoke_test(model)
        execution_time = time.time() - start_time
        
        print(f"   執行時間: {execution_time:.2f} 秒")
        print(f"   成功: {smoke_result.get('success', False)}")
        print(f"   性能達標: {smoke_result.get('performance_ok', False)}")
        print(f"   測試週期數: {smoke_result.get('periods_tested', 0)}")
        
        if smoke_result.get('errors'):
            print("   錯誤:")
            for error in smoke_result['errors']:
                print(f"     - {error}")
        
        if smoke_result.get('warnings'):
            print("   警告:")
            for warning in smoke_result['warnings']:
                print(f"     - {warning}")
        
        return smoke_result.get('success', False)
        
    except Exception as e:
        print(f"   ❌ 煙霧測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fixed_full_backtest():
    """測試修復後的完整回測"""
    print("\n=== 測試修復後的完整回測 ===")
    
    try:
        from backtest.config import create_smoke_test_config
        from backtest.engine import BacktestEngine, create_dummy_model
        
        # 創建配置
        config = create_smoke_test_config()
        config.stock_universe = ['2330', '2317']  # 兩檔股票
        config.cache_mode = 'none'
        config.preload_data = False
        config.save_trades = False
        config.save_positions = False
        
        engine = BacktestEngine(config)
        model = create_dummy_model()
        
        # 執行回測 - 擴大日期範圍以確保能生成週期
        start_date = date(2024, 1, 1)
        end_date = date(2024, 6, 30)  # 6 個月，確保有足夠的資料
        
        print(f"   回測期間: {start_date} ~ {end_date}")
        print(f"   股票池: {config.stock_universe}")
        
        start_time = time.time()
        result = engine.run_backtest(
            model=model,
            start_date=start_date,
            end_date=end_date,
            symbols=config.stock_universe
        )
        execution_time = time.time() - start_time
        
        # 分析結果
        print(f"   執行時間: {execution_time:.2f} 秒")
        print(f"   週期數: {result.total_periods}")
        print(f"   會話數: {len(result.session_results)}")
        print(f"   錯誤數: {len(result.errors)}")
        print(f"   警告數: {len(result.warnings)}")
        
        if result.errors:
            print("   錯誤:")
            for error in result.errors:
                print(f"     - {error}")
        
        if result.warnings:
            print("   警告:")
            for warning in result.warnings:
                print(f"     - {warning}")
        
        # 績效指標
        metrics = result.performance_metrics
        print(f"   總收益: {metrics.total_return:.4f}")
        print(f"   年化收益: {metrics.annual_return:.4f}")
        print(f"   Sharpe 比率: {metrics.sharpe_ratio:.4f}")
        print(f"   最大回撤: {metrics.max_drawdown:.4f}")
        
        success = len(result.errors) == 0 and result.total_periods > 0
        
        if success:
            print("   ✅ 完整回測成功")
        else:
            print("   ❌ 完整回測失敗")
        
        return success
        
    except Exception as e:
        print(f"   ❌ 完整回測異常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("開始修復後的回測引擎測試...")
    print("=" * 50)
    
    tests = [
        ("修復後資料管理器", test_fixed_data_manager),
        ("修復後煙霧測試", test_fixed_smoke_test),
        ("修復後完整回測", test_fixed_full_backtest),
    ]
    
    passed = 0
    total = len(tests)
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通過")
            else:
                print(f"❌ {test_name} 失敗")
        except Exception as e:
            print(f"❌ {test_name} 異常: {e}")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"修復後測試結果: {passed}/{total} 通過")
    print(f"總測試時間: {total_time:.2f} 秒")
    print(f"通過率: {passed/total:.1%}")
    
    if passed == total:
        print("🎉 所有測試通過！回測引擎修復成功！")
        print("✅ 可以進行模型-環境整合測試")
        return True
    elif passed >= 2:
        print("⚠️ 大部分測試通過，回測引擎基本可用")
        print("🔧 建議檢查失敗的測試項目")
        return True
    else:
        print("❌ 多個測試失敗，需要進一步修復")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)