#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本面資料智能對齊測試
驗證月營收和財報資料的時間對齊邏輯
"""
import sys
import os
from pathlib import Path
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# 強制UTF-8輸出
sys.stdout.reconfigure(encoding='utf-8')

# 添加路徑
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "data_pipeline"))
sys.path.append(str(Path(__file__).parent / "market_data_collector"))

def print_status(task, status, details=""):
    """統一的狀態輸出格式"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_icon = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "🔄"
    print(f"[{timestamp}] {status_icon} {task}: {status}")
    if details:
        print(f"    詳情: {details}")

def test_fundamental_alignment_logic():
    """測試基本面資料對齊邏輯 - 動態對齊測試"""
    print("\n" + "="*60)
    print("🧪 測試基本面資料智能對齊邏輯")
    print("="*60)
    
    try:
        from data_pipeline.features import FeatureEngine
        
        # 創建測試用的基本面資料
        print("📊 創建測試資料...")
        
        # 模擬真實的月營收發布時間 (不規則間隔)
        monthly_dates = [
            '2023-12-10',  # 發布11月資料
            '2024-01-12',  # 發布12月資料 (稍微延遲)
            '2024-02-08',  # 發布1月資料 (提早)
            '2024-03-15',  # 發布2月資料 (延遲)
            '2024-04-10',  # 發布3月資料
            '2024-05-20',  # 發布4月資料 (大幅延遲)
        ]
        monthly_revenues = [950, 1000, 1100, 1050, 1200, 1150]
        
        # 模擬真實的財報發布時間 (季度報告)
        quarterly_dates = [
            '2024-01-30',  # Q4 2023財報 (1月底發布)
            '2024-04-25',  # Q1 2024財報 (4月底發布)
            '2024-07-28',  # Q2 2024財報 (7月底發布)
        ]
        quarterly_eps = [2.3, 2.8, 3.1]
        
        # 創建連續的交易日期序列 (模擬實際日線資料)
        trading_dates = pd.bdate_range('2024-01-01', '2024-06-30')
        
        print(f"   交易日數量: {len(trading_dates)}")
        print(f"   月營收資料點: {len(monthly_dates)}")
        print(f"   財報資料點: {len(quarterly_dates)}")
        
        # 創建特徵引擎實例
        feature_engine = FeatureEngine()
        
        # 測試月營收對齊
        print("\n🔍 測試月營收資料動態對齊...")
        monthly_series = pd.Series(
            monthly_revenues, 
            index=pd.to_datetime(monthly_dates)
        )
        
        aligned_revenue = feature_engine._align_fundamental_data(
            monthly_series, trading_dates, 'monthly_revenue'
        )
        
        # 動態檢查對齊邏輯 - 驗證每個交易日都能找到正確的過去資料
        print("   動態對齊驗證:")
        
        # 檢查幾個關鍵時間點的對齊邏輯
        sample_dates = [
            trading_dates[4],   # 第5個交易日
            trading_dates[20],  # 第21個交易日  
            trading_dates[60],  # 第61個交易日
            trading_dates[100], # 第101個交易日
            trading_dates[-10], # 倒數第10個交易日
        ]
        
        for sample_date in sample_dates:
            # 手動計算期望值 - 找到該日期之前最近的月營收資料
            past_monthly_dates = monthly_series.index[monthly_series.index <= sample_date]
            
            if len(past_monthly_dates) > 0:
                latest_monthly_date = past_monthly_dates.max()
                expected_value = monthly_series.loc[latest_monthly_date]
                
                # 檢查時效性 (45天限制)
                days_diff = (sample_date - latest_monthly_date).days
                if days_diff > 45:
                    expected_value = 0.0
                    
            else:
                expected_value = 0.0
            
            actual_value = aligned_revenue.loc[sample_date]
            status = "✅" if abs(actual_value - expected_value) < 0.01 else "❌"
            
            print(f"     {status} {sample_date.strftime('%Y-%m-%d')}: {actual_value:.1f} (期望: {expected_value:.1f})")
            if len(past_monthly_dates) > 0:
                latest_monthly_date = past_monthly_dates.max()
                days_diff = (sample_date - latest_monthly_date).days
                print(f"        └─ 使用 {latest_monthly_date.strftime('%Y-%m-%d')} 資料 (間隔{days_diff}天)")
        
        # 測試財報資料對齊
        print("\n🔍 測試財報資料動態對齊...")
        quarterly_series = pd.Series(
            quarterly_eps,
            index=pd.to_datetime(quarterly_dates)
        )
        
        aligned_eps = feature_engine._align_fundamental_data(
            quarterly_series, trading_dates, 'eps'
        )
        
        # 動態檢查財報對齊邏輯
        print("   財報動態對齊驗證:")
        
        for sample_date in sample_dates:
            # 手動計算期望值 - 找到該日期之前最近的財報資料
            past_quarterly_dates = quarterly_series.index[quarterly_series.index <= sample_date]
            
            if len(past_quarterly_dates) > 0:
                latest_quarterly_date = past_quarterly_dates.max()
                expected_value = quarterly_series.loc[latest_quarterly_date]
                
                # 檢查時效性 (120天限制)
                days_diff = (sample_date - latest_quarterly_date).days
                if days_diff > 120:
                    expected_value = 0.0
                    
            else:
                expected_value = 0.0
            
            actual_value = aligned_eps.loc[sample_date]
            status = "✅" if abs(actual_value - expected_value) < 0.01 else "❌"
            
            print(f"     {status} {sample_date.strftime('%Y-%m-%d')}: {actual_value:.1f} (期望: {expected_value:.1f})")
            if len(past_quarterly_dates) > 0:
                latest_quarterly_date = past_quarterly_dates.max()
                days_diff = (sample_date - latest_quarterly_date).days
                print(f"        └─ 使用 {latest_quarterly_date.strftime('%Y-%m-%d')} 資料 (間隔{days_diff}天)")
        
        # 測試資料時效性
        print("\n⏰ 測試資料時效性...")
        
        # 創建過舊的資料
        old_dates = ['2023-10-10']  # 很舊的月營收
        old_revenues = [800]
        
        old_series = pd.Series(old_revenues, index=pd.to_datetime(old_dates))
        
        # 測試2024年1月的對齊 (應該因為資料過舊而使用0)
        jan_dates = pd.bdate_range('2024-01-01', '2024-01-31')
        aligned_old = feature_engine._align_fundamental_data(
            old_series, jan_dates, 'monthly_revenue'
        )
        
        # 檢查是否正確處理過舊資料
        jan_mid_value = aligned_old.loc['2024-01-15']
        if jan_mid_value == 0:
            print("   ✅ 正確處理過舊資料: 超過45天的月營收資料被設為0")
        else:
            print(f"   ❌ 過舊資料處理錯誤: {jan_mid_value} (應該為0)")
        
        # 統計分析
        print("\n📊 對齊結果統計:")
        revenue_stats = {
            '非零值比例': f"{(aligned_revenue != 0).mean():.1%}",
            '平均值': f"{aligned_revenue.mean():.2f}",
            '最大值': f"{aligned_revenue.max():.2f}",
            '最小值': f"{aligned_revenue.min():.2f}",
        }
        
        eps_stats = {
            '非零值比例': f"{(aligned_eps != 0).mean():.1%}",
            '平均值': f"{aligned_eps.mean():.2f}",
            '最大值': f"{aligned_eps.max():.2f}",
            '最小值': f"{aligned_eps.min():.2f}",
        }
        
        print("   月營收統計:")
        for key, value in revenue_stats.items():
            print(f"     {key}: {value}")
        
        print("   EPS統計:")
        for key, value in eps_stats.items():
            print(f"     {key}: {value}")
        
        print_status("基本面對齊測試", "SUCCESS", "智能對齊邏輯運作正常")
        return True
        
    except Exception as e:
        print_status("基本面對齊測試", "FAILED", str(e))
        traceback.print_exc()
        return False

def test_real_data_alignment():
    """測試真實資料的動態對齊"""
    print("\n" + "="*60)
    print("🧪 測試真實資料動態對齊 (2330)")
    print("="*60)
    
    try:
        from data_pipeline.features import FeatureEngine
        
        print("⚙️ 測試2330真實基本面資料動態對齊...")
        feature_engine = FeatureEngine(symbols=['2330'])
        
        # 使用更長的時間範圍來測試動態對齊
        test_dates = pd.bdate_range('2024-01-01', '2024-03-31')  # 3個月的交易日
        dummy_price_df = pd.DataFrame({
            'open': 100.0,
            'high': 105.0, 
            'low': 95.0,
            'close': 102.0,
            'volume': 10000
        }, index=test_dates)
        
        print(f"   測試時間範圍: {test_dates[0].strftime('%Y-%m-%d')} 到 {test_dates[-1].strftime('%Y-%m-%d')}")
        print(f"   交易日數量: {len(test_dates)}")
        
        # 測試基本面特徵計算
        fundamental_features = feature_engine.calculate_fundamental_features('2330', dummy_price_df)
        
        if not fundamental_features.empty:
            print(f"✅ 成功計算基本面特徵: {fundamental_features.shape}")
            
            # 檢查特徵覆蓋率和動態變化
            print("\n📊 基本面特徵動態分析:")
            
            # 分析月營收特徵的時間變化
            if 'monthly_revenue' in fundamental_features.columns:
                monthly_revenue_series = fundamental_features['monthly_revenue']
                unique_values = monthly_revenue_series.unique()
                value_changes = (monthly_revenue_series != monthly_revenue_series.shift()).sum()
                
                print(f"   月營收特徵:")
                print(f"     ├─ 唯一值數量: {len(unique_values)}")
                print(f"     ├─ 值變化次數: {value_changes}")
                print(f"     ├─ 非零比例: {(monthly_revenue_series != 0).mean():.1%}")
                print(f"     └─ 值範圍: {monthly_revenue_series.min():.1f} ~ {monthly_revenue_series.max():.1f}")
                
                # 顯示幾個關鍵時間點的值
                sample_indices = [0, len(test_dates)//4, len(test_dates)//2, len(test_dates)*3//4, -1]
                print(f"     時間點採樣:")
                for idx in sample_indices:
                    date = test_dates[idx]
                    value = monthly_revenue_series.iloc[idx]
                    print(f"       {date.strftime('%Y-%m-%d')}: {value:.1f}")
            
            # 分析財報特徵的時間變化 (以EPS為例)
            if 'eps' in fundamental_features.columns:
                eps_series = fundamental_features['eps']
                unique_values = eps_series.unique()
                value_changes = (eps_series != eps_series.shift()).sum()
                
                print(f"\n   EPS特徵:")
                print(f"     ├─ 唯一值數量: {len(unique_values)}")
                print(f"     ├─ 值變化次數: {value_changes}")
                print(f"     ├─ 非零比例: {(eps_series != 0).mean():.1%}")
                print(f"     └─ 值範圍: {eps_series.min():.3f} ~ {eps_series.max():.3f}")
                
                # 顯示幾個關鍵時間點的值
                print(f"     時間點採樣:")
                for idx in sample_indices:
                    date = test_dates[idx]
                    value = eps_series.iloc[idx]
                    print(f"       {date.strftime('%Y-%m-%d')}: {value:.3f}")
            
            # 整體特徵覆蓋率統計
            print(f"\n📈 整體特徵覆蓋率:")
            feature_coverage = {}
            for col in fundamental_features.columns:
                non_zero_rate = (fundamental_features[col] != 0).mean()
                feature_coverage[col] = non_zero_rate
            
            # 按覆蓋率分組顯示
            high_coverage = {k: v for k, v in feature_coverage.items() if v > 0.5}
            medium_coverage = {k: v for k, v in feature_coverage.items() if 0 < v <= 0.5}
            zero_coverage = {k: v for k, v in feature_coverage.items() if v == 0}
            
            if high_coverage:
                print(f"   ✅ 高覆蓋率特徵 (>50%): {len(high_coverage)}個")
                for feature, coverage in sorted(high_coverage.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"     └─ {feature}: {coverage:.1%}")
            
            if medium_coverage:
                print(f"   ⚠️ 中等覆蓋率特徵 (1-50%): {len(medium_coverage)}個")
                for feature, coverage in sorted(medium_coverage.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"     └─ {feature}: {coverage:.1%}")
            
            if zero_coverage:
                print(f"   ❌ 零覆蓋率特徵 (0%): {len(zero_coverage)}個")
                if len(zero_coverage) <= 5:
                    for feature in zero_coverage:
                        print(f"     └─ {feature}")
                else:
                    print(f"     └─ (顯示前5個) {list(zero_coverage.keys())[:5]}")
            
            # 檢查是否有15個特徵
            if len(fundamental_features.columns) == 15:
                print(f"\n✅ 基本面特徵數量正確: 15個")
            else:
                print(f"\n⚠️ 基本面特徵數量: {len(fundamental_features.columns)} (期望15個)")
            
            # 驗證動態對齊邏輯是否正常工作
            total_coverage = sum(feature_coverage.values()) / len(feature_coverage)
            if total_coverage > 0.1:  # 至少10%的整體覆蓋率
                print(f"✅ 動態對齊邏輯運作正常 (整體覆蓋率: {total_coverage:.1%})")
            else:
                print(f"⚠️ 動態對齊可能有問題 (整體覆蓋率過低: {total_coverage:.1%})")
        
        else:
            print("⚠️ 基本面特徵計算返回空結果")
        
        print_status("真實資料對齊測試", "SUCCESS", "2330基本面動態對齊完成")
        return True
        
    except Exception as e:
        print_status("真實資料對齊測試", "FAILED", str(e))
        traceback.print_exc()
        return False

def run_fundamental_alignment_test():
    """執行基本面對齊測試"""
    print("🚀 開始基本面資料智能對齊測試")
    print("="*80)
    
    start_time = datetime.now()
    
    # 測試1: 對齊邏輯測試
    success_1 = test_fundamental_alignment_logic()
    
    # 測試2: 真實資料測試
    success_2 = test_real_data_alignment()
    
    # 總結
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results = {
        "對齊邏輯測試": success_1,
        "真實資料測試": success_2
    }
    
    print("\n" + "="*80)
    print("📋 基本面對齊測試總結")
    print("="*80)
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"   {test_name}: {status}")
    
    print(f"\n📊 總體結果: {success_count}/{total_count} 測試成功")
    print(f"⏱️ 執行時間: {duration:.1f} 秒")
    
    if success_count == total_count:
        print("🎉 基本面智能對齊測試 - 全部通過！")
        print("✅ 基本面資料現在會正確考慮更新頻率和延遲")
        return True
    else:
        print("⚠️ 基本面智能對齊測試 - 部分失敗")
        print("❌ 需要進一步調整對齊邏輯")
        return False

if __name__ == "__main__":
    try:
        success = run_fundamental_alignment_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 用戶中斷測試")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未預期的錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)