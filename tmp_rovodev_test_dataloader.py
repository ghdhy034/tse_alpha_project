#!/usr/bin/env python3
"""
TSE Alpha 資料載入器功能測試
測試完善後的資料載入器是否能正常運作
"""

import sys
import sqlite3
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import traceback

def test_database_connection():
    """測試資料庫連接"""
    print("🔍 測試資料庫連接")
    print("-" * 30)
    
    db_path = Path("market_data_collector/data/stock_data.db")
    
    if not db_path.exists():
        print(f"❌ 資料庫不存在: {db_path}")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print(f"✅ 資料庫連接成功")
            print(f"   發現 {len(tables)} 個資料表")
            
            # 顯示資料表
            for table in tables[:8]:  # 顯示前8個
                table_name = table[0]
                count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = count_cursor.fetchone()[0]
                print(f"   📊 {table_name}: {count:,} 筆記錄")
            
            return True
            
    except Exception as e:
        print(f"❌ 資料庫連接失敗: {e}")
        return False

def test_db_structure_analysis():
    """測試資料庫結構分析"""
    print("\n📊 測試資料庫結構分析")
    print("-" * 30)
    
    db_structure_path = Path("db_structure.json")
    
    if not db_structure_path.exists():
        print(f"❌ db_structure.json 不存在")
        return False
    
    try:
        with open(db_structure_path, 'r', encoding='utf-8') as f:
            db_structure = json.load(f)
        
        print(f"✅ 成功載入資料庫結構")
        print(f"   資料庫: {db_structure['database']}")
        print(f"   資料表數: {len(db_structure['tables'])}")
        
        # 分析每個資料表
        feature_count = 0
        for table_data in db_structure['tables']:
            table_name = table_data['table_name']
            columns = [col['name'] for col in table_data['columns']]
            
            # 計算特徵欄位 (排除 id, market, symbol, date/timestamp)
            feature_columns = [col for col in columns 
                             if col not in ['id', 'market', 'symbol', 'date', 'timestamp']]
            
            feature_count += len(feature_columns)
            print(f"   📋 {table_name}: {len(feature_columns)} 個特徵欄位")
        
        print(f"\n🎯 總特徵數: {feature_count}")
        return True
        
    except Exception as e:
        print(f"❌ 資料庫結構分析失敗: {e}")
        return False

def test_simple_query():
    """測試簡單查詢"""
    print("\n🔍 測試簡單查詢")
    print("-" * 30)
    
    db_path = Path("market_data_collector/data/stock_data.db")
    
    if not db_path.exists():
        print("❌ 資料庫不存在，跳過查詢測試")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            # 測試基本查詢
            print("1. 測試日線資料查詢...")
            query = """
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM candlesticks_daily 
            WHERE symbol = '2330' 
            ORDER BY timestamp DESC 
            LIMIT 5
            """
            
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                print(f"   ✅ 成功查詢 {len(df)} 筆日線資料")
                print(f"   📅 最新日期: {df.iloc[0]['timestamp']}")
                print(f"   💰 最新收盤價: {df.iloc[0]['close']}")
            else:
                print("   ⚠️ 沒有找到 2330 的資料")
            
            # 測試技術指標查詢
            print("\n2. 測試技術指標查詢...")
            tech_query = """
            SELECT symbol, date, sma_5, sma_20, rsi_14, macd
            FROM technical_indicators 
            WHERE symbol = '2330' 
            ORDER BY date DESC 
            LIMIT 3
            """
            
            tech_df = pd.read_sql_query(tech_query, conn)
            
            if not tech_df.empty:
                print(f"   ✅ 成功查詢 {len(tech_df)} 筆技術指標")
                print(f"   📈 最新 SMA5: {tech_df.iloc[0]['sma_5']:.2f}")
            else:
                print("   ⚠️ 沒有找到技術指標資料")
            
            return True
            
    except Exception as e:
        print(f"❌ 查詢測試失敗: {e}")
        return False

def test_join_query():
    """測試 JOIN 查詢"""
    print("\n🔗 測試 JOIN 查詢")
    print("-" * 30)
    
    db_path = Path("market_data_collector/data/stock_data.db")
    
    if not db_path.exists():
        print("❌ 資料庫不存在，跳過 JOIN 測試")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            # 測試日線 + 技術指標 JOIN
            join_query = """
            SELECT 
                cd.symbol,
                cd.timestamp as date,
                cd.open, cd.high, cd.low, cd.close, cd.volume,
                ti.sma_5, ti.sma_20, ti.rsi_14, ti.macd
            FROM candlesticks_daily cd
            LEFT JOIN technical_indicators ti 
                ON cd.symbol = ti.symbol 
                AND DATE(cd.timestamp) = DATE(ti.date)
            WHERE cd.symbol = '2330' 
                AND DATE(cd.timestamp) BETWEEN '2024-01-01' AND '2024-01-10'
            ORDER BY cd.timestamp ASC
            LIMIT 5
            """
            
            df = pd.read_sql_query(join_query, conn)
            
            if not df.empty:
                print(f"   ✅ JOIN 查詢成功: {len(df)} 筆記錄")
                print(f"   📊 欄位數: {len(df.columns)}")
                print(f"   🔗 JOIN 成功率: {(df['sma_5'].notna().sum() / len(df) * 100):.1f}%")
                
                # 顯示樣本資料
                print(f"\n   📋 樣本資料:")
                for i, row in df.head(2).iterrows():
                    print(f"      {row['date']}: 收盤 {row['close']:.2f}, SMA5 {row['sma_5']:.2f if pd.notna(row['sma_5']) else 'N/A'}")
                
            else:
                print("   ⚠️ JOIN 查詢沒有結果")
            
            return True
            
    except Exception as e:
        print(f"❌ JOIN 查詢失敗: {e}")
        traceback.print_exc()
        return False

def test_stock_config():
    """測試股票配置"""
    print("\n📈 測試股票配置")
    print("-" * 30)
    
    try:
        from stock_config import STOCK_SPLITS, get_split_stocks, validate_splits
        
        print("✅ 成功導入股票配置")
        
        # 驗證分割
        is_valid, message = validate_splits()
        print(f"   📊 分割驗證: {message}")
        
        # 顯示分割統計
        for split_name in ['train', 'validation', 'test']:
            stocks = get_split_stocks(split_name)
            print(f"   📋 {split_name}: {len(stocks)} 支股票")
            print(f"      範例: {stocks[:3]}")
        
        return is_valid
        
    except ImportError as e:
        print(f"❌ 無法導入股票配置: {e}")
        return False

def test_feature_dimensions():
    """測試特徵維度計算"""
    print("\n🎯 測試特徵維度計算")
    print("-" * 30)
    
    try:
        # 模擬特徵映射 (基於實際資料庫結構)
        feature_mapping = {
            'price_volume_basic': ['open', 'high', 'low', 'close', 'volume'],  # 5個
            'technical_indicators': [  # 預期約17個
                'sma_5', 'sma_20', 'sma_60', 'ema_12', 'ema_26', 'ema_50',
                'macd', 'macd_signal', 'macd_hist', 'keltner_upper', 'keltner_middle', 
                'keltner_lower', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                'rsi_14', 'bandwidth'
            ],
            'financials': [  # 預期約20個
                'revenue', 'operating_income', 'net_income', 'total_assets',
                'total_liabilities', 'shareholders_equity', 'cash_flow_operating',
                'cash_flow_investing', 'cash_flow_financing', 'eps'
            ],
            'monthly_revenue': ['monthly_revenue'],  # 1個
            'margin_purchase_shortsale': [  # 預期約13個
                'margin_purchase_buy', 'margin_purchase_sell', 'short_sale_buy',
                'short_sale_sell', 'margin_purchase_today_balance', 'short_sale_today_balance'
            ],
            'institutional_investors': [  # 預期約10個
                'foreign_investor_buy', 'foreign_investor_sell', 'investment_trust_buy',
                'investment_trust_sell', 'dealer_buy', 'dealer_sell'
            ],
            'financial_per': ['pe_ratio', 'pb_ratio', 'dividend_yield']  # 3個
        }
        
        total_features = 0
        print("📊 特徵維度分析:")
        
        for category, features in feature_mapping.items():
            count = len(features)
            total_features += count
            print(f"   {category}: {count} 個特徵")
        
        print(f"\n🎯 總計: {total_features} 個特徵")
        
        # 與 SSOT 規範比較
        ssot_target = 70
        if total_features == ssot_target:
            print(f"✅ 特徵維度符合 SSOT 規範 ({ssot_target})")
        else:
            print(f"⚠️ 特徵維度與 SSOT 不符: {total_features} != {ssot_target}")
            if total_features < ssot_target:
                print(f"   需要補充 {ssot_target - total_features} 個特徵")
            else:
                print(f"   需要減少 {total_features - ssot_target} 個特徵")
        
        return True
        
    except Exception as e:
        print(f"❌ 特徵維度測試失敗: {e}")
        return False

def run_comprehensive_test():
    """執行綜合測試"""
    print("🚀 TSE Alpha 資料載入器功能測試")
    print("=" * 50)
    
    tests = [
        ("資料庫連接", test_database_connection),
        ("資料庫結構分析", test_db_structure_analysis),
        ("簡單查詢", test_simple_query),
        ("JOIN 查詢", test_join_query),
        ("股票配置", test_stock_config),
        ("特徵維度", test_feature_dimensions)
    ]
    
    results = {}
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} 測試異常: {e}")
            results[test_name] = False
    
    # 生成測試報告
    print(f"\n{'='*50}")
    print("📋 測試結果總覽")
    print(f"{'='*50}")
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    success_rate = (passed_tests / len(tests)) * 100
    print(f"\n📊 測試統計:")
    print(f"   通過: {passed_tests}/{len(tests)} 項測試")
    print(f"   成功率: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\n🎉 資料載入器功能基本正常！")
        print("💡 建議: 可以開始整合到訓練模組")
    elif success_rate >= 60:
        print("\n⚠️ 資料載入器部分功能正常")
        print("🔧 建議: 修復失敗的測試項目後再整合")
    else:
        print("\n❌ 資料載入器需要重大修復")
        print("🛠️ 建議: 檢查資料庫和配置文件")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_test()