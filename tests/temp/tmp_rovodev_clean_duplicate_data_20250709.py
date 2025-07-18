#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 candlesticks_min 資料表中的重複資料
"""
import sys
from pathlib import Path

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "market_data_collector"))

def check_duplicate_data():
    """檢查重複資料"""
    print("🔍 檢查 candlesticks_min 資料表中的重複資料...")
    
    try:
        from market_data_collector.utils.db import query_df
        
        # 檢查重複資料
        duplicate_query = """
        SELECT market, symbol, timestamp, COUNT(*) as count
        FROM candlesticks_min
        GROUP BY market, symbol, timestamp
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        """
        
        duplicates = query_df(duplicate_query)
        
        if not duplicates.empty:
            print(f"❌ 發現 {len(duplicates)} 組重複資料:")
            print(duplicates.head(10))
            return len(duplicates)
        else:
            print("✅ 沒有發現重複資料")
            return 0
            
    except Exception as e:
        print(f"❌ 檢查重複資料失敗: {e}")
        return -1

def check_data_quality():
    """檢查資料品質"""
    print("\n📊 檢查資料品質...")
    
    try:
        from market_data_collector.utils.db import query_df
        
        # 檢查總資料量
        count_query = "SELECT COUNT(*) as total_count FROM candlesticks_min"
        count_result = query_df(count_query)
        total_count = count_result['total_count'].iloc[0] if not count_result.empty else 0
        print(f"📈 總資料筆數: {total_count}")
        
        # 檢查股票數量
        symbol_query = "SELECT COUNT(DISTINCT symbol) as symbol_count FROM candlesticks_min"
        symbol_result = query_df(symbol_query)
        symbol_count = symbol_result['symbol_count'].iloc[0] if not symbol_result.empty else 0
        print(f"📊 股票數量: {symbol_count}")
        
        # 檢查時間範圍
        time_query = "SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time FROM candlesticks_min"
        time_result = query_df(time_query)
        if not time_result.empty:
            print(f"📅 時間範圍: {time_result['min_time'].iloc[0]} ~ {time_result['max_time'].iloc[0]}")
        
        # 檢查問題資料
        problem_query = """
        SELECT 
            SUM(CASE WHEN market = '0' OR market = 0 THEN 1 ELSE 0 END) as bad_market,
            SUM(CASE WHEN symbol = '0' OR symbol = 0 THEN 1 ELSE 0 END) as bad_symbol,
            SUM(CASE WHEN open = 0 AND high = 0 AND low = 0 AND close = 0 THEN 1 ELSE 0 END) as zero_prices
        FROM candlesticks_min
        """
        problem_result = query_df(problem_query)
        
        if not problem_result.empty:
            bad_market = problem_result['bad_market'].iloc[0]
            bad_symbol = problem_result['bad_symbol'].iloc[0]
            zero_prices = problem_result['zero_prices'].iloc[0]
            
            print(f"⚠️  問題資料統計:")
            print(f"   market 欄位異常: {bad_market} 筆")
            print(f"   symbol 欄位異常: {bad_symbol} 筆")
            print(f"   價格全為 0: {zero_prices} 筆")
        
        # 顯示樣本資料
        sample_query = "SELECT * FROM candlesticks_min LIMIT 5"
        sample_result = query_df(sample_query)
        
        if not sample_result.empty:
            print("\n📋 樣本資料:")
            print(sample_result)
        
    except Exception as e:
        print(f"❌ 檢查資料品質失敗: {e}")

def clean_duplicate_data():
    """清理重複資料"""
    print("\n🧹 清理重複資料...")
    
    try:
        from market_data_collector.utils.db import execute_sql
        
        # 建立臨時表保留唯一資料
        clean_sql = """
        CREATE TABLE candlesticks_min_clean AS
        SELECT market, symbol, timestamp, open, high, low, close, volume, interval
        FROM candlesticks_min
        GROUP BY market, symbol, timestamp
        """
        
        # 刪除原表
        drop_sql = "DROP TABLE candlesticks_min"
        
        # 重命名臨時表
        rename_sql = "ALTER TABLE candlesticks_min_clean RENAME TO candlesticks_min"
        
        # 重建索引
        index_sql = """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_min_market_symbol_timestamp 
        ON candlesticks_min (market, symbol, timestamp)
        """
        
        print("🔧 執行清理...")
        execute_sql(clean_sql)
        execute_sql(drop_sql)
        execute_sql(rename_sql)
        execute_sql(index_sql)
        
        print("✅ 重複資料清理完成")
        
        # 重新檢查
        return check_duplicate_data()
        
    except Exception as e:
        print(f"❌ 清理重複資料失敗: {e}")
        return -1

def main():
    """主函數"""
    print("=" * 60)
    print("🧹 candlesticks_min 資料表清理工具")
    print("=" * 60)
    
    # 檢查資料品質
    check_data_quality()
    
    # 檢查重複資料
    duplicate_count = check_duplicate_data()
    
    if duplicate_count > 0:
        print(f"\n發現 {duplicate_count} 組重複資料")
        choice = input("是否要清理重複資料? (y/n): ").strip().lower()
        
        if choice == 'y':
            result = clean_duplicate_data()
            if result == 0:
                print("🎉 資料清理完成，沒有重複資料")
            else:
                print("❌ 清理過程中發生問題")
        else:
            print("跳過清理")
    else:
        print("✅ 資料表狀態良好")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()