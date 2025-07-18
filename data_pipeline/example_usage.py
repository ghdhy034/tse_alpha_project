#!/usr/bin/env python3
"""
fetch_minute.py 使用範例
"""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent / "market_data_collector"))

from fetch_minute import fetch_symbol_date, store_minute_bars
from market_data_collector.utils.db import get_database_info, create_minute_bars_table

def example_download_single_stock():
    """範例：下載單一股票的分鐘線資料"""
    print("=== 範例：下載單一股票分鐘線資料 ===")
    
    # 設定參數
    symbol = '2330'  # 台積電
    target_date = date(2024, 1, 2)  # 測試日期
    
    print(f"下載 {symbol} {target_date} 的分鐘線資料...")
    
    # 下載資料
    df = fetch_symbol_date(symbol, target_date)
    
    if not df.empty:
        print(f"✅ 下載成功！獲得 {len(df)} 筆 5分鐘線資料")
        print("\n前 5 筆資料:")
        print(df.head().to_string())
        
        # 存入資料庫
        store_minute_bars(df)
        print("✅ 資料已存入 minute_bars 資料表")
        
    else:
        print("❌ 下載失敗或無資料")

def example_download_multiple_stocks():
    """範例：下載多檔股票的分鐘線資料"""
    print("\n=== 範例：下載多檔股票分鐘線資料 ===")
    
    # 設定參數
    symbols = ['2330', '2317', '2603']  # 台積電、鴻海、長榮
    target_date = date(2024, 1, 2)
    
    total_rows = 0
    
    for symbol in symbols:
        print(f"\n處理 {symbol}...")
        
        df = fetch_symbol_date(symbol, target_date)
        
        if not df.empty:
            store_minute_bars(df)
            total_rows += len(df)
            print(f"✅ {symbol}: {len(df)} 筆資料")
        else:
            print(f"⚠️  {symbol}: 無資料")
    
    print(f"\n總計下載 {total_rows} 筆資料")

def example_historical_backfill():
    """範例：歷史資料回填"""
    print("\n=== 範例：歷史資料回填 ===")
    
    symbol = '2330'
    start_date = date(2024, 1, 1)
    end_date = date(2024, 1, 5)  # 回填 5 天
    
    current_date = start_date
    total_rows = 0
    
    while current_date <= end_date:
        print(f"處理 {symbol} {current_date}...")
        
        df = fetch_symbol_date(symbol, current_date)
        
        if not df.empty:
            store_minute_bars(df)
            total_rows += len(df)
            print(f"  ✅ {len(df)} 筆資料")
        else:
            print(f"  ⚠️  無資料")
        
        current_date += timedelta(days=1)
    
    print(f"\n歷史回填完成，總計 {total_rows} 筆資料")

def example_different_data_sources():
    """範例：測試不同資料源"""
    print("\n=== 範例：測試不同資料源 ===")
    
    symbol = '2330'
    
    # 測試代理資料（早期日期）
    print("1. 測試代理資料源...")
    proxy_date = date(2018, 1, 2)
    df_proxy = fetch_symbol_date(symbol, proxy_date)
    print(f"   代理資料: {len(df_proxy)} 筆")
    
    # 測試 FinMind 資料源
    print("2. 測試 FinMind 資料源...")
    finmind_date = date(2019, 6, 1)
    df_finmind = fetch_symbol_date(symbol, finmind_date)
    print(f"   FinMind 資料: {len(df_finmind)} 筆")
    
    # 測試 Shioaji 資料源
    print("3. 測試 Shioaji 資料源...")
    shioaji_date = date(2024, 1, 2)
    df_shioaji = fetch_symbol_date(symbol, shioaji_date)
    print(f"   Shioaji 資料: {len(df_shioaji)} 筆")

def setup_database():
    """設置資料庫"""
    print("=== 設置資料庫 ===")
    
    # 顯示資料庫資訊
    db_info = get_database_info()
    print(f"資料庫類型: {db_info['type']}")
    print(f"資料庫路徑: {db_info['path']}")
    
    # 建立 minute_bars 資料表
    create_minute_bars_table()
    print("✅ minute_bars 資料表已準備就緒")

def main():
    """主函數"""
    print("fetch_minute.py 使用範例")
    print("="*50)
    
    # 設置資料庫
    setup_database()
    
    # 執行範例
    try:
        example_download_single_stock()
        example_download_multiple_stocks()
        example_historical_backfill()
        example_different_data_sources()
        
        print("\n" + "="*50)
        print("✅ 所有範例執行完成！")
        
    except Exception as e:
        print(f"\n❌ 範例執行失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()