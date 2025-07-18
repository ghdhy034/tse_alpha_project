#!/usr/bin/env python3
"""
測試 Shioaji 分鐘線資料下載功能
驗證能否正常下載並存入 minute_bars 資料表
"""

import sys
import os
from pathlib import Path
from datetime import datetime, date, timedelta

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent / "data_pipeline"))
sys.path.insert(0, str(Path(__file__).parent / "market_data_collector"))

def test_minute_data_download():
    """測試分鐘線資料下載"""
    print("🔧 測試 Shioaji 分鐘線資料下載")
    print("=" * 50)
    
    # 測試股票代號 (選擇活躍的大型股)
    test_symbols = ["2330", "2317", "2454"]  # 台積電、鴻海、聯發科
    
    # 測試日期 (選擇最近的交易日)
    test_date = date(2024, 12, 16)  # 可以調整為最近的交易日
    
    print(f"📊 測試股票: {test_symbols}")
    print(f"📅 測試日期: {test_date}")
    
    try:
        # 導入分鐘線下載模組
        from fetch_minute import fetch_symbol_date, store_minute_bars
        print("✅ 分鐘線下載模組導入成功")
    except ImportError as e:
        print(f"❌ 分鐘線下載模組導入失敗: {e}")
        return False
    
    try:
        # 導入資料庫模組
        from market_data_collector.utils.db import query_df
        print("✅ 資料庫模組導入成功")
    except ImportError as e:
        print(f"❌ 資料庫模組導入失敗: {e}")
        return False
    
    # 檢查 minute_bars 資料表
    print("\n--- 檢查 minute_bars 資料表 ---")
    try:
        count_before = query_df("SELECT COUNT(*) as count FROM minute_bars")
        print(f"✅ minute_bars 資料表存在，當前資料筆數: {count_before.iloc[0]['count']}")
    except Exception as e:
        print(f"❌ minute_bars 資料表檢查失敗: {e}")
        return False
    
    # 開始測試下載
    print(f"\n--- 開始測試下載 {test_date} 的分鐘線資料 ---")
    
    total_downloaded = 0
    successful_symbols = []
    failed_symbols = []
    
    for symbol in test_symbols:
        print(f"\n🔄 測試股票: {symbol}")
        
        try:
            # 下載資料
            df = fetch_symbol_date(symbol, test_date)
            
            if not df.empty:
                print(f"✅ {symbol} 下載成功: {len(df)} 筆資料")
                print(f"   時間範圍: {df['ts'].min()} ~ {df['ts'].max()}")
                print(f"   價格範圍: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
                print(f"   成交量總計: {df['volume'].sum():,}")
                
                # 存入資料庫
                store_minute_bars(df)
                print(f"✅ {symbol} 資料已存入 minute_bars")
                
                total_downloaded += len(df)
                successful_symbols.append(symbol)
                
                # 顯示前幾筆資料樣本
                print(f"   資料樣本 (前3筆):")
                for i, row in df.head(3).iterrows():
                    print(f"     {row['ts']} | O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:,}")
                
            else:
                print(f"⚠️  {symbol} 無資料 (可能非交易日或資料源問題)")
                failed_symbols.append(symbol)
                
        except Exception as e:
            print(f"❌ {symbol} 下載失敗: {e}")
            failed_symbols.append(symbol)
    
    # 驗證資料庫存儲
    print(f"\n--- 驗證資料庫存儲 ---")
    try:
        count_after = query_df("SELECT COUNT(*) as count FROM minute_bars")
        new_records = count_after.iloc[0]['count'] - count_before.iloc[0]['count']
        print(f"✅ 新增資料筆數: {new_records}")
        
        # 檢查各股票的資料
        for symbol in successful_symbols:
            symbol_count = query_df(
                "SELECT COUNT(*) as count FROM minute_bars WHERE symbol = ? AND DATE(ts) = ?", 
                (symbol, test_date.strftime('%Y-%m-%d'))
            )
            print(f"   {symbol}: {symbol_count.iloc[0]['count']} 筆")
            
    except Exception as e:
        print(f"❌ 資料庫驗證失敗: {e}")
    
    # 測試總結
    print(f"\n" + "=" * 50)
    print(f"📊 測試結果總結")
    print(f"=" * 50)
    print(f"✅ 成功股票: {successful_symbols} ({len(successful_symbols)}/{len(test_symbols)})")
    if failed_symbols:
        print(f"❌ 失敗股票: {failed_symbols}")
    print(f"📈 總下載筆數: {total_downloaded}")
    
    # 成功率評估
    success_rate = len(successful_symbols) / len(test_symbols) * 100
    if success_rate >= 100:
        print(f"🎉 測試完全成功！成功率: {success_rate:.0f}%")
        return True
    elif success_rate >= 50:
        print(f"⚠️  測試部分成功，成功率: {success_rate:.0f}%")
        return True
    else:
        print(f"💥 測試失敗，成功率: {success_rate:.0f}%")
        return False

def test_data_quality():
    """測試下載資料的品質"""
    print(f"\n--- 資料品質檢查 ---")
    
    try:
        from market_data_collector.utils.db import query_df
        
        # 檢查最新資料
        latest_data = query_df("""
            SELECT symbol, COUNT(*) as count, 
                   MIN(ts) as start_time, MAX(ts) as end_time,
                   AVG(volume) as avg_volume
            FROM minute_bars 
            WHERE DATE(ts) = (SELECT MAX(DATE(ts)) FROM minute_bars)
            GROUP BY symbol
            ORDER BY symbol
        """)
        
        if not latest_data.empty:
            print("📊 最新資料統計:")
            for _, row in latest_data.iterrows():
                print(f"   {row['symbol']}: {row['count']} 筆, "
                      f"{row['start_time']} ~ {row['end_time']}, "
                      f"平均成交量: {row['avg_volume']:,.0f}")
        
        # 檢查資料完整性
        integrity_check = query_df("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(DISTINCT DATE(ts)) as unique_dates,
                SUM(CASE WHEN volume = 0 THEN 1 ELSE 0 END) as zero_volume_count
            FROM minute_bars
        """)
        
        if not integrity_check.empty:
            row = integrity_check.iloc[0]
            print(f"\n📈 資料完整性:")
            print(f"   總記錄數: {row['total_records']:,}")
            print(f"   股票數量: {row['unique_symbols']}")
            print(f"   日期數量: {row['unique_dates']}")
            print(f"   零成交量記錄: {row['zero_volume_count']}")
        
    except Exception as e:
        print(f"❌ 資料品質檢查失敗: {e}")

def main():
    """主函數"""
    print("🚀 開始 Shioaji 分鐘線資料下載測試")
    
    # 執行下載測試
    success = test_minute_data_download()
    
    # 執行品質檢查
    test_data_quality()
    
    print(f"\n🏁 測試完成")
    if success:
        print("🎉 Shioaji 分鐘線下載功能正常！")
        print("💡 建議: 可以開始定期下載分鐘線資料了")
    else:
        print("💥 發現問題，需要進一步調試")
    
    return success

if __name__ == "__main__":
    main()