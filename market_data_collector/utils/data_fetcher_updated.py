# utils/data_fetcher_updated.py
"""
更新版的 data_fetcher - 使用新的資料庫抽象層
這是原始 data_fetcher.py 的修改版本，展示如何遷移到新的 DB 抽象層
"""
from __future__ import annotations
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import holidays

from utils.config import *
from utils.db import get_conn, insert_df, query_df, execute_sql, create_minute_bars_table


def get_last_trading_day(date):
    # 使用 holidays 套件定義台灣假日
    tw_holidays = holidays.Taiwan()
    # 檢查是否為週末（週六或週日）或假日，若是則往前推一天
    while date.weekday() >= 5 or date in tw_holidays:
        date -= timedelta(days=1)
    return date

def get_expected_latest_date():
    """
    根據台灣時區，若當前時間 >= 22:00，則最新資料日期為今日，
    否則為昨日；若該日期為休盤日（週末或假日），則返回最近一個交易日。
    """
    tz = pytz.timezone("Asia/Taipei")
    now = datetime.now(tz)
    # 根據時間決定初步日期
    if now.hour >= 22:
        latest_date = now.date()
    else:
        latest_date = (now - timedelta(days=1)).date()
    # 如果初步日期為非交易日，則往前推到最近一個交易日
    return get_last_trading_day(latest_date)


def create_db_and_table():
    """建立資料庫與各資料表 - 使用新的 DB 抽象層"""
    
    # 使用新的資料庫抽象層
    conn = get_conn()
    cursor = conn.cursor()
    
    try:
        # candlesticks_daily table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candlesticks_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market TEXT NOT NULL,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                UNIQUE(market, symbol, date)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_market_symbol_date 
            ON candlesticks_daily (market, symbol, date)
        """)
        
        # candlesticks_min table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candlesticks_min (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                interval TEXT,
                UNIQUE(market, symbol, timestamp)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_min_market_symbol_timestamp 
            ON candlesticks_min (market, symbol, timestamp)
        """)
        
        # financials table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                cost_of_goods_sold REAL,
                eps REAL,
                pe_ratio REAL,
                equity_attributable_to_owners REAL,
                gross_profit REAL,
                income_after_taxes REAL,
                income_from_continuing_operations REAL,
                noncontrolling_interests REAL,
                operating_expenses REAL,
                operating_income REAL,
                other_comprehensive_income REAL,
                pre_tax_income REAL,
                realized_gain REAL,
                revenue REAL,
                tax REAL,
                total_profit REAL,
                nonoperating_income_expense REAL,
                UNIQUE(symbol, date)
            )
        """)
        
        # technical_indicators table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market TEXT NOT NULL,
                symbol TEXT NOT NULL,
                date   TEXT NOT NULL,

                -- ── 新指標 ───────────────────────
                sma_5  REAL,
                sma_20 REAL,
                sma_60 REAL,

                ema_12 REAL,
                ema_26 REAL,
                ema_50 REAL,

                macd        REAL,
                macd_signal REAL,
                macd_hist   REAL,

                keltner_upper  REAL,
                keltner_middle REAL,
                keltner_lower  REAL,

                bollinger_upper  REAL,
                bollinger_middle REAL,
                bollinger_lower  REAL,

                pct_b     REAL,
                bandwidth REAL,

                UNIQUE (market, symbol, date)
            );
        """)

        
        # monthly_revenue table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS monthly_revenue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                monthly_revenue REAL,
                UNIQUE(symbol, date)
            )
        """)
        
        # margin_purchase_shortsale table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS margin_purchase_shortsale (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                MarginPurchaseBuy REAL,
                MarginPurchaseCashRepayment REAL,
                MarginPurchaseLimit REAL,
                MarginPurchaseSell REAL,
                MarginPurchaseTodayBalance REAL,
                MarginPurchaseYesterdayBalance REAL,
                OffsetLoanAndShort REAL,
                ShortSaleBuy REAL,
                ShortSaleCashRepayment REAL,
                ShortSaleLimit REAL,
                ShortSaleSell REAL,
                ShortSaleTodayBalance REAL,
                ShortSaleYesterdayBalance REAL,
                UNIQUE(symbol, date)
            )
        """)
        
        # institutional_investors_buy_sell table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS institutional_investors_buy_sell (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                Dealer_Hedging_buy REAL,
                Dealer_self_buy REAL,
                Foreign_Dealer_Self_buy REAL,
                Foreign_Investor_buy REAL,
                Investment_Trust_buy REAL,
                Dealer_Hedging_sell REAL,
                Dealer_self_sell REAL,
                Foreign_Dealer_Self_sell REAL,
                Foreign_Investor_sell REAL,
                Investment_Trust_sell REAL,
                UNIQUE(symbol, date)
            )
        """)
        
        # financial_per table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_per (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                dividend_yield REAL,
                PER REAL,
                PBR REAL,
                UNIQUE(symbol, date)
            )
        """)
        
        conn.commit()
        print("[data_fetcher_updated] 所有資料表已建立完成")
        
        # 建立新的 minute_bars 資料表
        create_minute_bars_table()
        
    finally:
        conn.close()


# --------------------
# FinMind 日線資料抓取與存入 - 使用新的 DB 抽象層
def fetch_stock_data(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    
    print(f"[data_fetcher_updated] 正在從 FinMind 下載 {stock_id} 的日線資料...")
    
    params = {
        "dataset": DATASET,
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": TOKEN
    }
    response = requests.get(API_ENDPOINT, params=params)
    if response.status_code == 200:
        json_data = response.json()
        if json_data.get("status") == 200 and "data" in json_data:
            df = pd.DataFrame(json_data["data"])
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.sort_values("date", inplace=True)
                df.reset_index(drop=True, inplace=True)
                return df
            else:
                raise ValueError(f"[data_fetcher_updated] FinMind 回傳資料為空：{stock_id}")
        else:
            raise ValueError(f"[data_fetcher_updated] FinMind 回傳錯誤: {json_data}")
    else:
        raise Exception(f"[data_fetcher_updated] HTTP Error: {response.status_code} {response.text}")

def store_stock_data_to_db_new(df, stock_id):
    """使用新的 DB 抽象層存儲股票資料"""
    if df.empty:
        print(f"[data_fetcher_updated] DataFrame 為空，跳過存儲 {stock_id}")
        return
    
    # 準備資料
    market = "TW"
    processed_data = []
    
    for _, row in df.iterrows():
        processed_data.append({
            'market': market,
            'symbol': stock_id,
            'date': row["date"].strftime("%Y-%m-%d"),
            'open': row.get("open", None),
            'high': row.get("max", None),  # FinMind 使用 'max' 作為最高價
            'low': row.get("min", None),   # FinMind 使用 'min' 作為最低價
            'close': row.get("close", None),
            'volume': row.get("Trading_Volume", None)
        })
    
    # 轉換為 DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    # 使用新的 insert_df 函數
    try:
        # 先刪除重複資料（如果存在）
        conn = get_conn()
        cursor = conn.cursor()
        
        for _, row in processed_df.iterrows():
            cursor.execute("""
                DELETE FROM candlesticks_daily 
                WHERE market = ? AND symbol = ? AND date = ?
            """, (row['market'], row['symbol'], row['date']))
        
        conn.commit()
        conn.close()
        
        # 插入新資料
        insert_df('candlesticks_daily', processed_df, if_exists='append')
        print(f"[data_fetcher_updated] {stock_id} 的日線資料已存入資料庫（{len(processed_df)} 筆）")
        
    except Exception as e:
        print(f"[data_fetcher_updated] 存儲失敗 {stock_id}: {e}")
        raise

def load_stock_data_from_db_new(stock_id, start_date=None, end_date=None):
    """使用新的 DB 抽象層載入股票資料"""
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    
    query = """
    SELECT * FROM candlesticks_daily
    WHERE symbol = ?
      AND date BETWEEN ? AND ?
    ORDER BY date ASC
    """
    
    df = query_df(query, (stock_id, start_date, end_date))
    
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    
    return df


def demo_new_db_functions():
    """演示新的資料庫功能"""
    print("=== 演示新的資料庫抽象層功能 ===")
    
    # 1. 建立資料表
    print("1. 建立資料表...")
    create_db_and_table()
    
    # 2. 測試資料下載和存儲（使用現有的 STOCK_IDS 中的第一個）
    if STOCK_IDS:
        test_stock = STOCK_IDS[0]
        print(f"2. 測試下載 {test_stock} 的資料...")
        
        try:
            # 下載最近 5 天的資料
            end_date = datetime.today().strftime("%Y-%m-%d")
            start_date = (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d")
            
            df = fetch_stock_data(test_stock, start_date=start_date, end_date=end_date)
            print(f"   下載了 {len(df)} 筆資料")
            
            # 存儲資料
            store_stock_data_to_db_new(df, test_stock)
            
            # 讀取資料
            loaded_df = load_stock_data_from_db_new(test_stock, start_date=start_date, end_date=end_date)
            print(f"   從資料庫讀取了 {len(loaded_df)} 筆資料")
            
            if not loaded_df.empty:
                print("   最新一筆資料:")
                print(loaded_df.tail(1).to_string())
            
        except Exception as e:
            print(f"   測試失敗: {e}")
    
    print("=== 演示完成 ===")


if __name__ == "__main__":
    demo_new_db_functions()