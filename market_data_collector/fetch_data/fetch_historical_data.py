# fetch_data/fetch_historical_data.py
import sys, os
from datetime import timedelta,datetime
# 將專案根目錄加入 PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_fetcher import *
from utils.config import *

def main():
    # 建立資料庫與資料表
    create_db_and_table()
    
    expected_latest_date = get_expected_latest_date()  # 依據台灣時間22:00判定最新資料日期
    print(f"[main] 預期最新資料日期：{expected_latest_date}")

    for stock_id in STOCK_IDS:
        print(f"=== 處理股票代號: {stock_id} ===")
        
        # 處理日線資料：檢查DB是否已有資料且為最新資料
        df_db = load_stock_data_from_db(stock_id, start_date=START_DATE)
        if df_db.empty:
            print(f"[main] DB 中無 {stock_id} 的日線資料，開始下載全部資料...")
            df_daily = fetch_stock_data(stock_id, start_date=START_DATE)
            store_stock_data_to_db(df_daily, stock_id)
        else:
            latest_date_in_db = df_db["date"].max().date()
            if latest_date_in_db < expected_latest_date:
                new_start_date = (latest_date_in_db + timedelta(days=1)).strftime("%Y-%m-%d")
                new_end_date = expected_latest_date.strftime("%Y-%m-%d")
                print(f"[main] {stock_id} 最新DB日期 {latest_date_in_db}，更新至 {new_end_date}...")
                df_new = fetch_stock_data(stock_id, start_date=new_start_date, end_date=new_end_date)
                if not df_new.empty:
                    store_stock_data_to_db(df_new, stock_id)
            else:
                print(f"[main] {stock_id} 的日線資料已更新至 {latest_date_in_db}。")
        
        # 處理財報資料
        df_fin_db = load_financial_data_from_db(stock_id, start_date=START_DATE)
        if df_fin_db.empty:
            print(f"[main] DB 中無 {stock_id} 的財報資料，開始下載...")
            df_fin = fetch_financial_data(stock_id, start_date=START_DATE)
            store_financial_data_to_db(df_fin, stock_id)
        else:
            print(f"[main] {stock_id} 的財報資料已存在於資料庫中，筆數：{len(df_fin_db)}")
        
        # 處理月營收資料
        df_monthly_db = load_monthly_revenue_from_db(stock_id, start_date=START_DATE)
        if df_monthly_db.empty:
            print(f"[main] DB 中無 {stock_id} 的月營收資料，開始下載...")
            df_monthly = fetch_monthly_revenue(stock_id, start_date=START_DATE)
            store_monthly_revenue_to_db(df_monthly, stock_id)
        else:
            print(f"[main] {stock_id} 的月營收資料已存在於資料庫中，筆數：{len(df_monthly_db)}")
        
        # 處理技術指標資料
        indicators_db = load_technical_indicators_from_db(stock_id)
        if indicators_db.empty:
            print(f"[main] DB 中無 {stock_id} 的技術指標資料，開始計算...")
            df_daily = load_stock_data_from_db(stock_id, start_date=START_DATE)
            indicators = compute_technical_indicators(stock_id, df_daily)
            store_technical_indicators_to_db(indicators, stock_id)
        else:
            print(f"[main] {stock_id} 的技術指標資料已存在於資料庫中，筆數：{len(indicators_db)}")
        
        # 處理融資融券資料
        margin_db = load_margin_purchase_shortsale_from_db(stock_id, start_date=START_DATE)
        if margin_db.empty:
            print(f"[main] DB 中無 {stock_id} 的融資融券資料，開始下載...")
            df_margin = fetch_margin_purchase_shortsale(stock_id, start_date=START_DATE)
            store_margin_purchase_shortsale_to_db(df_margin, stock_id)
        else:
            print(f"[main] {stock_id} 的融資融券資料已存在於資料庫中，筆數：{len(margin_db)}")
        
        # 處理機構投信買賣資料
        inv_db = load_investors_buy_sell_from_db(stock_id, start_date=START_DATE)
        if inv_db.empty:
            print(f"[main] DB 中無 {stock_id} 的機構投信買賣資料，開始下載...")
            df_inv = fetch_investors_buy_sell(stock_id, start_date=START_DATE)
            store_investors_buy_sell_to_db(df_inv, stock_id)
        else:
            print(f"[main] {stock_id} 的機構投信買賣資料已存在於資料庫中，筆數：{len(inv_db)}")
        
        # 處理 PER 資料
        per_db = load_per_data_from_db(stock_id, start_date=START_DATE)
        if per_db.empty:
            print(f"[main] DB 中無 {stock_id} 的本益比資料，開始下載...")
            df_per = fetch_per_data(stock_id, start_date=START_DATE)
            store_per_data_to_db(df_per, stock_id)
        else:
            print(f"[main] {stock_id} 的本益比資料已存在於資料庫中，筆數：{len(per_db)}")
        
if __name__ == "__main__":
    main()
