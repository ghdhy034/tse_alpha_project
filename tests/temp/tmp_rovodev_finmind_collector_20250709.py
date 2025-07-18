#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinMind 資料收集器 - 專門收集日線、財報、融資融券、法人進出等資料
"""
import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple
import pandas as pd
import requests

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "market_data_collector"))

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinMindAPIManager:
    """FinMind API Key 管理器 - 支援多Key輪換"""
    
    def __init__(self, api_keys_file: str = "findmind_api_keys.txt"):
        self.api_keys = []
        self.current_key_index = 0
        self.key_usage_count = {}
        self.key_daily_limit = 1000  # 每個Key每日限制
        self.load_api_keys(api_keys_file)
    
    def load_api_keys(self, api_keys_file: str):
        """從檔案載入API Keys"""
        try:
            with open(api_keys_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析API Keys
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('finmind api key'):
                    # 這是一個API Key
                    if line.startswith('eyJ'):  # JWT token 格式
                        self.api_keys.append(line)
                        self.key_usage_count[line] = 0
            
            logger.info(f"載入 {len(self.api_keys)} 個 FinMind API Keys")
            
        except Exception as e:
            logger.error(f"載入API Keys失敗: {e}")
            # 使用預設Key
            from market_data_collector.utils.config import TOKEN
            self.api_keys = [TOKEN]
            self.key_usage_count[TOKEN] = 0
    
    def get_current_key(self) -> str:
        """獲取當前可用的API Key"""
        if not self.api_keys:
            raise ValueError("沒有可用的API Keys")
        
        # 檢查當前Key是否超過限制
        current_key = self.api_keys[self.current_key_index]
        if self.key_usage_count[current_key] >= self.key_daily_limit:
            # 切換到下一個Key
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            current_key = self.api_keys[self.current_key_index]
            
            # 如果所有Key都用完了
            if self.key_usage_count[current_key] >= self.key_daily_limit:
                logger.warning("所有API Keys都已達到每日限制")
                # 重置計數器（新的一天）
                for key in self.key_usage_count:
                    self.key_usage_count[key] = 0
        
        return current_key
    
    def record_usage(self, api_key: str):
        """記錄API Key使用次數"""
        if api_key in self.key_usage_count:
            self.key_usage_count[api_key] += 1
    
    def get_usage_status(self) -> Dict:
        """獲取使用狀況"""
        return {
            'current_key_index': self.current_key_index,
            'total_keys': len(self.api_keys),
            'usage_count': dict(self.key_usage_count)
        }


class FinMindDataCollector:
    """FinMind 資料收集器"""
    
    def __init__(self):
        self.api_manager = FinMindAPIManager()
        self.rate_limit_delay = 0.5  # 請求間隔（秒）
        
        # 匯入資料庫模組
        try:
            from market_data_collector.utils.db import insert_df, query_df
            from market_data_collector.utils.config import API_ENDPOINT
            self.insert_df = insert_df
            self.query_df = query_df
            self.api_endpoint = API_ENDPOINT
        except ImportError as e:
            logger.error(f"無法匯入資料庫模組: {e}")
            raise
    
    def get_stock_list(self) -> List[str]:
        """獲取股票清單"""
        # 使用完整的180支股票清單
        group_A = [
            "2330","2317","2454","2303","2408","2412","2382","2357","2379","3034",
            "3008","4938","2449","2383","2356","3006","3661","2324","8046","3017",
            "6121","3037","3014","3035","3062","3030","3529","5443","2337","8150",
            "3293","3596","2344","2428","2345","2338","6202","5347","3673","3105",
            "6231","6669","4961","4967","6668","4960","3528","6147","3526","6547",
            "8047","3227","4968","5274","6415","6414","6770","2331","6290","2342"
        ]
        
        group_B = [
            "2603","2609","2615","2610","2618","2637","2606","2002","2014","2027",
            "2201","1201","1216","1301","1303","1326","1710","1717","1722","1723",
            "1402","1409","1434","1476","2006","2049","2105","2106","2107","1605",
            "1609","1608","1612","2308","1727","1730","1101","1102","1108","1210",
            "1215","1802","1806","1810","1104","1313","1314","1310","5608","5607",
            "8105","8940","5534","5609","5603","2023","2028","2114","9933","2501"
        ]
        
        group_C = [
            "2880","2881","2882","2883","2884","2885","2886","2887","2888","2890",
            "2891","2892","2812","2823","2834","2850","2801","2836","2845","4807",
            "3702","3706","4560","8478","4142","4133","6525","6548","6843","1513",
            "1514","1516","1521","1522","1524","1533","1708","3019","5904","5906",
            "5902","6505","6806","6510","2207","2204","2231","1736","4105","4108",
            "4162","1909","1702","9917","1217","1218","1737","1783","3708","1795"
        ]
        
        all_stocks = group_A + group_B + group_C
        all_stocks = list(set(all_stocks))  # 去重
        all_stocks.sort()
        
        logger.info(f"準備收集 {len(all_stocks)} 支股票的 FinMind 資料")
        return all_stocks
    
    def fetch_finmind_data(self, dataset: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """從FinMind API獲取資料"""
        try:
            api_key = self.api_manager.get_current_key()
            
            params = {
                "dataset": dataset,
                "data_id": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "token": api_key
            }
            
            logger.info(f"📡 請求 {dataset}: {symbol} ({start_date} ~ {end_date})")
            
            # 速率限制
            time.sleep(self.rate_limit_delay)
            
            response = requests.get(self.api_endpoint, params=params, timeout=30)
            
            if response.status_code == 200:
                json_data = response.json()
                
                if json_data.get("status") == 200 and "data" in json_data:
                    data = json_data["data"]
                    
                    if data:
                        df = pd.DataFrame(data)
                        self.api_manager.record_usage(api_key)
                        logger.info(f"✅ 成功獲取 {len(df)} 筆 {dataset}: {symbol}")
                        return df
                    else:
                        logger.warning(f"⚠️  {symbol} 無 {dataset} 資料")
                        return pd.DataFrame()
                else:
                    logger.warning(f"❌ API 回傳錯誤: {json_data}")
                    return pd.DataFrame()
            else:
                logger.error(f"❌ HTTP 錯誤 {response.status_code}: {response.text}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ 獲取 {dataset} 資料失敗 {symbol}: {e}")
            return pd.DataFrame()
    
    def collect_stock_data(self, symbol: str, start_date: str = "2020-03-02", end_date: str = "2025-07-08"):
        """收集單一股票的所有 FinMind 資料 - 使用原始 data_fetcher 處理方式"""
        logger.info(f"🎯 開始收集 {symbol} 的 FinMind 資料...")
        
        # 匯入原始處理函數
        try:
            from market_data_collector.utils.data_fetcher import (
                create_db_and_table,
                fetch_stock_data, store_stock_data_to_db,
                fetch_financial_data, store_financial_data_to_db,
                fetch_monthly_revenue, store_monthly_revenue_to_db,
                fetch_margin_purchase_shortsale, store_margin_purchase_shortsale_to_db,
                fetch_investors_buy_sell, store_investors_buy_sell_to_db,
                fetch_per_data, store_per_data_to_db,
                compute_technical_indicators, store_technical_indicators_to_db
            )
            
            # 確保資料表已建立
            create_db_and_table()
            logger.info("✅ 資料表檢查/建立完成")
            
        except ImportError as e:
            logger.error(f"❌ 無法匯入原始處理函數: {e}")
            return {}
        
        results = {}
        
        # 1. 日線資料
        try:
            logger.info(f"📈 收集 {symbol} 日線資料...")
            df_daily = fetch_stock_data(symbol, start_date, end_date)
            if not df_daily.empty:
                store_stock_data_to_db(df_daily, symbol)
                results['daily_price'] = len(df_daily)
                logger.info(f"✅ {symbol} 日線資料: {len(df_daily)} 筆")
                
                # 計算技術指標
                try:
                    tech_indicators = compute_technical_indicators(symbol, df_daily)
                    store_technical_indicators_to_db(tech_indicators, symbol)
                    results['technical_indicators'] = len(tech_indicators)
                    logger.info(f"✅ {symbol} 技術指標: {len(tech_indicators)} 筆")
                except Exception as e:
                    logger.warning(f"⚠️  {symbol} 技術指標計算失敗: {e}")
                    results['technical_indicators'] = 0
            else:
                results['daily_price'] = 0
        except Exception as e:
            logger.error(f"❌ {symbol} 日線資料失敗: {e}")
            results['daily_price'] = -1
        
        # 2. 財報資料
        try:
            logger.info(f"📊 收集 {symbol} 財報資料...")
            df_financial = fetch_financial_data(symbol, start_date, end_date)
            if not df_financial.empty:
                store_financial_data_to_db(df_financial, symbol)
                results['financial'] = len(df_financial)
                logger.info(f"✅ {symbol} 財報資料: {len(df_financial)} 筆")
            else:
                results['financial'] = 0
        except Exception as e:
            logger.error(f"❌ {symbol} 財報資料失敗: {e}")
            results['financial'] = -1
        
        # 3. 月營收資料
        try:
            logger.info(f"💰 收集 {symbol} 月營收資料...")
            df_revenue = fetch_monthly_revenue(symbol, start_date, end_date)
            if not df_revenue.empty:
                store_monthly_revenue_to_db(df_revenue, symbol)
                results['monthly_revenue'] = len(df_revenue)
                logger.info(f"✅ {symbol} 月營收資料: {len(df_revenue)} 筆")
            else:
                results['monthly_revenue'] = 0
        except Exception as e:
            logger.error(f"❌ {symbol} 月營收資料失敗: {e}")
            results['monthly_revenue'] = -1
        
        # 4. 融資融券資料
        try:
            logger.info(f"💳 收集 {symbol} 融資融券資料...")
            df_margin = fetch_margin_purchase_shortsale(symbol, start_date, end_date)
            if not df_margin.empty:
                store_margin_purchase_shortsale_to_db(df_margin, symbol)
                results['margin_shortsale'] = len(df_margin)
                logger.info(f"✅ {symbol} 融資融券資料: {len(df_margin)} 筆")
            else:
                results['margin_shortsale'] = 0
        except Exception as e:
            logger.error(f"❌ {symbol} 融資融券資料失敗: {e}")
            results['margin_shortsale'] = -1
        
        # 5. 機構投信買賣資料
        try:
            logger.info(f"🏛️ 收集 {symbol} 機構投信買賣資料...")
            df_institutional = fetch_investors_buy_sell(symbol, start_date, end_date)
            if not df_institutional.empty:
                store_investors_buy_sell_to_db(df_institutional, symbol)
                results['institutional'] = len(df_institutional)
                logger.info(f"✅ {symbol} 機構投信買賣資料: {len(df_institutional)} 筆")
            else:
                results['institutional'] = 0
        except Exception as e:
            logger.error(f"❌ {symbol} 機構投信買賣資料失敗: {e}")
            results['institutional'] = -1
        
        # 6. 本益比資料
        try:
            logger.info(f"📋 收集 {symbol} 本益比資料...")
            df_per = fetch_per_data(symbol, start_date, end_date)
            if not df_per.empty:
                store_per_data_to_db(df_per, symbol)
                results['per_data'] = len(df_per)
                logger.info(f"✅ {symbol} 本益比資料: {len(df_per)} 筆")
            else:
                results['per_data'] = 0
        except Exception as e:
            logger.error(f"❌ {symbol} 本益比資料失敗: {e}")
            results['per_data'] = -1
        
        return results
    
    def get_table_name(self, data_type: str) -> str:
        """獲取資料表名稱"""
        table_mapping = {
            "daily_price": "candlesticks_daily",
            "margin_shortsale": "margin_purchase_shortsale",
            "institutional": "institutional_investors_buy_sell",
            "financial": "financial_statements",
            "balance_sheet": "balance_sheet",
            "monthly_revenue": "monthly_revenue",
            "per_data": "financial_per"
        }
        return table_mapping.get(data_type, data_type)
    
    def collect_all_data(self, symbols: Optional[List[str]] = None, 
                        start_date: str = "2020-03-02", 
                        end_date: str = "2025-07-08",
                        test_mode: bool = True):
        """收集所有股票的 FinMind 資料"""
        if symbols is None:
            symbols = self.get_stock_list()
        
        if test_mode:
            # 測試模式：只收集前3支股票
            symbols = symbols[:3]
            logger.info(f"🧪 測試模式：收集 {symbols} 的資料")
        
        logger.info(f"📊 開始收集 {len(symbols)} 支股票的 FinMind 資料...")
        logger.info(f"📅 日期範圍: {start_date} ~ {end_date}")
        
        total_results = {}
        
        try:
            for i, symbol in enumerate(symbols):
                logger.info(f"📈 處理進度: {i+1}/{len(symbols)} - {symbol}")
                
                # 收集該股票的所有資料
                results = self.collect_stock_data(symbol, start_date, end_date)
                total_results[symbol] = results
                
                # 每5支股票顯示API使用狀況
                if (i + 1) % 5 == 0:
                    usage = self.api_manager.get_usage_status()
                    logger.info(f"📊 API 使用狀況: Key {usage['current_key_index']+1}/{usage['total_keys']}")
                
                # 短暫休息
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("⏹️  收到中斷信號，正在停止...")
            raise
        
        except Exception as e:
            logger.error(f"❌ 資料收集過程發生錯誤: {e}")
            raise
        
        finally:
            # 顯示收集結果統計
            self.show_collection_summary(total_results)
    
    def show_collection_summary(self, results: Dict):
        """顯示收集結果統計"""
        logger.info("=" * 50)
        logger.info("📊 FinMind 資料收集完成統計")
        logger.info("=" * 50)
        
        data_types = ["daily_price", "technical_indicators", "financial", "monthly_revenue", "margin_shortsale", "institutional", "per_data"]
        
        for data_type in data_types:
            total_records = 0
            success_count = 0
            
            for symbol, symbol_results in results.items():
                if data_type in symbol_results:
                    if symbol_results[data_type] > 0:
                        total_records += symbol_results[data_type]
                        success_count += 1
            
            logger.info(f"{data_type:20}: {success_count:3} 支股票, {total_records:6} 筆資料")
        
        logger.info("=" * 50)


def main():
    """主函數"""
    print("=" * 50)
    print("🏦 FinMind 歷史資料收集器")
    print("=" * 50)
    print("收集內容：")
    print("• 日線價格資料 (TaiwanStockPrice)")
    print("• 融資融券資料 (TaiwanStockMarginPurchaseShortSale)")
    print("• 法人進出資料 (TaiwanStockInstitutionalInvestorsBuySell)")
    print("• 財務報表資料 (TaiwanStockFinancialStatements)")
    print("• 資產負債表 (TaiwanStockBalanceSheet)")
    print("• 月營收資料 (TaiwanStockMonthRevenue)")
    print("• 本益比資料 (TaiwanStockPER)")
    print("=" * 50)
    
    try:
        collector = FinMindDataCollector()
        
        # 詢問用戶是否要測試模式
        choice = input("選擇模式 (1=測試模式收集3支股票, 2=完整模式收集180支股票): ").strip()
        
        if choice == "1":
            print("🧪 測試模式：收集前3支股票")
            collector.collect_all_data(test_mode=True)
        elif choice == "2":
            print("🚀 完整模式：收集全部180支股票")
            collector.collect_all_data(test_mode=False)
        else:
            print("🧪 預設測試模式：收集前3支股票")
            collector.collect_all_data(test_mode=True)
        
        print("✅ FinMind 資料收集完成！")
        
    except Exception as e:
        logger.error(f"❌ 執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()