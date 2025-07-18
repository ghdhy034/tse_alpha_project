#!/usr/bin/env python3
"""
增強版資料收集器 - 支援多API KEY輪換和斷點續傳
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

class APIKeyManager:
    """API Key 管理器 - 支援多Key輪換"""
    
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


class ProgressTracker:
    """進度追蹤器 - 支援斷點續傳"""
    
    def __init__(self, progress_file: str = "data_collection_progress.json"):
        self.progress_file = progress_file
        self.progress_data = self.load_progress()
    
    def load_progress(self) -> Dict:
        """載入進度資料"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"載入進度檔案失敗: {e}")
        
        return {
            'last_update': None,
            'completed_symbols': [],
            'failed_symbols': [],
            'current_date_range': {},
            'statistics': {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'start_time': None,
                'last_save_time': None
            }
        }
    
    def save_progress(self):
        """儲存進度資料"""
        try:
            self.progress_data['last_update'] = datetime.now().isoformat()
            self.progress_data['statistics']['last_save_time'] = datetime.now().isoformat()
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"儲存進度檔案失敗: {e}")
    
    def mark_symbol_completed(self, symbol: str, data_type: str):
        """標記股票完成"""
        key = f"{symbol}_{data_type}"
        if key not in self.progress_data['completed_symbols']:
            self.progress_data['completed_symbols'].append(key)
        self.save_progress()
    
    def mark_symbol_failed(self, symbol: str, data_type: str, error: str):
        """標記股票失敗"""
        key = f"{symbol}_{data_type}"
        self.progress_data['failed_symbols'].append({
            'key': key,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
        self.save_progress()
    
    def is_symbol_completed(self, symbol: str, data_type: str) -> bool:
        """檢查股票是否已完成"""
        key = f"{symbol}_{data_type}"
        return key in self.progress_data['completed_symbols']
    
    def get_remaining_symbols(self, all_symbols: List[str], data_types: List[str]) -> List[Tuple[str, str]]:
        """獲取剩餘待處理的股票"""
        remaining = []
        for symbol in all_symbols:
            for data_type in data_types:
                if not self.is_symbol_completed(symbol, data_type):
                    remaining.append((symbol, data_type))
        return remaining
    
    def update_statistics(self, success: bool):
        """更新統計資料"""
        stats = self.progress_data['statistics']
        stats['total_requests'] += 1
        
        if success:
            stats['successful_requests'] += 1
        else:
            stats['failed_requests'] += 1
        
        if stats['start_time'] is None:
            stats['start_time'] = datetime.now().isoformat()


class EnhancedDataCollector:
    """增強版資料收集器"""
    
    def __init__(self):
        self.api_manager = APIKeyManager()
        self.progress_tracker = ProgressTracker()
        self.rate_limit_delay = 0.5  # 請求間隔（秒）
        
        # 匯入資料庫模組
        try:
            from market_data_collector.utils.db import insert_df, query_df
            from market_data_collector.utils.config import API_ENDPOINT, STOCK_IDS
            self.insert_df = insert_df
            self.query_df = query_df
            self.api_endpoint = API_ENDPOINT
            self.stock_ids = STOCK_IDS
        except ImportError as e:
            logger.error(f"無法匯入資料庫模組: {e}")
            raise
    
    def get_full_stock_list(self) -> List[str]:
        """獲取完整的180支股票清單（來自stock_id.txt三組別）"""
        # 使用stock_id.txt中的三組別股票
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
        
        # 合併所有組別
        all_stocks = group_A + group_B + group_C
        
        # 移除重複（如果有的話）
        all_stocks = list(set(all_stocks))
        all_stocks.sort()  # 排序
        
        logger.info(f"準備收集 {len(all_stocks)} 支股票的資料")
        logger.info(f"  組別A (半導體‧電子供應鏈): {len(group_A)} 支")
        logger.info(f"  組別B (傳產／原物料＆運輸): {len(group_B)} 支") 
        logger.info(f"  組別C (金融‧內需消費／綠能生技): {len(group_C)} 支")
        
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
            
            logger.info(f"請求 {dataset} 資料: {symbol} ({start_date} ~ {end_date})")
            
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
                        self.progress_tracker.update_statistics(True)
                        logger.info(f"成功獲取 {len(df)} 筆 {dataset} 資料: {symbol}")
                        return df
                    else:
                        logger.warning(f"{symbol} 無 {dataset} 資料")
                        self.progress_tracker.update_statistics(True)
                        return pd.DataFrame()
                else:
                    logger.warning(f"API 回傳錯誤: {json_data}")
                    self.progress_tracker.update_statistics(False)
                    return pd.DataFrame()
            else:
                logger.error(f"HTTP 錯誤 {response.status_code}: {response.text}")
                self.progress_tracker.update_statistics(False)
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"獲取 {dataset} 資料失敗 {symbol}: {e}")
            self.progress_tracker.update_statistics(False)
            return pd.DataFrame()
    
    def collect_stock_data(self, symbol: str, start_date: str = "2020-03-02", end_date: str = "2024-12-31"):
        """收集單一股票的所有資料"""
        logger.info(f"開始收集 {symbol} 的資料...")
        
        # 資料類型定義
        datasets = {
            "daily_price": "TaiwanStockPrice",
            "margin_shortsale": "TaiwanStockMarginPurchaseShortSale", 
            "institutional": "TaiwanStockInstitutionalInvestorsBuySell",
            "financial": "TaiwanStockFinancialStatements",
            "balance_sheet": "TaiwanStockBalanceSheet",
            "monthly_revenue": "TaiwanStockMonthRevenue"
        }
        
        for data_type, dataset in datasets.items():
            # 檢查是否已完成
            if self.progress_tracker.is_symbol_completed(symbol, data_type):
                logger.info(f"跳過已完成的 {symbol} {data_type}")
                continue
            
            try:
                # 獲取資料
                df = self.fetch_finmind_data(dataset, symbol, start_date, end_date)
                
                if not df.empty:
                    # 儲存到資料庫
                    table_name = self.get_table_name(data_type)
                    
                    # 添加symbol欄位（如果不存在）
                    if 'symbol' not in df.columns:
                        df['symbol'] = symbol
                    
                    # 儲存資料
                    self.insert_df(table_name, df, if_exists='append')
                    logger.info(f"成功儲存 {len(df)} 筆 {data_type} 資料: {symbol}")
                
                # 標記完成
                self.progress_tracker.mark_symbol_completed(symbol, data_type)
                
            except Exception as e:
                logger.error(f"處理 {symbol} {data_type} 失敗: {e}")
                self.progress_tracker.mark_symbol_failed(symbol, data_type, str(e))
                continue
    
    def get_table_name(self, data_type: str) -> str:
        """獲取資料表名稱"""
        table_mapping = {
            "daily_price": "candlesticks_daily",
            "margin_shortsale": "margin_purchase_shortsale",
            "institutional": "institutional_investors_buy_sell",
            "financial": "financial_statements",
            "balance_sheet": "balance_sheet",
            "monthly_revenue": "monthly_revenue"
        }
        return table_mapping.get(data_type, data_type)
    
    def collect_all_data(self, symbols: Optional[List[str]] = None, 
                        start_date: str = "2020-03-02", 
                        end_date: str = "2024-12-31"):
        """收集所有股票資料"""
        if symbols is None:
            symbols = self.get_full_stock_list()
        
        logger.info(f"開始收集 {len(symbols)} 支股票的資料...")
        logger.info(f"日期範圍: {start_date} ~ {end_date}")
        
        # 獲取剩餘待處理項目
        data_types = ["daily_price", "margin_shortsale", "institutional", "financial", "balance_sheet", "monthly_revenue"]
        remaining_tasks = self.progress_tracker.get_remaining_symbols(symbols, data_types)
        
        logger.info(f"剩餘待處理任務: {len(remaining_tasks)} 個")
        
        try:
            for i, (symbol, data_type) in enumerate(remaining_tasks):
                logger.info(f"處理進度: {i+1}/{len(remaining_tasks)} - {symbol} {data_type}")
                
                # 獲取資料
                dataset = {
                    "daily_price": "TaiwanStockPrice",
                    "margin_shortsale": "TaiwanStockMarginPurchaseShortSale",
                    "institutional": "TaiwanStockInstitutionalInvestorsBuySell",
                    "financial": "TaiwanStockFinancialStatements",
                    "balance_sheet": "TaiwanStockBalanceSheet",
                    "monthly_revenue": "TaiwanStockMonthRevenue"
                }[data_type]
                
                df = self.fetch_finmind_data(dataset, symbol, start_date, end_date)
                
                if not df.empty:
                    # 儲存資料
                    table_name = self.get_table_name(data_type)
                    if 'symbol' not in df.columns:
                        df['symbol'] = symbol
                    
                    self.insert_df(table_name, df, if_exists='append')
                    logger.info(f"✅ 成功: {symbol} {data_type} - {len(df)} 筆")
                else:
                    logger.warning(f"⚠️  無資料: {symbol} {data_type}")
                
                # 標記完成
                self.progress_tracker.mark_symbol_completed(symbol, data_type)
                
                # 每10個任務顯示API使用狀況
                if (i + 1) % 10 == 0:
                    usage = self.api_manager.get_usage_status()
                    logger.info(f"API 使用狀況: Key {usage['current_key_index']+1}/{usage['total_keys']}")
                    
                    # 每50個任務儲存進度
                    if (i + 1) % 50 == 0:
                        self.progress_tracker.save_progress()
                        logger.info("進度已儲存")
                
        except KeyboardInterrupt:
            logger.info("收到中斷信號，正在儲存進度...")
            self.progress_tracker.save_progress()
            logger.info("進度已儲存，可以稍後繼續執行")
            raise
        
        except Exception as e:
            logger.error(f"資料收集過程發生錯誤: {e}")
            self.progress_tracker.save_progress()
            raise
        
        finally:
            # 最終儲存進度
            self.progress_tracker.save_progress()
            
            # 顯示統計資料
            stats = self.progress_tracker.progress_data['statistics']
            logger.info("=== 收集完成統計 ===")
            logger.info(f"總請求數: {stats['total_requests']}")
            logger.info(f"成功請求: {stats['successful_requests']}")
            logger.info(f"失敗請求: {stats['failed_requests']}")
            logger.info(f"成功率: {stats['successful_requests']/max(stats['total_requests'],1)*100:.1f}%")


def main():
    """主函數"""
    print("=== 增強版資料收集器 ===")
    
    try:
        # 先測試籌碼面特徵
        print("1. 測試籌碼面特徵功能...")
        exec(open('tmp_rovodev_test_chip_features.py').read())
        
        print("\n2. 開始資料收集...")
        collector = EnhancedDataCollector()
        
        # 測試模式：先收集少數股票
        test_symbols = ["2330", "2317", "2603"]
        
        print(f"測試模式：收集 {test_symbols} 的資料")
        collector.collect_all_data(
            symbols=test_symbols,
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        
        print("✅ 測試完成！")
        print("如要收集全部180支股票，請執行：")
        print("collector.collect_all_data()  # 不指定symbols參數")
        
    except Exception as e:
        logger.error(f"執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()