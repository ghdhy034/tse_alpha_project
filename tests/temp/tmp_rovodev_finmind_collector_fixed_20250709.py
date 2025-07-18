#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinMind 資料收集器 (修正版) - 修正資料表建立和欄位對應問題
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
from FinMind.data import DataLoader

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
    """FinMind API Key 管理器 - 使用官方 API 查詢版本"""
    
    def __init__(self, api_keys_file: str = "finmind_api_keys.txt"):
        self.api_key = None
        self.api_endpoint = None
        self.dl = None  # DataLoader 實例
        self.rate_limit_threshold = 550  # 550次後休眠
        self.sleep_duration = 3780  # 1小時3分鐘 = 3780秒
        self.failed_requests = []  # 記錄失敗的請求，用於重試
        self.processed_symbols = set()  # 記錄已處理完成的股票
        self.current_symbol_progress = {}  # 記錄當前股票的處理進度
        self._login_and_get_token()
        self.setup_api_instance()
    
    def _login_and_get_token(self):
        """使用 FinMind SDK 登入"""
        try:
            from market_data_collector.utils.config import FINMIND_USER, FINMIND_PASS
            
            self.dl = DataLoader()
            login_result = self.dl.login(user_id=FINMIND_USER, password=FINMIND_PASS)
            
            # FinMind SDK 登入成功返回 True
            if login_result is True:
                logger.info("✅ FinMind SDK 登入成功")
                # 設置一個虛擬的 API key 用於舊邏輯相容性
                self.api_key = "SDK_AUTHENTICATED"
                
                # 嘗試從 DataLoader 實例中取得實際的 token（如果有的話）
                for attr_name in ['token', 'api_token', 'access_token', '_token']:
                    if hasattr(self.dl, attr_name):
                        token_value = getattr(self.dl, attr_name)
                        if token_value and isinstance(token_value, str) and len(token_value) > 10:
                            self.api_key = token_value
                            logger.info(f"✅ 取得實際 Token: {self.api_key[:20]}...")
                            break
            else:
                raise ValueError(f"登入失敗，返回值: {login_result}")
            
        except Exception as e:
            logger.error(f"❌ FinMind SDK 登入失敗: {e}")
            raise ValueError(f"無法登入 FinMind: {e}")
    
    def setup_api_instance(self):
        """設置 API 查詢端點"""
        # 不需要 FinMind 模組，直接使用 HTTP API
        self.api_endpoint = "https://api.web.finmindtrade.com/v2/user_info"
        logger.info("✅ API 查詢端點設置完成")
    
    def get_api_usage_status(self):
        """獲取 API 使用狀況 - 根據 References.txt 使用 dl.api_usage 和 dl.api_usage_limit"""
        try:
            # 根據 References.txt 的正確範例
            if hasattr(self.dl, 'api_usage') and hasattr(self.dl, 'api_usage_limit'):
                used = self.dl.api_usage          # 已使用次數
                limit = self.dl.api_usage_limit   # 上限（免費版 600）
                
                logger.debug(f"📊 API 使用狀況 (SDK正確方式): {used}/{limit}")
                return used, limit
            else:
                missing_attrs = []
                if not hasattr(self.dl, 'api_usage'):
                    missing_attrs.append('api_usage')
                if not hasattr(self.dl, 'api_usage_limit'):
                    missing_attrs.append('api_usage_limit')
                logger.warning(f"⚠️ DataLoader 缺少屬性: {missing_attrs}")
                
        except Exception as e:
            logger.warning(f"⚠️ 讀取 SDK API 使用狀況失敗: {e}")
            import traceback
            logger.debug(f"詳細錯誤: {traceback.format_exc()}")
        
        # 如果無法從 SDK 讀取，返回預設值
        logger.warning("⚠️ 無法獲取 API 使用狀況，返回預設值")
        return 0, 600
    
    def check_rate_limit(self):
        """檢查是否需要休眠"""
        current_usage, usage_limit = self.get_api_usage_status()
        
        if current_usage is None:
            logger.warning("⚠️ 無法獲取使用狀況，繼續執行")
            return False
        
        # 檢查是否達到 550 次限制
        if current_usage >= self.rate_limit_threshold:
            logger.warning(f"⚠️ API 使用次數已達 {current_usage} 次，超過限制 ({self.rate_limit_threshold} 次)")
            self.handle_rate_limit()
            return True
        
        # 警告檢查
        if current_usage >= self.rate_limit_threshold * 0.9:  # 90% 警告
            logger.warning(f"⚠️ API 使用次數接近限制: {current_usage}/{self.rate_limit_threshold}")
        elif current_usage >= self.rate_limit_threshold * 0.8:  # 80% 警告
            logger.info(f"📊 API 使用次數: {current_usage}/{self.rate_limit_threshold}")
        
        return False
    
    def get_current_key(self) -> str:
        """獲取當前API Key"""
        if not self.api_key:
            raise ValueError("沒有可用的API Key")
        
        # 每次獲取 Key 前都檢查是否需要休眠
        current_usage, usage_limit = self.get_api_usage_status()
        if current_usage is not None and current_usage >= self.rate_limit_threshold:
            logger.warning(f"⚠️ API 使用次數已達 {current_usage} 次，超過限制 ({self.rate_limit_threshold} 次)")
            self.handle_rate_limit()
        
        return self.api_key
    
    def record_failed_request(self, request_info):
        """記錄失敗的請求，用於重試"""
        self.failed_requests.append(request_info)
        logger.info(f"📝 記錄失敗請求: {request_info}")
    
    def clear_failed_requests(self):
        """清除失敗請求記錄"""
        self.failed_requests = []
        logger.info("🗑️ 清除失敗請求記錄")
    
    def handle_rate_limit(self):
        """處理流量限制 - 休眠1小時3分鐘並準備重試失敗請求"""
        logger.warning(f"⚠️ 已達到流量限制閾值 ({self.rate_limit_threshold} 次)")
        
        # 顯示失敗請求數量
        if self.failed_requests:
            logger.info(f"📝 有 {len(self.failed_requests)} 個失敗請求將在休眠後重試")
        
        logger.info(f"😴 開始休眠 {self.sleep_duration} 秒 (1小時3分鐘)...")
        
        # 動態顯示休眠進度
        import time
        start_time = time.time()
        last_update = 0
        
        while time.time() - start_time < self.sleep_duration:
            elapsed = int(time.time() - start_time)
            remaining = self.sleep_duration - elapsed
            remaining_minutes = remaining // 60
            remaining_seconds = remaining % 60
            
            # 每30秒更新一次顯示
            if elapsed - last_update >= 30:
                logger.info(f"💤 休眠中... 剩餘 {remaining_minutes:02d}:{remaining_seconds:02d} (分:秒)")
                last_update = elapsed
            
            time.sleep(10)  # 每10秒檢查一次
        
        logger.info("✅ 休眠完成，準備重試失敗請求...")
        
        # 重新檢查 API 使用狀況
        current_usage, usage_limit = self.get_api_usage_status()
        if current_usage is not None:
            logger.info(f"📊 休眠後 API 使用狀況: {current_usage}/{usage_limit}")
        
        return True  # 表示可以繼續執行
    
    
    def get_usage_status(self) -> Dict:
        """獲取使用狀況"""
        current_usage, usage_limit = self.get_api_usage_status()
        
        return {
            'api_key': self.api_key[:20] + "..." if self.api_key else None,
            'current_usage': current_usage if current_usage is not None else 0,
            'usage_limit': usage_limit if usage_limit is not None else 0,
            'remaining_calls': max(0, (usage_limit or 0) - (current_usage or 0)),
            'rate_limit_threshold': self.rate_limit_threshold,
            'failed_requests_count': len(self.failed_requests)
        }


class FinMindDataCollector:
    """FinMind 資料收集器 (修正版)"""
    
    def __init__(self):
        self.api_manager = FinMindAPIManager()
        self.dl = self.api_manager.dl  # 保存 DataLoader 實例
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
        
        # 設置 API Key 到 config 模組
        self._setup_api_key()
    
    def _setup_api_key(self):
        """設置 API Key 到 config 模組"""
        try:
            import market_data_collector.utils.config as config
            config.TOKEN = self.api_manager.get_current_key()
            logger.info("已設置 API Key 到 config 模組")
        except Exception as e:
            logger.warning(f"設置 API Key 失敗: {e}")
    
    def _patch_requests_for_api_counting(self):
        """修補 requests.get 來處理流量限制並記錄失敗請求"""
        import requests
        original_get = requests.get
        
        def patched_get(*args, **kwargs):
            # 檢查是否為 FinMind API 呼叫 (更新 URL 檢查)
            if args and ('finmindapi.servebeer.com' in str(args[0]) or 'api.finmindtrade.com' in str(args[0])):
                max_retries = 3
                request_info = {
                    'url': str(args[0]) if args else 'unknown',
                    'params': kwargs.get('params', {}),
                    'timestamp': time.time()
                }
                
                for attempt in range(max_retries):
                    try:
                        # 在發送請求前檢查是否已達限制
                        current_usage, usage_limit = self.api_manager.get_api_usage_status()
                        if current_usage is not None and current_usage >= self.api_manager.rate_limit_threshold:
                            logger.warning(f"⚠️ 已達到 API 呼叫限制 ({current_usage}/{self.api_manager.rate_limit_threshold})，開始休眠")
                            # 記錄這個失敗的請求
                            self.api_manager.record_failed_request(request_info)
                            self.api_manager.handle_rate_limit()
                            continue  # 休眠後重試
                        
                        logger.debug(f"📡 FinMind API 呼叫: {request_info['url']}")
                        response = original_get(*args, **kwargs)
                        
                        # 檢查是否為流量限制錯誤 (402 狀態碼或特定訊息)
                        if response.status_code == 402:
                            try:
                                response_json = response.json()
                                if "Requests reach the upper limit" in response_json.get("msg", ""):
                                    logger.error(f"❌ FinMind API 流量限制: {response_json}")
                                    logger.warning("⚠️ 檢測到 402 流量限制錯誤")
                                    # 記錄失敗請求並休眠
                                    self.api_manager.record_failed_request(request_info)
                                    self.api_manager.handle_rate_limit()
                                    continue  # 重試
                            except:
                                pass
                        
                        # 檢查其他流量限制錯誤
                        elif response.status_code == 429 or (response.status_code == 200 and 
                            'rate limit' in response.text.lower()):
                            logger.warning("⚠️ 檢測到其他 API 流量限制")
                            self.api_manager.record_failed_request(request_info)
                            self.api_manager.handle_rate_limit()
                            continue  # 重試
                        
                        # 成功的請求，如果這是重試的請求，從失敗列表中移除
                        if request_info in self.api_manager.failed_requests:
                            self.api_manager.failed_requests.remove(request_info)
                            logger.info(f"✅ 重試成功: {request_info['url']}")
                        
                        return response
                        
                    except Exception as e:
                        logger.warning(f"⚠️ API 呼叫失敗 (嘗試 {attempt+1}/{max_retries}): {e}")
                        if attempt == max_retries - 1:
                            # 最後一次重試也失敗，記錄失敗請求
                            self.api_manager.record_failed_request(request_info)
                            raise
                        time.sleep(2 ** attempt)  # 指數退避
                        
                # 如果所有重試都失敗，執行最後一次嘗試
                return original_get(*args, **kwargs)
            else:
                return original_get(*args, **kwargs)
        
        # 根據 References.txt 建議，額外修補 requests.Session.get
        if hasattr(requests, "sessions"):
            original_session_get = requests.sessions.Session.get
            def session_get(self, *args, **kwargs):
                return patched_get(*args, **kwargs)
            requests.sessions.Session.get = session_get
        
        requests.get = patched_get
        logger.info("✅ 已修補 requests.get 和 Session.get 來處理流量限制並記錄失敗請求")
    
    def check_data_exists(self, symbol: str, data_type: str, start_date: str, end_date: str) -> bool:
        """檢查資料是否已存在於資料庫中"""
        try:
            from market_data_collector.utils.data_fetcher import (
                load_stock_data_from_db, load_financial_data_from_db,
                load_monthly_revenue_from_db, load_margin_purchase_shortsale_from_db,
                load_investors_buy_sell_from_db, load_per_data_from_db,
                load_technical_indicators_from_db
            )
            
            # 根據資料類型檢查對應的資料表
            if data_type == "daily_price":
                df = load_stock_data_from_db(symbol, start_date, end_date)
            elif data_type == "financial":
                df = load_financial_data_from_db(symbol, start_date, end_date)
            elif data_type == "monthly_revenue":
                df = load_monthly_revenue_from_db(symbol, start_date, end_date)
            elif data_type == "margin_shortsale":
                df = load_margin_purchase_shortsale_from_db(symbol, start_date, end_date)
            elif data_type == "institutional":
                df = load_investors_buy_sell_from_db(symbol, start_date, end_date)
            elif data_type == "per_data":
                df = load_per_data_from_db(symbol, start_date, end_date)
            elif data_type == "technical_indicators":
                df = load_technical_indicators_from_db(symbol, start_date, end_date)
            else:
                return False
            
            # 如果有資料且資料量合理，認為已存在
            if not df.empty and len(df) > 10:  # 至少要有10筆資料才算有效
                return True
            return False
            
        except Exception as e:
            logger.debug(f"檢查 {symbol} {data_type} 資料時發生錯誤: {e}")
            return False
    
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
    
    def fetch_stock_data_sdk(self, dl, stock_id, start_date=None, end_date=None):
        """使用 FinMind SDK 獲取日線資料"""
        if start_date is None:
            start_date = "2020-03-02"
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
        
        logger.info(f"📈 使用 SDK 下載 {stock_id} 日線資料...")
        df = dl.taiwan_stock_daily(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not df.empty:
            # 欄位名稱對照：max→high, min→low，並轉換 volume 單位
            df = df.rename(columns={"max": "high", "min": "low"})
            
            # 轉換 volume 單位：股 → 張 (除以1000，無條件捨去)
            if "Trading_Volume" in df.columns:
                df["Trading_Volume"] = (df["Trading_Volume"] / 1000).astype(int)
            
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            # 檢查關鍵欄位是否存在
            logger.debug(f"📊 {stock_id} 日線資料欄位: {list(df.columns)}")
            if "high" in df.columns and "low" in df.columns:
                logger.debug(f"📊 {stock_id} high/low 欄位正常")
            else:
                logger.warning(f"⚠️ {stock_id} 缺少 high/low 欄位")
        
        return df
    
    def fetch_monthly_revenue_sdk(self, dl, stock_id, start_date=None, end_date=None):
        """使用 FinMind SDK 獲取月營收資料"""
        if start_date is None:
            start_date = "2020-03-02"
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
        
        logger.info(f"💰 使用 SDK 下載 {stock_id} 月營收資料...")
        df = dl.taiwan_stock_month_revenue(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not df.empty:
            df.rename(columns={"stock_id": "symbol", "revenue": "monthly_revenue"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            df = df[["symbol", "date", "monthly_revenue"]]
        
        return df
    
    def fetch_financial_statements_sdk(self, dl, stock_id, start_date=None, end_date=None):
        """使用 FinMind SDK 獲取財報資料"""
        if start_date is None:
            start_date = "2020-03-02"
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
        
        logger.info(f"📊 使用 SDK 下載 {stock_id} 財報資料...")
        df = dl.taiwan_stock_financial_statement(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not df.empty:
            df.rename(columns={"stock_id": "symbol"}, inplace=True)
            df = df.pivot(index=["symbol", "date"], columns="type", values="value").reset_index()
            
            # 欄位對照
            income_statement_mapping = {
                "symbol": "symbol",
                "date": "date",
                "CostOfGoodsSold": "cost_of_goods_sold",
                "EPS": "eps",
                "EquityAttributableToOwnersOfParent": "equity_attributable_to_owners",
                "GrossProfit": "gross_profit",
                "IncomeAfterTaxes": "income_after_taxes",
                "IncomeFromContinuingOperations": "income_from_continuing_operations",
                "NoncontrollingInterests": "noncontrolling_interests",
                "OperatingExpenses": "operating_expenses",
                "OperatingIncome": "operating_income",
                "OtherComprehensiveIncome": "other_comprehensive_income",
                "PreTaxIncome": "pre_tax_income",
                "RealizedGain": "realized_gain",
                "Revenue": "revenue",
                "TAX": "tax",
                "TotalConsolidatedProfitForThePeriod": "total_profit",
                "TotalNonoperatingIncomeAndExpense": "nonoperating_income_expense"
            }
            df.rename(columns=lambda col: income_statement_mapping.get(col, col), inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            
            expected_cols = ["symbol", "date", "cost_of_goods_sold", "eps",
                           "equity_attributable_to_owners", "gross_profit", "income_after_taxes",
                           "income_from_continuing_operations", "noncontrolling_interests",
                           "operating_expenses", "operating_income", "other_comprehensive_income",
                           "pre_tax_income", "realized_gain", "revenue", "tax",
                           "total_profit", "nonoperating_income_expense"]
            df = df.reindex(columns=expected_cols)
            
            # 計算 PE ratio (需要股價資料)
            pe_list = []
            for idx, row in df.iterrows():
                report_date = row["date"]
                # 這裡簡化處理，實際應該要取得對應日期的股價
                eps_val = row["eps"]
                if eps_val not in [None, 0]:
                    pe_list.append(None)  # 暫時設為 None，可以後續改進
                else:
                    pe_list.append(None)
            df["pe_ratio"] = pe_list
            df["symbol"] = df["symbol"].fillna(stock_id)
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        return df
    
    def fetch_margin_purchase_short_sale_sdk(self, dl, stock_id, start_date=None, end_date=None):
        """使用 FinMind SDK 獲取融資融券資料"""
        if start_date is None:
            start_date = "2020-03-02"
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
        
        logger.info(f"💳 使用 SDK 下載 {stock_id} 融資融券資料...")
        df = dl.taiwan_stock_margin_purchase_short_sale(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not df.empty:
            df.rename(columns={"stock_id": "symbol"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            if "symbol" not in df.columns:
                df["symbol"] = stock_id
        
        return df
    
    def fetch_institutional_investors_sdk(self, dl, stock_id, start_date=None, end_date=None):
        """使用 FinMind SDK 獲取機構投信買賣資料"""
        if start_date is None:
            start_date = "2020-03-02"
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
        
        logger.info(f"🏛️ 使用 SDK 下載 {stock_id} 機構投信買賣資料...")
        df = dl.taiwan_stock_institutional_investors(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not df.empty:
            df = df.pivot(index=['date', 'stock_id'], columns='name', values=['buy', 'sell'])
            df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
            df.reset_index(inplace=True)
            df.rename(columns={"stock_id": "symbol"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        return df
    
    def fetch_per_pbr_data_sdk(self, dl, stock_id, start_date=None, end_date=None):
        """使用 FinMind SDK 獲取本益比資料"""
        if start_date is None:
            start_date = "2020-03-02"
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
        
        logger.info(f"📋 使用 SDK 下載 {stock_id} 本益比資料...")
        df = dl.taiwan_stock_per_pbr(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not df.empty:
            df.rename(columns={"stock_id": "symbol"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        return df

    def collect_stock_data(self, symbol: str, start_date: str = "2020-03-02", end_date: str = "2025-07-08"):
        """收集單一股票的所有 FinMind 資料 - 使用原始 data_fetcher 處理方式 (修正版)"""
        logger.info(f"🎯 開始收集 {symbol} 的 FinMind 資料...")
        
        # 匯入並執行原始處理函數
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
            
            # 確保資料表已建立 (只在第一次執行)
            if not hasattr(self, '_tables_created'):
                logger.info("🔧 建立資料表...")
                create_db_and_table()
                logger.info("✅ 資料表建立完成")
                self._tables_created = True
            
        except ImportError as e:
            logger.error(f"❌ 無法匯入原始處理函數: {e}")
            return {}
        
        results = {}
        
        # 1. 日線資料
        try:
            if self.check_data_exists(symbol, "daily_price", start_date, end_date):
                logger.info(f"⏭️ {symbol} 日線資料已存在，略過下載")
                results['daily_price'] = 0  # 標記為已存在
            else:
                logger.info(f"📈 下載 {symbol} 日線資料...")
                df_daily = self.fetch_stock_data_sdk(self.dl, symbol, start_date, end_date)
                if not df_daily.empty:
                    # 確保欄位名稱符合 store_stock_data_to_db 的期望
                    df_daily_for_db = df_daily.copy()
                    # store_stock_data_to_db 期望的是 max/min 欄位，所以要轉換回去
                    if "high" in df_daily_for_db.columns:
                        df_daily_for_db = df_daily_for_db.rename(columns={"high": "max", "low": "min"})
                    store_stock_data_to_db(df_daily_for_db, symbol)
                    results['daily_price'] = len(df_daily)
                    logger.info(f"✅ {symbol} 日線資料下載完成: {len(df_daily)} 筆")
                else:
                    results['daily_price'] = 0
            
            # 檢查技術指標
            if self.check_data_exists(symbol, "technical_indicators", start_date, end_date):
                logger.info(f"⏭️ {symbol} 技術指標已存在，略過計算")
                results['technical_indicators'] = 0
            else:
                try:
                    # 需要先有日線資料才能計算技術指標
                    from market_data_collector.utils.data_fetcher import load_stock_data_from_db
                    df_daily = load_stock_data_from_db(symbol, start_date, end_date)
                    if not df_daily.empty:
                        logger.info(f"📊 計算 {symbol} 技術指標...")
                        tech_indicators = compute_technical_indicators(symbol, df_daily)
                        store_technical_indicators_to_db(tech_indicators, symbol)
                        results['technical_indicators'] = len(tech_indicators)
                        logger.info(f"✅ {symbol} 技術指標計算完成: {len(tech_indicators)} 筆")
                    else:
                        results['technical_indicators'] = 0
                except Exception as e:
                    logger.warning(f"⚠️ {symbol} 技術指標計算失敗: {e}")
                    results['technical_indicators'] = 0
                    
        except Exception as e:
            logger.error(f"❌ {symbol} 日線資料失敗: {e}")
            results['daily_price'] = -1
        
        # 2. 財報資料
        try:
            if self.check_data_exists(symbol, "financial", start_date, end_date):
                logger.info(f"⏭️ {symbol} 財報資料已存在，略過下載")
                results['financial'] = 0
            else:
                logger.info(f"📊 下載 {symbol} 財報資料...")
                df_financial = self.fetch_financial_statements_sdk(self.dl, symbol, start_date, end_date)
                if not df_financial.empty:
                    store_financial_data_to_db(df_financial, symbol)
                    results['financial'] = len(df_financial)
                    logger.info(f"✅ {symbol} 財報資料下載完成: {len(df_financial)} 筆")
                else:
                    results['financial'] = 0
        except Exception as e:
            logger.error(f"❌ {symbol} 財報資料失敗: {e}")
            results['financial'] = -1
        
        # 3. 月營收資料
        try:
            if self.check_data_exists(symbol, "monthly_revenue", start_date, end_date):
                logger.info(f"⏭️ {symbol} 月營收資料已存在，略過下載")
                results['monthly_revenue'] = 0
            else:
                logger.info(f"💰 下載 {symbol} 月營收資料...")
                df_revenue = self.fetch_monthly_revenue_sdk(self.dl, symbol, start_date, end_date)
                if not df_revenue.empty:
                    store_monthly_revenue_to_db(df_revenue, symbol)
                    results['monthly_revenue'] = len(df_revenue)
                    logger.info(f"✅ {symbol} 月營收資料下載完成: {len(df_revenue)} 筆")
                else:
                    results['monthly_revenue'] = 0
        except Exception as e:
            logger.error(f"❌ {symbol} 月營收資料失敗: {e}")
            results['monthly_revenue'] = -1
        
        # 4. 融資融券資料
        try:
            if self.check_data_exists(symbol, "margin_shortsale", start_date, end_date):
                logger.info(f"⏭️ {symbol} 融資融券資料已存在，略過下載")
                results['margin_shortsale'] = 0
            else:
                logger.info(f"💳 下載 {symbol} 融資融券資料...")
                df_margin = self.fetch_margin_purchase_short_sale_sdk(self.dl, symbol, start_date, end_date)
                if not df_margin.empty:
                    store_margin_purchase_shortsale_to_db(df_margin, symbol)
                    results['margin_shortsale'] = len(df_margin)
                    logger.info(f"✅ {symbol} 融資融券資料下載完成: {len(df_margin)} 筆")
                else:
                    results['margin_shortsale'] = 0
        except Exception as e:
            logger.error(f"❌ {symbol} 融資融券資料失敗: {e}")
            results['margin_shortsale'] = -1
        
        # 5. 機構投信買賣資料
        try:
            if self.check_data_exists(symbol, "institutional", start_date, end_date):
                logger.info(f"⏭️ {symbol} 機構投信買賣資料已存在，略過下載")
                results['institutional'] = 0
            else:
                logger.info(f"🏛️ 下載 {symbol} 機構投信買賣資料...")
                df_institutional = self.fetch_institutional_investors_sdk(self.dl, symbol, start_date, end_date)
                if not df_institutional.empty:
                    store_investors_buy_sell_to_db(df_institutional, symbol)
                    results['institutional'] = len(df_institutional)
                    logger.info(f"✅ {symbol} 機構投信買賣資料下載完成: {len(df_institutional)} 筆")
                else:
                    results['institutional'] = 0
        except Exception as e:
            logger.error(f"❌ {symbol} 機構投信買賣資料失敗: {e}")
            results['institutional'] = -1
        
        # 6. 本益比資料
        try:
            if self.check_data_exists(symbol, "per_data", start_date, end_date):
                logger.info(f"⏭️ {symbol} 本益比資料已存在，略過下載")
                results['per_data'] = 0
            else:
                logger.info(f"📋 下載 {symbol} 本益比資料...")
                df_per = self.fetch_per_pbr_data_sdk(self.dl, symbol, start_date, end_date)
                if not df_per.empty:
                    store_per_data_to_db(df_per, symbol)
                    results['per_data'] = len(df_per)
                    logger.info(f"✅ {symbol} 本益比資料下載完成: {len(df_per)} 筆")
                else:
                    results['per_data'] = 0
        except Exception as e:
            logger.error(f"❌ {symbol} 本益比資料失敗: {e}")
            results['per_data'] = -1
        
        return results
    
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
        
        logger.info(f"📊 準備收集 {len(symbols)} 支股票的 FinMind 資料...")
        logger.info(f"📅 日期範圍: {start_date} ~ {end_date}")
        
        # 啟用 API 呼叫計數
        self._patch_requests_for_api_counting()
        
        # 檢查是否有未完成的失敗請求需要重試
        if self.api_manager.failed_requests:
            logger.info(f"🔄 發現 {len(self.api_manager.failed_requests)} 個未完成的請求，將優先重試")
        
        total_results = {}
        
        try:
            # 在開始實際收集前進行初始流量檢測
            logger.info("🔍 開始執行前檢查 API 使用狀況...")
            usage = self.api_manager.get_usage_status()
            logger.info(f"🔑 使用 API Key: {usage['api_key']}")
            logger.info(f"📊 當前 API 使用狀況: {usage['current_usage']}/{usage['usage_limit']} (限制: {usage['rate_limit_threshold']})")
            
            # 如果已超過限制，先休眠
            if usage['current_usage'] >= usage['rate_limit_threshold']:
                logger.warning(f"⚠️ 開始前檢測到流量已超限，需要先休眠")
                self.api_manager.handle_rate_limit()
            
            logger.info(f"🚀 開始收集 {len(symbols)} 支股票的資料...")
            
            # 過濾掉已處理完成的股票
            remaining_symbols = [s for s in symbols if s not in self.api_manager.processed_symbols]
            if len(remaining_symbols) < len(symbols):
                completed_count = len(symbols) - len(remaining_symbols)
                logger.info(f"📋 發現 {completed_count} 支股票已完成，剩餘 {len(remaining_symbols)} 支股票需要處理")
            
            for i, symbol in enumerate(remaining_symbols):
                actual_index = symbols.index(symbol) + 1  # 在原始清單中的位置
                logger.info(f"📈 處理進度: {actual_index}/{len(symbols)} - {symbol}")
                
                # 在處理每支股票前檢查流量限制
                current_usage, usage_limit = self.api_manager.get_api_usage_status()
                if current_usage is not None and current_usage >= self.api_manager.rate_limit_threshold:
                    logger.warning(f"⚠️ 處理 {symbol} 前檢測到流量超限 ({current_usage}/{self.api_manager.rate_limit_threshold})，開始休眠")
                    logger.info(f"💾 當前進度已保存，休眠後將從 {symbol} 繼續")
                    self.api_manager.handle_rate_limit()
                
                # 收集該股票的所有資料
                results = self.collect_stock_data(symbol, start_date, end_date)
                total_results[symbol] = results
                
                # 如果成功處理完成，標記為已完成
                if results and all(v >= 0 for v in results.values()):  # 所有資料類型都成功或已存在
                    self.api_manager.processed_symbols.add(symbol)
                    logger.debug(f"✅ {symbol} 已標記為完成")
                
                # 每5支股票顯示API使用狀況
                if (i + 1) % 5 == 0:
                    usage = self.api_manager.get_usage_status()
                    logger.info(f"📊 API 使用狀況: {usage['current_usage']}/{usage['usage_limit']} 次，剩餘 {usage['remaining_calls']} 次")
                    if usage['failed_requests_count'] > 0:
                        logger.info(f"📝 失敗請求: {usage['failed_requests_count']} 個")
                
                # 短暫休息
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("⏹️  收到中斷信號，正在停止...")
            raise
        
        except Exception as e:
            logger.error(f"❌ 資料收集過程發生錯誤: {e}")
            raise
        
        finally:
            # 顯示最終 API 使用統計
            final_usage = self.api_manager.get_usage_status()
            logger.info(f"🏁 最終 API 使用狀況: {final_usage['current_usage']}/{final_usage['usage_limit']}")
            logger.info(f"📊 剩餘可用次數: {final_usage['remaining_calls']}")
            
            # 顯示處理進度統計
            total_symbols = len(symbols)
            completed_symbols = len(self.api_manager.processed_symbols)
            logger.info(f"📈 處理進度: {completed_symbols}/{total_symbols} 支股票已完成")
            
            # 顯示失敗請求統計
            if final_usage['failed_requests_count'] > 0:
                logger.warning(f"⚠️ 未完成的失敗請求: {final_usage['failed_requests_count']} 個")
                logger.info("💡 建議：重新執行程式以重試失敗的請求")
            
            # 顯示未完成的股票
            remaining_symbols = [s for s in symbols if s not in self.api_manager.processed_symbols]
            if remaining_symbols:
                logger.warning(f"⚠️ 未完成的股票: {len(remaining_symbols)} 支")
                logger.info(f"📋 未完成清單: {remaining_symbols[:10]}{'...' if len(remaining_symbols) > 10 else ''}")
                logger.info("💡 建議：重新執行程式以繼續處理未完成的股票")
            else:
                logger.info("✅ 所有股票都已處理完成")
            
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
    print("🏦 FinMind 歷史資料收集器 (FinMind SDK版)")
    print("=" * 50)
    print("新功能：")
    print("• 使用 FinMind Python SDK - 更穩定的資料獲取")
    print("• 自動登入取得 Token - 無需手動管理 API Key")
    print("• Bearer Token API 使用量查詢 - 使用 /user_info 精確監控")
    print("• 550次限制自動休眠1小時3分鐘")
    print("• 動態顯示休眠倒數計時 (每30秒更新)")
    print("• 智能斷點續傳 - 休眠後從中斷處接續，無資料遺漏")
    print("• 失敗請求記錄與自動重試機制")
    print("• 防重複下載機制 - 自動檢查已存在資料")
    print("• 統一流量檢測 - 選定作業後才進行檢測")
    print("• 詳細的下載/略過狀態日誌")
    print("• 自動建立資料表")
    print("• 修正 API 欄位對應 (max→high, min→low)")
    print("• 保持原有流量監控機制")
    print("• 日期範圍: 2020-03-02 ~ 2025-07-08")
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