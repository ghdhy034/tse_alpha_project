#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shioaji 分鐘線資料收集器 - 專門收集5分鐘K線資料 (更新到2025-07-08)
"""
import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "data_pipeline"))
sys.path.insert(0, str(current_dir / "market_data_collector"))

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ShioajiFlowMonitor:
    """Shioaji API 流量監控器 - 使用官方 api.usage() 方法"""
    
    def __init__(self, api=None):
        self.api = api
        self.last_usage = None
    
    def set_api(self, api):
        """設定 API 實例"""
        self.api = api
    
    def get_usage_status(self):
        """獲取 API 使用狀況"""
        if not self.api:
            logger.debug("API 物件為 None，無法獲取使用狀況")
            return None
        
        try:
            # 檢查 API 是否還有效
            if not hasattr(self.api, 'usage'):
                logger.warning("⚠️ API 物件沒有 usage 方法")
                return None
            
            # 使用官方 api.usage() 方法
            usage_status = self.api.usage()
            self.last_usage = usage_status
            return usage_status
        except AttributeError as e:
            logger.warning(f"⚠️ API 物件屬性錯誤: {e}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ 無法獲取 API 使用狀況: {e}")
            return None
    
    def show_status(self):
        """顯示使用狀況"""
        usage_status = self.get_usage_status()
        
        if usage_status:
            try:
                # 解析 UsageStatus
                connections = getattr(usage_status, 'connections', 0)
                bytes_used = getattr(usage_status, 'bytes', 0)
                limit_bytes = getattr(usage_status, 'limit_bytes', 0)
                remaining_bytes = getattr(usage_status, 'remaining_bytes', 0)
                
                # 轉換為 MB
                bytes_used_mb = bytes_used / (1024 * 1024)
                limit_mb = limit_bytes / (1024 * 1024)
                remaining_mb = remaining_bytes / (1024 * 1024)
                
                # 計算使用百分比
                if limit_bytes > 0:
                    percentage = (bytes_used / limit_bytes) * 100
                else:
                    percentage = 0
                
                logger.info("=" * 40)
                logger.info(f"📊 Shioaji API 流量狀況")
                logger.info("=" * 40)
                logger.info(f"🔗 連線數: {connections}")
                logger.info(f"📈 已使用: {bytes_used_mb:.1f} MB")
                logger.info(f"📊 總限制: {limit_mb:.1f} MB")
                logger.info(f"💾 剩餘: {remaining_mb:.1f} MB")
                logger.info(f"⚡ 使用率: {percentage:.1f}%")
                logger.info("=" * 40)
                
                # 警告檢查
                if percentage > 85:
                    logger.warning(f"⚠️ 流量使用率已達 {percentage:.1f}%，接近限制！")
                elif percentage > 90:
                    logger.error(f"🚨 流量使用率已達 {percentage:.1f}%，即將達到限制！")
                
                return {
                    'connections': connections,
                    'bytes_used': bytes_used,
                    'limit_bytes': limit_bytes,
                    'remaining_bytes': remaining_bytes,
                    'percentage': percentage
                }
                
            except Exception as e:
                logger.error(f"❌ 解析 UsageStatus 失敗: {e}")
                logger.info(f"📊 原始 UsageStatus: {usage_status}")
                return None
        else:
            logger.info("📊 Shioaji API 流量狀況: 無法獲取（API 可能已登出）")
            return None
    
    def check_flow_limit(self, threshold_percentage: float = 95.0) -> bool:
        """檢查是否接近流量限制"""
        usage_status = self.get_usage_status()
        
        if usage_status:
            try:
                bytes_used = getattr(usage_status, 'bytes', 0)
                limit_bytes = getattr(usage_status, 'limit_bytes', 0)
                
                if limit_bytes > 0:
                    percentage = (bytes_used / limit_bytes) * 100
                    return percentage >= threshold_percentage
                    
            except Exception as e:
                logger.warning(f"⚠️ 檢查流量限制失敗: {e}")
        
        return False

class ShioajiDataCollector:
    """Shioaji 分鐘線資料收集器"""
    
    def __init__(self):
        self.api = None
        self.flow_monitor = None
        
        # 匯入必要模組
        try:
            from market_data_collector.utils.db import insert_df, query_df
            from market_data_collector.utils.config import (
                SHIOAJI_USER, SHIOAJI_PASS, SHIOAJI_CA_PATH, SHIOAJI_CA_PASS, SHIOAJI_SIMULATION
            )
            
            self.insert_df = insert_df
            self.query_df = query_df
            self.shioaji_config = {
                'user': SHIOAJI_USER,
                'pass': SHIOAJI_PASS,
                'ca_path': SHIOAJI_CA_PATH,
                'ca_pass': SHIOAJI_CA_PASS,
                'simulation': SHIOAJI_SIMULATION
            }
            
            # 嘗試匯入 data_fetcher，如果失敗則跳過資料表建立
            try:
                from market_data_collector.utils.data_fetcher import create_db_and_table
                
                # 確保資料表已建立 (包含 candlesticks_min)
                if not hasattr(self, '_tables_created'):
                    logger.info("🔧 建立資料表 (包含 candlesticks_min)...")
                    create_db_and_table()
                    logger.info("✅ 資料表建立完成")
                    self._tables_created = True
                    
            except ImportError as e:
                logger.warning(f"⚠️ 無法匯入 data_fetcher: {e}")
                logger.warning("⚠️ 跳過資料表建立，請確保資料表已存在")
                
        except ImportError as e:
            logger.error(f"無法匯入必要模組: {e}")
            raise
    
    def login_shioaji(self):
        """登入 Shioaji API"""
        try:
            import shioaji as sj
            
            logger.info("🔐 正在登入 Shioaji API...")
            
            # 建立API連線
            self.api = sj.Shioaji(simulation=self.shioaji_config['simulation'])
            
            # 登入 (設定 fetch_contract=False 節省流量)
            self.api.login(
                self.shioaji_config['user'],
                self.shioaji_config['pass'],
                fetch_contract=False
            )
            
            logger.info("💡 已設定 fetch_contract=False 以節省流量")
            
            # 啟用憑證
            self.api.activate_ca(
                ca_path=self.shioaji_config['ca_path'],
                ca_passwd=self.shioaji_config['ca_pass'],
                person_id=self.shioaji_config['user']
            )
            
            logger.info("✅ Shioaji 登入成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ Shioaji 登入失敗: {e}")
            return False
    
    def setup_flow_monitor(self):
        """設定流量監控"""
        try:
            # 使用新的 Shioaji 流量監控器
            self.flow_monitor = ShioajiFlowMonitor(self.api)
            logger.info("📊 Shioaji 流量監控器已啟動")
        except Exception as e:
            logger.warning(f"⚠️ 流量監控器啟動失敗: {e}")
            self.flow_monitor = None
    
    def get_stock_list(self) -> List[str]:
        """獲取股票清單 - 180支股票"""
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
            "2891","2892","2812","3665","2834","2850","2801","2836","2845","4807",
            "3702","3706","4560","8478","4142","4133","6525","6548","6843","1513",
            "1514","1516","1521","1522","1524","1533","1708","3019","5904","5906",
            "5902","6505","6806","6510","2207","2204","2231","1736","4105","4108",
            "4162","1909","1702","9917","1217","1218","1737","1783","3708","1795"
        ]
        
        all_stocks = group_A + group_B + group_C
        all_stocks = list(set(all_stocks))  # 去重
        all_stocks.sort()
        
        logger.info(f"準備收集 {len(all_stocks)} 支股票的分鐘線資料")
        return all_stocks
    
    def fetch_minute_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """獲取單一股票的分鐘線資料"""
        try:
            if not self.api:
                logger.error("❌ Shioaji API 未登入")
                return pd.DataFrame()
            
            logger.info(f"📡 請求分鐘線資料: {symbol} ({start_date} ~ {end_date})")
            
            # 取得股票合約 (因為設定 fetch_contract=False，需要手動獲取)
            try:
                # 檢查是否已有 Contracts 屬性
                if hasattr(self.api, 'Contracts') and hasattr(self.api.Contracts, 'Stocks'):
                    contract = self.api.Contracts.Stocks[symbol]
                    logger.debug(f"✅ 使用已載入的 {symbol} 合約")
                else:
                    raise AttributeError("Contracts 屬性不存在")
            except (KeyError, AttributeError):
                # 如果合約不存在或 Contracts 屬性不存在，手動獲取合約資料
                logger.info(f"🔄 手動獲取 {symbol} 合約資料...")
                try:
                    # 獲取股票合約
                    contracts = self.api.fetch_contracts(contract_download=True)
                    
                    # 再次嘗試訪問合約
                    if hasattr(self.api, 'Contracts') and hasattr(self.api.Contracts, 'Stocks'):
                        contract = self.api.Contracts.Stocks[symbol]
                        logger.info(f"✅ 成功獲取 {symbol} 合約")
                    else:
                        logger.error(f"❌ 獲取合約後仍無法訪問 Contracts.Stocks")
                        return pd.DataFrame()
                except Exception as e:
                    logger.error(f"❌ 無法獲取 {symbol} 合約: {e}")
                    return pd.DataFrame()
            
            # 取得K線資料
            kbars = self.api.kbars(
                contract=contract,
                start=start_date,
                end=end_date,
                timeout=30000
            )
            
            if kbars and hasattr(kbars, '__dict__'):
                # 轉換為DataFrame (修復版本)
                df = pd.DataFrame({**kbars})
                
                if not df.empty:
                    # 添加symbol欄位
                    df['symbol'] = symbol
                    
                    # 轉換時間格式
                    if 'ts' in df.columns:
                        df['ts'] = pd.to_datetime(df['ts'])
                    
                    # 重新排序欄位
                    columns_order = ['symbol', 'ts', 'Open', 'High', 'Low', 'Close', 'Volume']
                    available_columns = [col for col in columns_order if col in df.columns]
                    df = df[available_columns]
                    
                    # 顯示流量監控狀況 (每次獲取資料後都顯示)
                    if self.flow_monitor:
                        logger.info(f"📊 {symbol} 資料獲取後流量狀況:")
                        self.flow_monitor.show_status()
                    
                    logger.info(f"✅ 成功獲取 {len(df)} 筆分鐘線資料: {symbol}")
                    return df
                else:
                    logger.warning(f"⚠️  {symbol} 無分鐘線資料")
                    return pd.DataFrame()
            else:
                logger.warning(f"⚠️  {symbol} API 回傳空資料")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ 獲取 {symbol} 分鐘線資料失敗: {e}")
            return pd.DataFrame()
    
    def aggregate_to_5min(self, df: pd.DataFrame) -> pd.DataFrame:
        """將1分鐘資料聚合為5分鐘"""
        if df.empty:
            return df
        
        try:
            # 設定時間索引
            df_copy = df.copy()
            df_copy.set_index('ts', inplace=True)
            
            # 5分鐘聚合
            agg_dict = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            
            # 使用 '5min' 而不是 '5T'
            df_5min = df_copy.resample('5min').agg(agg_dict)
            
            # 移除空值行
            df_5min = df_5min.dropna()
            
            # 重置索引
            df_5min.reset_index(inplace=True)
            
            # 添加symbol欄位
            if 'symbol' in df.columns:
                df_5min['symbol'] = df['symbol'].iloc[0]
            
            # 計算VWAP
            if 'Volume' in df_5min.columns and df_5min['Volume'].sum() > 0:
                df_5min['vwap'] = (df_5min['High'] + df_5min['Low'] + df_5min['Close']) / 3
            else:
                df_5min['vwap'] = df_5min['Close']
            
            logger.info(f"📊 聚合為 {len(df_5min)} 筆5分鐘資料")
            return df_5min
            
        except Exception as e:
            logger.error(f"❌ 聚合5分鐘資料失敗: {e}")
            return pd.DataFrame()
    
    def format_for_candlesticks_min(self, df_5min: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """將5分鐘資料格式化為 candlesticks_min 資料表格式"""
        try:
            # 檢查輸入資料
            if df_5min.empty:
                logger.error("❌ 輸入的5分鐘資料為空")
                return pd.DataFrame()
            
            if 'ts' not in df_5min.columns:
                logger.error(f"❌ 輸入資料缺少 'ts' 欄位，現有欄位: {list(df_5min.columns)}")
                return pd.DataFrame()
            
            logger.info(f"🔧 開始格式化 {len(df_5min)} 筆5分鐘資料...")
            logger.info(f"📋 輸入欄位: {list(df_5min.columns)}")
            
            # 建立新的 DataFrame，逐一設定欄位
            df_formatted = pd.DataFrame(index=df_5min.index)
            
            # 設定固定值欄位
            df_formatted['market'] = 'TW'
            df_formatted['symbol'] = symbol
            df_formatted['interval'] = '5min'
            
            # 設定時間戳記
            df_formatted['timestamp'] = df_5min['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 設定價格和成交量欄位 - 直接使用欄位名稱
            price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            target_columns = ['open', 'high', 'low', 'close', 'volume']
            
            for price_col, target_col in zip(price_columns, target_columns):
                if price_col in df_5min.columns:
                    df_formatted[target_col] = df_5min[price_col]
                    logger.info(f"✅ 設定 {target_col}: {price_col}")
                else:
                    df_formatted[target_col] = 0.0
                    logger.warning(f"⚠️  欄位 {price_col} 不存在，設為 0")
            
            # 檢查結果
            logger.info(f"📊 格式化結果:")
            logger.info(f"   market: {df_formatted['market'].iloc[0] if len(df_formatted) > 0 else 'N/A'}")
            logger.info(f"   symbol: {df_formatted['symbol'].iloc[0] if len(df_formatted) > 0 else 'N/A'}")
            logger.info(f"   timestamp 範例: {df_formatted['timestamp'].iloc[0] if len(df_formatted) > 0 else 'N/A'}")
            logger.info(f"   open 範例: {df_formatted['open'].iloc[0] if len(df_formatted) > 0 else 'N/A'}")
            
            # 最終檢查
            if df_formatted.empty:
                logger.error("❌ 格式化後資料為空")
                return pd.DataFrame()
            
            # 檢查必要欄位是否有值
            required_fields = ['market', 'symbol', 'timestamp']
            for field in required_fields:
                if df_formatted[field].isnull().any():
                    logger.error(f"❌ 必要欄位 {field} 包含空值")
                    return pd.DataFrame()
            
            logger.info(f"✅ 成功格式化 {len(df_formatted)} 筆資料為 candlesticks_min 格式")
            return df_formatted
            
        except Exception as e:
            logger.error(f"❌ 格式化 candlesticks_min 資料失敗: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def check_minute_data_exists(self, symbol: str, start_date: str, end_date: str) -> bool:
        """檢查分鐘線資料是否已存在於資料庫中"""
        try:
            # 查詢資料庫中是否已有該股票在指定時間範圍的資料
            query_sql = """
            SELECT COUNT(*) as count 
            FROM candlesticks_min 
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            """
            
            # 轉換日期格式
            start_timestamp = f"{start_date} 00:00:00"
            end_timestamp = f"{end_date} 23:59:59"
            
            df_check = self.query_df(query_sql, (symbol, start_timestamp, end_timestamp))
            
            if not df_check.empty and df_check['count'].iloc[0] > 0:
                record_count = df_check['count'].iloc[0]
                logger.info(f"📊 {symbol} 已有 {record_count} 筆分鐘線資料在 {start_date} ~ {end_date}")
                
                # 如果資料量合理（至少100筆），認為已存在
                if record_count >= 100:
                    return True
                else:
                    logger.info(f"⚠️ {symbol} 資料量較少 ({record_count} 筆)，重新下載")
                    return False
            else:
                return False
                
        except Exception as e:
            logger.debug(f"檢查 {symbol} 分鐘線資料時發生錯誤: {e}")
            return False

    def collect_stock_minute_data(self, symbol: str, start_date: str = "2020-03-02", end_date: str = "2025-07-08"):
        """收集單一股票的分鐘線資料"""
        logger.info(f"🎯 開始收集 {symbol} 的分鐘線資料...")
        
        # 顯示開始處理前的流量狀況
        if self.flow_monitor:
            logger.info(f"📊 開始處理 {symbol} 前的詳細流量狀況:")
            self.flow_monitor.show_status()
        
        # 檢查資料是否已存在
        if self.check_minute_data_exists(symbol, start_date, end_date):
            logger.info(f"⏭️ {symbol} 分鐘線資料已存在，略過下載")
            # 即使略過也顯示流量狀況
            if self.flow_monitor:
                logger.info(f"📊 略過 {symbol} 後的流量狀況:")
                self.flow_monitor.show_status()
            return 0  # 標記為已存在
        
        try:
            # 獲取1分鐘資料
            df_1min = self.fetch_minute_data(symbol, start_date, end_date)
            
            if not df_1min.empty:
                # 聚合為5分鐘
                df_5min = self.aggregate_to_5min(df_1min)
                
                if not df_5min.empty:
                    # 轉換為 candlesticks_min 格式並儲存
                    df_formatted = self.format_for_candlesticks_min(df_5min, symbol)
                    
                    if not df_formatted.empty:
                        # 檢查是否有重複資料，如果有則先刪除
                        try:
                            from market_data_collector.utils.db import execute_sql
                            
                            # 刪除該股票在相同時間範圍的舊資料
                            min_timestamp = df_formatted['timestamp'].min()
                            max_timestamp = df_formatted['timestamp'].max()
                            
                            delete_sql = """
                            DELETE FROM candlesticks_min 
                            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                            """
                            execute_sql(delete_sql, (symbol, min_timestamp, max_timestamp))
                            logger.info(f"🗑️ 清除 {symbol} 在 {min_timestamp} ~ {max_timestamp} 的舊資料")
                            
                        except ImportError:
                            logger.warning("⚠️ 無法匯入 execute_sql，跳過清除舊資料")
                        except Exception as e:
                            logger.warning(f"⚠️ 清除舊資料失敗: {e}")
                        
                        # 插入新資料
                        self.insert_df('candlesticks_min', df_formatted, if_exists='append')
                        logger.info(f"💾 成功儲存 {len(df_formatted)} 筆5分鐘資料到 candlesticks_min: {symbol}")
                        
                        # 顯示資料儲存後的流量狀況
                        if self.flow_monitor:
                            logger.info(f"📊 {symbol} 資料儲存後的流量狀況:")
                            self.flow_monitor.show_status()
                        
                        return len(df_formatted)
                    else:
                        logger.error(f"❌ {symbol} 格式化後資料為空")
                        return 0
                else:
                    logger.warning(f"⚠️  {symbol} 聚合5分鐘資料失敗")
                    return 0
            else:
                logger.warning(f"⚠️  {symbol} 無1分鐘資料")
                return 0
                
        except Exception as e:
            logger.error(f"❌ 處理 {symbol} 分鐘線資料失敗: {e}")
            
            # 即使失敗也顯示流量狀況
            if self.flow_monitor:
                logger.info(f"📊 {symbol} 處理失敗後的流量狀況:")
                self.flow_monitor.show_status()
            
            return -1
    
    def collect_all_minute_data(self, symbols: Optional[List[str]] = None,
                               start_date: str = "2020-03-02",
                               end_date: str = "2025-07-08",
                               test_mode: bool = True):
        """收集所有股票的分鐘線資料"""
        if symbols is None:
            symbols = self.get_stock_list()
        
        if test_mode:
            # 測試模式：只收集前3支股票，使用短時間範圍節省流量
            symbols = symbols[:3]
            start_date = "2024-12-01"  # 測試模式使用短時間範圍
            end_date = "2024-12-31"
            logger.info(f"🧪 測試模式：收集 {symbols} 的分鐘線資料")
            logger.info(f"🧪 測試模式使用短時間範圍節省流量: {start_date} ~ {end_date}")
        
        logger.info(f"📊 開始收集 {len(symbols)} 支股票的分鐘線資料...")
        logger.info(f"📅 日期範圍: {start_date} ~ {end_date}")
        
        # 登入API
        if not self.login_shioaji():
            logger.error("❌ 無法登入 Shioaji，停止執行")
            return
        
        # 設定流量監控 (必須在登入後設置)
        self.setup_flow_monitor()
        
        # 顯示初始流量狀況
        if self.flow_monitor:
            logger.info("📊 初始流量狀況:")
            self.flow_monitor.show_status()
        
        results = {}
        
        try:
            for i, symbol in enumerate(symbols):
                logger.info(f"📈 處理進度: {i+1}/{len(symbols)} - {symbol}")
                
                # 每支股票處理前顯示流量狀況
                if self.flow_monitor:
                    logger.info(f"📊 處理 {symbol} 前的流量狀況:")
                    usage_info = self.flow_monitor.show_status()
                    
                    # 檢查流量限制
                    if usage_info and usage_info.get('percentage', 0) >= 95.0:
                        logger.warning("🚨 流量使用率已達 95%，停止收集以避免超限")
                        break
                
                # 收集該股票的分鐘線資料
                result = self.collect_stock_minute_data(symbol, start_date, end_date)
                results[symbol] = result
                
                # 每支股票處理後顯示流量狀況
                if self.flow_monitor:
                    logger.info(f"📊 處理 {symbol} 後的流量狀況:")
                    usage_info = self.flow_monitor.show_status()
                    
                    # 如果流量使用率超過 90%，增加休息時間
                    if usage_info and usage_info.get('percentage', 0) > 90:
                        logger.warning("⚠️ 流量使用率超過 90%，增加休息時間")
                        time.sleep(3)
                
                # 每支股票後休息一下，避免API限制
                time.sleep(0.75)
                
        except KeyboardInterrupt:
            logger.info("⏹️  收到中斷信號，正在停止...")
            raise
        
        except Exception as e:
            logger.error(f"❌ 分鐘線資料收集過程發生錯誤: {e}")
            raise
        
        finally:
            # 顯示最終流量使用狀況 (在登出前)
            if self.flow_monitor and self.api:
                logger.info("=" * 50)
                logger.info("📊 最終流量使用狀況")
                logger.info("=" * 50)
                try:
                    final_usage = self.flow_monitor.show_status()
                    if final_usage:
                        bytes_used_mb = final_usage['bytes_used'] / (1024 * 1024)
                        limit_mb = final_usage['limit_bytes'] / (1024 * 1024)
                        percentage = final_usage['percentage']
                        logger.info(f"🏁 流量使用總結: {bytes_used_mb:.1f}MB/{limit_mb:.1f}MB 已使用{percentage:.2f}%")
                    else:
                        logger.warning("⚠️ 無法獲取最終流量使用狀況")
                except Exception as e:
                    logger.warning(f"⚠️ 獲取最終流量狀況時發生錯誤: {e}")
                logger.info("=" * 50)
            
            # 登出API
            if self.api:
                try:
                    self.api.logout()
                    logger.info("🔓 Shioaji 已登出")
                except:
                    pass
            
            # 顯示收集結果統計
            self.show_minute_collection_summary(results)
    
    def show_minute_collection_summary(self, results: Dict):
        """顯示分鐘線收集結果統計"""
        logger.info("=" * 50)
        logger.info("📊 Shioaji 分鐘線資料收集完成統計")
        logger.info("=" * 50)
        
        total_new_records = 0
        success_count = 0
        skipped_count = 0
        failed_count = 0
        
        for symbol, result in results.items():
            if result > 0:
                total_new_records += result
                success_count += 1
                logger.info(f"✅ {symbol}: {result:6} 筆新5分鐘資料")
            elif result == 0:
                skipped_count += 1
                logger.info(f"⏭️ {symbol}: 已存在，略過")
            else:
                failed_count += 1
                logger.info(f"❌ {symbol}: 失敗")
        
        logger.info("=" * 50)
        logger.info(f"📈 新下載: {success_count} 支股票")
        logger.info(f"⏭️ 已存在: {skipped_count} 支股票")
        logger.info(f"❌ 失敗: {failed_count} 支股票")
        logger.info(f"📊 新增資料: {total_new_records} 筆5分鐘資料")
        logger.info("=" * 50)


def main():
    """主函數"""
    print("=" * 50)
    print("📈 Shioaji 分鐘線資料收集器 (智能版)")
    print("=" * 50)
    print("新功能：")
    print("• 防重複下載機制 - 自動檢查已存在資料")
    print("• 官方流量監控 - 使用 api.usage() 方法")
    print("• 智能警告與自動停止 (95% 流量限制)")
    print("• 流量節省登入 - fetch_contract=False")
    print("• 詳細的下載/略過狀態日誌")
    print("• 最終流量使用狀況報告")
    print("收集內容：")
    print("• 1分鐘K線資料 (從Shioaji API)")
    print("• 聚合為5分鐘K線資料")
    print("• 儲存到 candlesticks_min 資料表")
    print("• 日期範圍: 2020-03-02 ~ 2025-07-08 (正式模式)")
    print("• 測試模式: 2024-12-01 ~ 2024-12-31 (節省流量)")
    print("=" * 50)
    
    try:
        collector = ShioajiDataCollector()
        
        # 詢問用戶是否要測試模式
        choice = input("選擇模式 (1=測試模式收集3支股票, 2=完整模式收集180支股票): ").strip()
        
        if choice == "1":
            print("🧪 測試模式：收集前3支股票的分鐘線資料")
            collector.collect_all_minute_data(test_mode=True)
        elif choice == "2":
            print("🚀 完整模式：收集全部180支股票的分鐘線資料")
            collector.collect_all_minute_data(test_mode=False)
        else:
            print("🧪 預設測試模式：收集前3支股票的分鐘線資料")
            collector.collect_all_minute_data(test_mode=True)
        
        print("✅ Shioaji 分鐘線資料收集完成！")
        
    except Exception as e:
        logger.error(f"❌ 執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()