# data_pipeline/fetch_minute.py
"""
分鐘線資料下載器 - 支援 FinMind、Shioaji 和代理資料生成
根據日期自動路由到適當的資料源：
- < 2019-05-29: 生成代理 VWAP 資料
- 2019-05-29 ~ 2020-03-01: 使用 FinMind API
- >= 2020-03-02: 使用 Shioaji API
"""
from __future__ import annotations
import sys
import os
import asyncio
import time
import logging
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import requests
from pathlib import Path

# 添加 market_data_collector 到路徑
sys.path.append(str(Path(__file__).parent.parent / "market_data_collector"))

try:
    from market_data_collector.utils.config import (
        TOKEN, API_ENDPOINT, 
        SHIOAJI_USER, SHIOAJI_PASS, SHIOAJI_CA_PATH, SHIOAJI_CA_PASS,
        MINUTE_START_DATE
    )
    from market_data_collector.utils.db import get_conn, insert_df, query_df
except ImportError as e:
    print(f"警告: 無法導入配置或資料庫模組: {e}")
    print("請確保 market_data_collector 模組在正確路徑")
    # 提供預設值以便測試
    TOKEN = "dummy_token"
    API_ENDPOINT = "https://api.finmindtrade.com/api/v4/data"
    SHIOAJI_USER = "dummy_user"
    SHIOAJI_PASS = "dummy_pass"
    SHIOAJI_CA_PATH = "dummy_path"
    SHIOAJI_CA_PASS = "dummy_pass"
    MINUTE_START_DATE = "2019-05-29"

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 日期路由邊界 (統一起始日期)
UNIFIED_START_DATE = date(2020, 3, 2)  # 所有資料統一起始日期

# API 限流設定
FINMIND_RATE_LIMIT = 200  # requests per minute
FINMIND_REQUEST_INTERVAL = 60 / FINMIND_RATE_LIMIT  # seconds between requests


class DataRouter:
    """資料源路由器 - 根據日期決定使用哪個資料源"""
    
    @staticmethod
    def route(symbol: str, target_date: date) -> str:
        """
        根據日期路由到適當的資料源 (統一起始日期版)
        
        Args:
            symbol: 股票代號
            target_date: 目標日期
            
        Returns:
            'shioaji' | 'no_data'
        """
        if target_date < UNIFIED_START_DATE:
            return 'no_data'  # 2020-03-02 之前無分鐘線資料
        else:
            return 'shioaji'  # 統一使用 Shioaji


class FinMindDownloader:
    """FinMind API 下載器"""
    
    def __init__(self):
        self.last_request_time = 0
        self.request_count = 0
        self.minute_reset_time = time.time()
    
    def _rate_limit(self):
        """實施速率限制"""
        current_time = time.time()
        
        # 每分鐘重置計數器
        if current_time - self.minute_reset_time >= 60:
            self.request_count = 0
            self.minute_reset_time = current_time
        
        # 檢查是否超過限制
        if self.request_count >= FINMIND_RATE_LIMIT:
            sleep_time = 60 - (current_time - self.minute_reset_time)
            if sleep_time > 0:
                logger.info(f"達到 FinMind 速率限制，等待 {sleep_time:.1f} 秒")
                time.sleep(sleep_time)
                self.request_count = 0
                self.minute_reset_time = time.time()
        
        # 請求間隔控制
        time_since_last = current_time - self.last_request_time
        if time_since_last < FINMIND_REQUEST_INTERVAL:
            sleep_time = FINMIND_REQUEST_INTERVAL - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def download_minute_data(self, symbol: str, target_date: date, retries: int = 3) -> pd.DataFrame:
        """
        從 FinMind 下載 1 分鐘線資料
        
        Args:
            symbol: 股票代號
            target_date: 目標日期
            retries: 重試次數
            
        Returns:
            包含 1 分鐘 OHLCV 資料的 DataFrame
        """
        date_str = target_date.strftime("%Y-%m-%d")
        
        for attempt in range(retries):
            try:
                self._rate_limit()
                
                params = {
                    "dataset": "TaiwanStockMinuteData",
                    "data_id": symbol,
                    "start_date": date_str,
                    "end_date": date_str,
                    "token": TOKEN
                }
                
                logger.info(f"下載 {symbol} {date_str} 的分鐘線資料 (嘗試 {attempt + 1}/{retries})")
                response = requests.get(API_ENDPOINT, params=params, timeout=30)
                
                if response.status_code == 200:
                    json_data = response.json()
                    if json_data.get("status") == 200 and "data" in json_data:
                        data = json_data["data"]
                        if data:  # 檢查是否有資料
                            df = pd.DataFrame(data)
                            df['datetime'] = pd.to_datetime(df['datetime'])
                            df = df.sort_values('datetime').reset_index(drop=True)
                            logger.info(f"成功下載 {len(df)} 筆 {symbol} 的分鐘線資料")
                            return df
                        else:
                            logger.warning(f"{symbol} {date_str} 無分鐘線資料")
                            return pd.DataFrame()
                    else:
                        logger.warning(f"FinMind API 回傳錯誤: {json_data}")
                        if attempt < retries - 1:
                            time.sleep(2 ** attempt)  # 指數退避
                            continue
                else:
                    logger.warning(f"HTTP 錯誤 {response.status_code}: {response.text}")
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                        
            except Exception as e:
                logger.error(f"下載失敗 {symbol} {date_str}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
        logger.error(f"所有重試失敗: {symbol} {date_str}")
        return pd.DataFrame()


class ShioajiDownloader:
    """Shioaji API 下載器"""
    
    def __init__(self):
        self.api = None
        self.login_time = None
        self.session_timeout = 3600  # 1 小時
    
    def _check_api_usage(self):
        """檢查 API 流量使用狀況"""
        try:
            usage = self.api.usage()
            
            # 已使用流量
            used_bytes = getattr(usage, 'bytes', 0)
            used_mb = used_bytes / (1024 * 1024)
            
            # 流量上限 (預設500MB如果無法取得)
            limit_bytes = getattr(usage, 'limit', 500 * 1024 * 1024)
            limit_mb = limit_bytes / (1024 * 1024)
            
            # 剩餘流量
            remaining_bytes = getattr(usage, 'remaining', limit_bytes - used_bytes)
            remaining_mb = remaining_bytes / (1024 * 1024)
            
            # 使用率
            usage_percent = (used_bytes / limit_bytes) * 100 if limit_bytes > 0 else 0
            
            logger.info(f"📊 API 流量: {used_mb:.1f}/{limit_mb:.1f} MB ({usage_percent:.1f}%) | 剩餘: {remaining_mb:.1f} MB")
            
            # 警告檢查
            if usage_percent >= 95:
                logger.warning("🚨 嚴重警告: API 流量使用率超過 95%，即將達到上限！")
            elif usage_percent >= 90:
                logger.warning("⚠️  警告: API 流量使用率超過 90%")
            elif usage_percent >= 80:
                logger.warning("⚠️  注意: API 流量使用率超過 80%")
            
            return {
                'used_mb': used_mb,
                'limit_mb': limit_mb,
                'remaining_mb': remaining_mb,
                'usage_percent': usage_percent
            }
            
        except Exception as e:
            logger.warning(f"⚠️  無法檢查 API 流量: {e}")
            return None
    
    def _ensure_login(self) -> bool:
        """確保 Shioaji 登入狀態"""
        try:
            import shioaji as sj
        except ImportError:
            logger.error("Shioaji 未安裝，請執行: pip install shioaji")
            return False
        
        current_time = time.time()
        
        # 檢查是否需要重新登入
        if (self.api is None or 
            self.login_time is None or 
            current_time - self.login_time > self.session_timeout):
            
            try:
                logger.info("正在登入 Shioaji...")
                self.api = sj.Shioaji()
                
                # 嘗試多種登入方式
                try:
                    # 方式 1: 位置參數登入 (根據用戶範例)
                    accounts = self.api.login(SHIOAJI_USER.strip(), SHIOAJI_PASS.strip())
                    logger.info("Shioaji 位置參數登入成功")
                except Exception as e1:
                    logger.warning(f"位置參數登入失敗: {e1}")
                    try:
                        # 方式 2: 關鍵字參數登入
                        accounts = self.api.login(
                            api_key=SHIOAJI_USER.strip(),
                            secret_key=SHIOAJI_PASS.strip(),
                            contracts_cb=lambda security_type: None
                        )
                        logger.info("Shioaji 關鍵字參數登入成功")
                    except Exception as e2:
                        logger.error(f"所有登入方式都失敗: 位置參數={e1}, 關鍵字參數={e2}")
                        raise e2
                
                self.login_time = current_time
                logger.info("Shioaji 登入成功")
                return True
                
            except Exception as e:
                logger.error(f"Shioaji 登入失敗: {e}")
                return False
        
        return True
    
    def download_minute_data(self, symbol: str, target_date: date) -> pd.DataFrame:
        """
        從 Shioaji 下載 1 分鐘線資料 (修復版)
        
        Args:
            symbol: 股票代號
            target_date: 目標日期
            
        Returns:
            包含 1 分鐘 OHLCV 資料的 DataFrame
        """
        if not self._ensure_login():
            return pd.DataFrame()
        
        try:
            # 構建合約
            contract = self.api.Contracts.Stocks[symbol]
            
            # 使用字串格式的日期 (修復關鍵)
            date_str = target_date.strftime("%Y-%m-%d")
            
            logger.info(f"下載 {symbol} {target_date} 的 Shioaji 分鐘線資料")
            
            # 檢查流量
            self._check_api_usage()
            
            # 下載 K 線資料 - 使用正確的日期格式
            kbars = self.api.kbars(
                contract=contract,
                start=date_str,    # 字串格式: "2024-12-16"
                end=date_str       # 字串格式: "2024-12-16"
            )
            
            # 修復: 正確處理 Kbars 物件
            df = pd.DataFrame({**kbars})
            
            if df.empty:
                logger.warning(f"{symbol} {target_date} 無 Shioaji 分鐘線資料")
                return pd.DataFrame()
            
            # 重新命名欄位以符合標準格式
            df = df.rename(columns={
                'ts': 'datetime',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 轉換時間格式
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            logger.info(f"成功下載 {len(df)} 筆 {symbol} 的 Shioaji 分鐘線資料")
            return df
            
        except Exception as e:
            logger.error(f"Shioaji 下載失敗 {symbol} {target_date}: {e}")
            return pd.DataFrame()


class ProxyDataGenerator:
    """代理資料生成器 - 為早期日期生成近似的 5 分鐘 VWAP 資料"""
    
    def generate_proxy_data(self, symbol: str, target_date: date) -> pd.DataFrame:
        """
        生成代理 5 分鐘 VWAP 資料
        
        Args:
            symbol: 股票代號
            target_date: 目標日期
            
        Returns:
            包含代理 5 分鐘資料的 DataFrame
        """
        try:
            # 獲取次日開盤價作為基準
            next_day = target_date + timedelta(days=1)
            base_price = self._get_next_day_open(symbol, next_day)
            
            if base_price is None:
                logger.warning(f"無法獲取 {symbol} {next_day} 的開盤價，跳過代理資料生成")
                return pd.DataFrame()
            
            # 生成交易時段的時間點 (09:00-13:30, 每 5 分鐘)
            trading_times = []
            current_time = datetime.combine(target_date, datetime.min.time().replace(hour=9))
            end_time = datetime.combine(target_date, datetime.min.time().replace(hour=13, minute=30))
            
            while current_time <= end_time:
                trading_times.append(current_time)
                current_time += timedelta(minutes=5)
            
            # 生成代理資料
            proxy_data = []
            tick_size = self._get_tick_size(base_price)
            
            for ts in trading_times:
                # 在開盤價附近生成隨機變動（±0.5 tick）
                price_variation = np.random.uniform(-0.5 * tick_size, 0.5 * tick_size)
                proxy_price = base_price + price_variation
                
                # 生成 OHLCV（簡化版本，所有價格都相同）
                proxy_data.append({
                    'datetime': ts,
                    'open': proxy_price,
                    'high': proxy_price,
                    'low': proxy_price,
                    'close': proxy_price,
                    'volume': np.random.randint(1000, 10000),  # 隨機成交量
                    'vwap': proxy_price
                })
            
            df = pd.DataFrame(proxy_data)
            logger.info(f"生成 {len(df)} 筆 {symbol} {target_date} 的代理資料")
            return df
            
        except Exception as e:
            logger.error(f"代理資料生成失敗 {symbol} {target_date}: {e}")
            return pd.DataFrame()
    
    def _get_next_day_open(self, symbol: str, next_date: date) -> Optional[float]:
        """獲取次日開盤價"""
        try:
            date_str = next_date.strftime("%Y-%m-%d")
            query = """
            SELECT open FROM candlesticks_daily 
            WHERE symbol = ? AND date = ?
            """
            df = query_df(query, (symbol, date_str))
            
            if not df.empty and pd.notna(df.iloc[0]['open']):
                return float(df.iloc[0]['open'])
            
            # 如果當日無資料，嘗試找最近的資料
            query = """
            SELECT open FROM candlesticks_daily 
            WHERE symbol = ? AND date >= ? 
            ORDER BY date ASC LIMIT 1
            """
            df = query_df(query, (symbol, date_str))
            
            if not df.empty and pd.notna(df.iloc[0]['open']):
                return float(df.iloc[0]['open'])
                
        except Exception as e:
            logger.error(f"獲取次日開盤價失敗 {symbol} {next_date}: {e}")
        
        return None
    
    def _get_tick_size(self, price: float) -> float:
        """根據價格計算最小跳動單位"""
        if price < 10:
            return 0.01
        elif price < 50:
            return 0.05
        elif price < 100:
            return 0.1
        elif price < 500:
            return 0.5
        elif price < 1000:
            return 1.0
        else:
            return 5.0


class MinuteBarAggregator:
    """1 分鐘 → 5 分鐘聚合器"""
    
    @staticmethod
    def to_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        將 1 分鐘資料聚合為 5 分鐘資料
        
        Args:
            df_1m: 1 分鐘 OHLCV DataFrame
            
        Returns:
            5 分鐘 OHLCV + VWAP DataFrame
        """
        if df_1m.empty:
            return pd.DataFrame()
        
        try:
            # 設定時間索引
            df = df_1m.copy()
            df.set_index('datetime', inplace=True)
            
            # 5 分鐘聚合
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            df_5m = df.resample('5min').agg(agg_rules)
            
            # 計算 VWAP (Volume Weighted Average Price)
            df_1m_indexed = df_1m.set_index('datetime')
            
            vwap_list = []
            for timestamp in df_5m.index:
                # 找到該 5 分鐘區間內的所有 1 分鐘資料
                start_time = timestamp
                end_time = timestamp + timedelta(minutes=5)
                
                mask = (df_1m_indexed.index >= start_time) & (df_1m_indexed.index < end_time)
                interval_data = df_1m_indexed[mask]
                
                if not interval_data.empty and interval_data['volume'].sum() > 0:
                    # VWAP = Σ(price × volume) / Σ(volume)
                    vwap = (interval_data['close'] * interval_data['volume']).sum() / interval_data['volume'].sum()
                    vwap_list.append(vwap)
                else:
                    # 如果無成交量，使用收盤價
                    vwap_list.append(interval_data['close'].iloc[-1] if not interval_data.empty else np.nan)
            
            df_5m['vwap'] = vwap_list
            
            # 移除無資料的行
            df_5m = df_5m.dropna()
            
            # 重置索引
            df_5m.reset_index(inplace=True)
            df_5m.rename(columns={'datetime': 'ts'}, inplace=True)
            
            logger.info(f"聚合完成: {len(df_1m)} 筆 1 分鐘 → {len(df_5m)} 筆 5 分鐘資料")
            return df_5m
            
        except Exception as e:
            logger.error(f"聚合失敗: {e}")
            return pd.DataFrame()


def fetch_symbol_date(symbol: str, target_date: date) -> pd.DataFrame:
    """
    下載指定股票和日期的 5 分鐘線資料
    
    Args:
        symbol: 股票代號
        target_date: 目標日期
        
    Returns:
        包含 5 分鐘 OHLCV + VWAP 資料的 DataFrame
    """
    # 路由到適當的資料源
    source = DataRouter.route(symbol, target_date)
    logger.info(f"處理 {symbol} {target_date}: 使用 {source} 資料源")
    
    df_1m = pd.DataFrame()
    
    try:
        if source == 'no_data':
            # 2020-03-02 之前無分鐘線資料
            logger.warning(f"{symbol} {target_date} 早於統一起始日期 {UNIFIED_START_DATE}，無分鐘線資料")
            return pd.DataFrame()
                
        elif source == 'shioaji':
            # Shioaji 下載 1 分鐘資料
            downloader = ShioajiDownloader()
            df_1m = downloader.download_minute_data(symbol, target_date)
            
            if not df_1m.empty:
                # 聚合為 5 分鐘
                aggregator = MinuteBarAggregator()
                df_5m = aggregator.to_5min(df_1m)
            else:
                df_5m = pd.DataFrame()
        
        else:
            logger.error(f"未知的資料源: {source}")
            return pd.DataFrame()
        
        # 添加 symbol 欄位
        if not df_5m.empty:
            df_5m['symbol'] = symbol
            
            # 確保欄位順序
            column_order = ['symbol', 'ts', 'open', 'high', 'low', 'close', 'volume', 'vwap']
            df_5m = df_5m.reindex(columns=column_order)
        
        return df_5m
        
    except Exception as e:
        logger.error(f"下載失敗 {symbol} {target_date}: {e}")
        return pd.DataFrame()


def store_minute_bars(df: pd.DataFrame) -> None:
    """
    將 5 分鐘線資料存入 minute_bars 資料表
    
    Args:
        df: 包含 5 分鐘線資料的 DataFrame
    """
    if df.empty:
        logger.warning("DataFrame 為空，跳過存儲")
        return
    
    try:
        # 使用 insert_df 進行 idempotent 插入
        insert_df('minute_bars', df, if_exists='append')
        logger.info(f"成功存入 {len(df)} 筆 minute_bars 資料")
        
    except Exception as e:
        logger.error(f"存儲 minute_bars 失敗: {e}")
        raise


def main():
    """主函數 - 命令列介面"""
    import argparse
    
    parser = argparse.ArgumentParser(description='下載股票分鐘線資料')
    parser.add_argument('--date', required=True, help='目標日期 (YYYY-MM-DD)')
    parser.add_argument('--symbols', required=True, nargs='+', help='股票代號列表')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細輸出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        symbols = args.symbols
        
        logger.info(f"開始下載 {target_date} 的分鐘線資料，股票: {symbols}")
        
        total_rows = 0
        for symbol in symbols:
            logger.info(f"處理股票: {symbol}")
            
            df = fetch_symbol_date(symbol, target_date)
            
            if not df.empty:
                store_minute_bars(df)
                total_rows += len(df)
                logger.info(f"{symbol} 完成，下載 {len(df)} 筆資料")
            else:
                logger.warning(f"{symbol} 無資料")
        
        logger.info(f"全部完成！總共下載 {total_rows} 筆資料")
        
    except Exception as e:
        logger.error(f"主程序失敗: {e}")
        raise


if __name__ == "__main__":
    main()