#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shioaji 分鐘線資料收集器 - 專門收集5分鐘K線資料
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
            
            # 登入 (使用位置參數)
            self.api.login(
                self.shioaji_config['user'],
                self.shioaji_config['pass']
            )
            
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
            from data_pipeline.fetch_minute import FlowMonitor
            self.flow_monitor = FlowMonitor()
            logger.info("📊 流量監控器已啟動")
        except Exception as e:
            logger.warning(f"⚠️  流量監控器啟動失敗: {e}")
    
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
        
        logger.info(f"準備收集 {len(all_stocks)} 支股票的分鐘線資料")
        return all_stocks
    
    def fetch_minute_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """獲取單一股票的分鐘線資料"""
        try:
            if not self.api:
                logger.error("❌ Shioaji API 未登入")
                return pd.DataFrame()
            
            logger.info(f"📡 請求分鐘線資料: {symbol} ({start_date} ~ {end_date})")
            
            # 取得股票合約
            contract = self.api.Contracts.Stocks[symbol]
            
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
                    
                    # 更新流量監控
                    if self.flow_monitor:
                        data_size = len(df) * len(df.columns) * 8  # 估算資料大小
                        self.flow_monitor.add_usage(data_size)
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
    
    def collect_stock_minute_data(self, symbol: str, start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
        """收集單一股票的分鐘線資料"""
        logger.info(f"🎯 開始收集 {symbol} 的分鐘線資料...")
        
        try:
            # 獲取1分鐘資料
            df_1min = self.fetch_minute_data(symbol, start_date, end_date)
            
            if not df_1min.empty:
                # 聚合為5分鐘
                df_5min = self.aggregate_to_5min(df_1min)
                
                if not df_5min.empty:
                    # 儲存到資料庫
                    self.insert_df('minute_bars', df_5min, if_exists='append')
                    logger.info(f"💾 成功儲存 {len(df_5min)} 筆5分鐘資料: {symbol}")
                    return len(df_5min)
                else:
                    logger.warning(f"⚠️  {symbol} 聚合5分鐘資料失敗")
                    return 0
            else:
                logger.warning(f"⚠️  {symbol} 無1分鐘資料")
                return 0
                
        except Exception as e:
            logger.error(f"❌ 處理 {symbol} 分鐘線資料失敗: {e}")
            return -1
    
    def collect_all_minute_data(self, symbols: Optional[List[str]] = None,
                               start_date: str = "2024-01-01",
                               end_date: str = "2024-12-31",
                               test_mode: bool = True):
        """收集所有股票的分鐘線資料"""
        if symbols is None:
            symbols = self.get_stock_list()
        
        if test_mode:
            # 測試模式：只收集前3支股票
            symbols = symbols[:3]
            logger.info(f"🧪 測試模式：收集 {symbols} 的分鐘線資料")
        
        logger.info(f"📊 開始收集 {len(symbols)} 支股票的分鐘線資料...")
        logger.info(f"📅 日期範圍: {start_date} ~ {end_date}")
        
        # 登入API
        if not self.login_shioaji():
            logger.error("❌ 無法登入 Shioaji，停止執行")
            return
        
        # 設定流量監控
        self.setup_flow_monitor()
        
        results = {}
        
        try:
            for i, symbol in enumerate(symbols):
                logger.info(f"📈 處理進度: {i+1}/{len(symbols)} - {symbol}")
                
                # 收集該股票的分鐘線資料
                result = self.collect_stock_minute_data(symbol, start_date, end_date)
                results[symbol] = result
                
                # 每支股票後休息一下，避免API限制
                time.sleep(2)
                
                # 每5支股票顯示流量狀況
                if (i + 1) % 5 == 0 and self.flow_monitor:
                    self.flow_monitor.show_status()
                
        except KeyboardInterrupt:
            logger.info("⏹️  收到中斷信號，正在停止...")
            raise
        
        except Exception as e:
            logger.error(f"❌ 分鐘線資料收集過程發生錯誤: {e}")
            raise
        
        finally:
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
        
        total_records = 0
        success_count = 0
        failed_count = 0
        
        for symbol, result in results.items():
            if result > 0:
                total_records += result
                success_count += 1
                logger.info(f"✅ {symbol}: {result:6} 筆5分鐘資料")
            elif result == 0:
                logger.info(f"⚠️  {symbol}: 無資料")
            else:
                failed_count += 1
                logger.info(f"❌ {symbol}: 失敗")
        
        logger.info("=" * 50)
        logger.info(f"📈 成功: {success_count} 支股票")
        logger.info(f"❌ 失敗: {failed_count} 支股票")
        logger.info(f"📊 總計: {total_records} 筆5分鐘資料")
        logger.info("=" * 50)


def main():
    """主函數"""
    print("=" * 50)
    print("📈 Shioaji 分鐘線資料收集器")
    print("=" * 50)
    print("收集內容：")
    print("• 1分鐘K線資料 (從Shioaji API)")
    print("• 聚合為5分鐘K線資料")
    print("• 儲存到 minute_bars 資料表")
    print("• 包含流量監控功能")
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