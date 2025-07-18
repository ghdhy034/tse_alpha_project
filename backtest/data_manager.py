# backtest/data_manager.py
"""
資料管理器 - 實作 References.txt 建議
包含 cache_mode 和 preload 功能
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, datetime
import pandas as pd
import numpy as np
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .config import BacktestConfig, Period
    from .time_manager import TimeSeriesManager
except ImportError:
    from config import BacktestConfig, Period
    from time_manager import TimeSeriesManager

logger = logging.getLogger(__name__)


class DataManager:
    """
    資料管理器 - 實作 References.txt 建議
    支援 cache_mode='arrow'|'duckdb'|'none' 和 preload 功能
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cache_mode = config.cache_mode
        self.preload_enabled = config.preload_data
        
        # 快取儲存
        self._memory_cache: Dict[str, Any] = {}
        self._cache_stats = {'hits': 0, 'misses': 0, 'preloads': 0}
        
        # 資料庫連接
        self._db_connection = None
        self._setup_database_connection()
        
        # 預載入狀態
        self._preloaded_symbols: set = set()
        self._preload_date_range: Optional[Tuple[date, date]] = None
        
        logger.info(f"資料管理器初始化: cache_mode={self.cache_mode}, preload={self.preload_enabled}")
    
    def _setup_database_connection(self):
        """設置資料庫連接"""
        try:
            if self.cache_mode == 'duckdb':
                import duckdb
                # 使用記憶體資料庫進行快取
                self._db_connection = duckdb.connect(':memory:')
                logger.info("DuckDB 快取連接建立成功")
            elif self.cache_mode == 'arrow':
                import pyarrow as pa
                logger.info("Arrow 快取模式啟用")
            else:
                logger.info("無快取模式")
        except ImportError as e:
            logger.warning(f"快取依賴套件未安裝: {e}，回退到無快取模式")
            self.cache_mode = 'none'
    
    def preload(self, symbols: List[str], date_range: Tuple[date, date]) -> bool:
        """
        預載入資料 - References.txt 建議
        warm-up parquet→memmap
        
        Args:
            symbols: 股票代碼列表
            date_range: 日期範圍 (start_date, end_date)
            
        Returns:
            是否成功預載入
        """
        if not self.preload_enabled:
            logger.info("預載入功能已停用")
            return True
        
        start_date, end_date = date_range
        logger.info(f"開始預載入: {len(symbols)} 檔股票，日期範圍 {start_date} ~ {end_date}")
        
        start_time = time.time()
        success_count = 0
        
        try:
            # 並行預載入
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for symbol in symbols:
                    future = executor.submit(self._preload_symbol_data, symbol, start_date, end_date)
                    futures.append((symbol, future))
                
                # 收集結果
                for symbol, future in futures:
                    try:
                        if future.result():
                            success_count += 1
                            self._preloaded_symbols.add(symbol)
                    except Exception as e:
                        logger.error(f"預載入 {symbol} 失敗: {e}")
            
            # 更新預載入狀態
            self._preload_date_range = date_range
            self._cache_stats['preloads'] += success_count
            
            elapsed = time.time() - start_time
            logger.info(f"預載入完成: {success_count}/{len(symbols)} 檔股票，耗時 {elapsed:.2f} 秒")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"預載入失敗: {e}")
            return False
    
    def _preload_symbol_data(self, symbol: str, start_date: date, end_date: date) -> bool:
        """預載入單檔股票資料"""
        try:
            # 載入日線資料
            daily_data = self._load_daily_data(symbol, start_date, end_date)
            if daily_data is not None and not daily_data.empty:
                cache_key = f"daily_{symbol}_{start_date}_{end_date}"
                self._store_in_cache(cache_key, daily_data)
            
            # 載入分鐘線資料 (如果需要)
            minute_data = self._load_minute_data(symbol, start_date, end_date)
            if minute_data is not None and not minute_data.empty:
                cache_key = f"minute_{symbol}_{start_date}_{end_date}"
                self._store_in_cache(cache_key, minute_data)
            
            # 載入籌碼面資料
            chip_data = self._load_chip_data(symbol, start_date, end_date)
            if chip_data is not None and not chip_data.empty:
                cache_key = f"chip_{symbol}_{start_date}_{end_date}"
                self._store_in_cache(cache_key, chip_data)
            
            return True
            
        except Exception as e:
            logger.error(f"預載入 {symbol} 資料失敗: {e}")
            return False
    
    def get_stock_data(self, 
                      symbol: str, 
                      start_date: date, 
                      end_date: date,
                      data_type: str = 'daily') -> Optional[pd.DataFrame]:
        """
        獲取股票資料 - 支援快取
        
        Args:
            symbol: 股票代碼
            start_date: 開始日期
            end_date: 結束日期
            data_type: 資料類型 ('daily', 'minute', 'chip')
            
        Returns:
            股票資料 DataFrame
        """
        cache_key = f"{data_type}_{symbol}_{start_date}_{end_date}"
        
        # 嘗試從快取獲取
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            self._cache_stats['hits'] += 1
            logger.debug(f"快取命中: {cache_key}")
            return cached_data
        
        # 快取未命中，從資料庫載入
        self._cache_stats['misses'] += 1
        logger.debug(f"快取未命中: {cache_key}")
        
        try:
            if data_type == 'daily':
                data = self._load_daily_data(symbol, start_date, end_date)
            elif data_type == 'minute':
                data = self._load_minute_data(symbol, start_date, end_date)
            elif data_type == 'chip':
                data = self._load_chip_data(symbol, start_date, end_date)
            else:
                logger.error(f"不支援的資料類型: {data_type}")
                return None
            
            # 儲存到快取
            if data is not None and not data.empty:
                self._store_in_cache(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"載入 {symbol} {data_type} 資料失敗: {e}")
            return None
    
    def _load_daily_data(self, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """載入日線資料"""
        try:
            from market_data_collector.utils.db import query_df
            
            query = """
            SELECT date, open, high, low, close, volume
            FROM candlesticks_daily 
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
            """
            
            df = query_df(query, (symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date']).dt.date
                df = df.set_index('date')
                return df
            else:
                logger.warning(f"無 {symbol} 日線資料: {start_date} ~ {end_date}")
                return None
                
        except Exception as e:
            logger.error(f"載入 {symbol} 日線資料失敗: {e}")
            return None
    
    def _load_minute_data(self, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """載入分鐘線資料"""
        try:
            from market_data_collector.utils.db import query_df
            
            # 修正欄位名稱：使用 timestamp 而不是 datetime
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candlesticks_min 
            WHERE symbol = ? AND DATE(timestamp) BETWEEN ? AND ?
            ORDER BY timestamp ASC
            """
            
            df = query_df(query, (symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                return df
            else:
                logger.debug(f"無 {symbol} 分鐘線資料: {start_date} ~ {end_date}")
                return None
                
        except Exception as e:
            logger.debug(f"載入 {symbol} 分鐘線資料失敗: {e}")
            return None
    
    def _load_chip_data(self, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """載入籌碼面資料"""
        try:
            from market_data_collector.utils.db import query_df
            
            # 根據 db_structure.json，使用正確的欄位名稱
            try:
                # 載入融資融券資料 - 使用實際的欄位名稱
                margin_query = """
                SELECT date, MarginPurchaseBuy, MarginPurchaseSell, ShortSaleSell, ShortSaleBuy
                FROM margin_purchase_shortsale 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date ASC
                """
                
                margin_df = query_df(margin_query, (symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            except Exception as e:
                logger.debug(f"融資融券資料查詢失敗: {e}")
                margin_df = pd.DataFrame()
            
            # 載入法人進出資料
            try:
                institutional_query = """
                SELECT date, Foreign_Investor_buy, Foreign_Investor_sell, 
                       Investment_Trust_buy, Investment_Trust_sell, 
                       Dealer_self_buy, Dealer_self_sell
                FROM institutional_investors_buy_sell 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date ASC
                """
                
                inst_df = query_df(institutional_query, (symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            except Exception as e:
                logger.debug(f"法人進出資料查詢失敗: {e}")
                inst_df = pd.DataFrame()
            
            # 合併資料
            if not margin_df.empty or not inst_df.empty:
                if not margin_df.empty:
                    margin_df['date'] = pd.to_datetime(margin_df['date']).dt.date
                if not inst_df.empty:
                    inst_df['date'] = pd.to_datetime(inst_df['date']).dt.date
                
                if not margin_df.empty and not inst_df.empty:
                    chip_df = pd.merge(margin_df, inst_df, on='date', how='outer')
                elif not margin_df.empty:
                    chip_df = margin_df
                else:
                    chip_df = inst_df
                
                chip_df = chip_df.set_index('date').sort_index()
                return chip_df
            else:
                logger.debug(f"無 {symbol} 籌碼面資料: {start_date} ~ {end_date}")
                return None
                
        except Exception as e:
            logger.debug(f"載入 {symbol} 籌碼面資料失敗: {e}")
            return None
    
    def _store_in_cache(self, key: str, data: pd.DataFrame):
        """儲存資料到快取"""
        try:
            if self.cache_mode == 'none':
                return
            elif self.cache_mode == 'duckdb':
                self._store_in_duckdb_cache(key, data)
            elif self.cache_mode == 'arrow':
                self._store_in_arrow_cache(key, data)
            else:
                # 預設使用記憶體快取
                self._memory_cache[key] = data.copy()
                
        except Exception as e:
            logger.warning(f"快取儲存失敗 {key}: {e}")
    
    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """從快取獲取資料"""
        try:
            if self.cache_mode == 'none':
                return None
            elif self.cache_mode == 'duckdb':
                return self._get_from_duckdb_cache(key)
            elif self.cache_mode == 'arrow':
                return self._get_from_arrow_cache(key)
            else:
                # 預設使用記憶體快取
                return self._memory_cache.get(key)
                
        except Exception as e:
            logger.warning(f"快取讀取失敗 {key}: {e}")
            return None
    
    def _store_in_duckdb_cache(self, key: str, data: pd.DataFrame):
        """儲存到 DuckDB 快取"""
        if self._db_connection is None:
            return
        
        table_name = f"cache_{key.replace('-', '_').replace('.', '_')}"
        self._db_connection.execute(f"DROP TABLE IF EXISTS {table_name}")
        self._db_connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM data", {"data": data})
    
    def _get_from_duckdb_cache(self, key: str) -> Optional[pd.DataFrame]:
        """從 DuckDB 快取獲取"""
        if self._db_connection is None:
            return None
        
        table_name = f"cache_{key.replace('-', '_').replace('.', '_')}"
        try:
            result = self._db_connection.execute(f"SELECT * FROM {table_name}").fetchdf()
            return result
        except:
            return None
    
    def _store_in_arrow_cache(self, key: str, data: pd.DataFrame):
        """儲存到 Arrow 快取"""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            cache_dir = Path("cache/arrow")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = cache_dir / f"{key}.parquet"
            table = pa.Table.from_pandas(data)
            pq.write_table(table, file_path)
            
        except ImportError:
            logger.warning("PyArrow 未安裝，無法使用 Arrow 快取")
    
    def _get_from_arrow_cache(self, key: str) -> Optional[pd.DataFrame]:
        """從 Arrow 快取獲取"""
        try:
            import pyarrow.parquet as pq
            
            cache_dir = Path("cache/arrow")
            file_path = cache_dir / f"{key}.parquet"
            
            if file_path.exists():
                table = pq.read_table(file_path)
                return table.to_pandas()
            else:
                return None
                
        except ImportError:
            logger.warning("PyArrow 未安裝，無法使用 Arrow 快取")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """獲取快取統計"""
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = self._cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'cache_mode': self.cache_mode,
            'hits': self._cache_stats['hits'],
            'misses': self._cache_stats['misses'],
            'hit_rate': hit_rate,
            'preloads': self._cache_stats['preloads'],
            'preloaded_symbols': len(self._preloaded_symbols),
            'memory_cache_size': len(self._memory_cache)
        }
    
    def clear_cache(self):
        """清空快取"""
        self._memory_cache.clear()
        self._preloaded_symbols.clear()
        self._cache_stats = {'hits': 0, 'misses': 0, 'preloads': 0}
        
        if self.cache_mode == 'duckdb' and self._db_connection:
            # 清空 DuckDB 快取表
            try:
                tables = self._db_connection.execute("SHOW TABLES").fetchall()
                for table in tables:
                    if table[0].startswith('cache_'):
                        self._db_connection.execute(f"DROP TABLE {table[0]}")
            except:
                pass
        
        logger.info("快取已清空")


def test_data_manager():
    """測試資料管理器"""
    print("=== 測試資料管理器 ===")
    
    # 創建測試配置
    from config import create_smoke_test_config
    config = create_smoke_test_config()
    config.cache_mode = 'none'  # 測試時不使用快取
    config.preload_data = True
    
    # 創建資料管理器
    print("1. 創建資料管理器...")
    try:
        data_manager = DataManager(config)
        print(f"   ✅ 資料管理器創建成功")
        print(f"   快取模式: {data_manager.cache_mode}")
    except Exception as e:
        print(f"   ❌ 資料管理器創建失敗: {e}")
        return False
    
    # 測試資料載入
    print("2. 測試資料載入...")
    try:
        from datetime import date
        
        test_symbol = '2330'
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        # 測試日線資料
        daily_data = data_manager.get_stock_data(test_symbol, start_date, end_date, 'daily')
        if daily_data is not None:
            print(f"   ✅ 日線資料載入成功: {len(daily_data)} 筆記錄")
        else:
            print(f"   ⚠️  無日線資料或載入失敗")
        
        # 測試籌碼面資料
        chip_data = data_manager.get_stock_data(test_symbol, start_date, end_date, 'chip')
        if chip_data is not None:
            print(f"   ✅ 籌碼面資料載入成功: {len(chip_data)} 筆記錄")
        else:
            print(f"   ⚠️  無籌碼面資料或載入失敗")
            
    except Exception as e:
        print(f"   ❌ 資料載入測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 測試預載入功能
    print("3. 測試預載入功能...")
    try:
        symbols = ['2330', '2317']
        date_range = (date(2024, 1, 1), date(2024, 1, 31))
        
        success = data_manager.preload(symbols, date_range)
        if success:
            print(f"   ✅ 預載入成功")
        else:
            print(f"   ⚠️  預載入失敗或無資料")
            
    except Exception as e:
        print(f"   ❌ 預載入測試失敗: {e}")
        return False
    
    # 測試快取統計
    print("4. 測試快取統計...")
    try:
        stats = data_manager.get_cache_stats()
        print("   快取統計:")
        for key, value in stats.items():
            print(f"     {key}: {value}")
        print("   ✅ 快取統計獲取成功")
    except Exception as e:
        print(f"   ❌ 快取統計測試失敗: {e}")
        return False
    
    print("✅ 資料管理器測試完成")
    return True


if __name__ == "__main__":
    test_data_manager()