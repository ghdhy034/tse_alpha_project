# utils/db.py
"""
資料庫抽象層 - 提供 SQLite 與 DuckDB 的透明切換
根據 config.py 中的 USE_DUCKDB 設定自動選擇資料庫引擎
"""
from __future__ import annotations
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional, Union
import pandas as pd

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

from .config import USE_DUCKDB, DB_PATH, DUCKDB_PATH


class DatabaseConnection:
    """資料庫連線抽象類別"""
    
    def __init__(self, use_duckdb: bool = False):
        self.use_duckdb = use_duckdb and DUCKDB_AVAILABLE
        self._conn = None
        
    def __enter__(self):
        return self.get_connection()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def get_connection(self):
        """取得資料庫連線"""
        if self._conn is None:
            if self.use_duckdb:
                if not DUCKDB_AVAILABLE:
                    raise ImportError("DuckDB not available. Please install: pip install duckdb")
                
                # 確保 DuckDB 檔案目錄存在
                duckdb_dir = os.path.dirname(DUCKDB_PATH)
                if not os.path.exists(duckdb_dir):
                    os.makedirs(duckdb_dir)
                
                self._conn = duckdb.connect(str(DUCKDB_PATH))
                print(f"[db] 使用 DuckDB: {DUCKDB_PATH}")
            else:
                # 確保 SQLite 檔案目錄存在
                sqlite_dir = os.path.dirname(DB_PATH)
                if not os.path.exists(sqlite_dir):
                    os.makedirs(sqlite_dir)
                
                self._conn = sqlite3.connect(DB_PATH)
                print(f"[db] 使用 SQLite: {DB_PATH}")
        
        return self._conn
    
    def execute(self, query: str, params: tuple = None):
        """執行 SQL 查詢"""
        conn = self.get_connection()
        if params:
            return conn.execute(query, params)
        else:
            return conn.execute(query)
    
    def commit(self):
        """提交事務"""
        if self._conn:
            self._conn.commit()
    
    def close(self):
        """關閉連線"""
        if self._conn:
            self._conn.close()
            self._conn = None


def get_conn() -> Union[sqlite3.Connection, Any]:
    """
    取得資料庫連線 - 根據設定自動選擇 SQLite 或 DuckDB
    
    Returns:
        資料庫連線物件
    """
    if USE_DUCKDB and DUCKDB_AVAILABLE:
        # 確保 DuckDB 檔案目錄存在
        duckdb_dir = os.path.dirname(DUCKDB_PATH)
        if not os.path.exists(duckdb_dir):
            os.makedirs(duckdb_dir)
        
        conn = duckdb.connect(str(DUCKDB_PATH))
        print(f"[db] 連線至 DuckDB: {DUCKDB_PATH}")
        return conn
    else:
        if USE_DUCKDB and not DUCKDB_AVAILABLE:
            print("[db] 警告: 設定使用 DuckDB 但未安裝，回退至 SQLite")
        
        # 確保 SQLite 檔案目錄存在
        sqlite_dir = os.path.dirname(DB_PATH)
        if not os.path.exists(sqlite_dir):
            os.makedirs(sqlite_dir)
        
        conn = sqlite3.connect(DB_PATH)
        print(f"[db] 連線至 SQLite: {DB_PATH}")
        return conn


def insert_df(table_name: str, df: pd.DataFrame, if_exists: str = 'append') -> None:
    """
    將 DataFrame 插入資料表
    
    Args:
        table_name: 資料表名稱
        df: 要插入的 DataFrame
        if_exists: 如果表已存在的處理方式 ('fail', 'replace', 'append')
    """
    if df.empty:
        print(f"[db] 警告: DataFrame 為空，跳過插入 {table_name}")
        return
    
    with DatabaseConnection(USE_DUCKDB) as conn:
        try:
            if USE_DUCKDB and DUCKDB_AVAILABLE:
                # DuckDB 使用 pandas 整合
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            else:
                # SQLite 使用 pandas 整合
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            
            print(f"[db] 成功插入 {len(df)} 筆資料至 {table_name}")
            
        except Exception as e:
            print(f"[db] 插入資料失敗 {table_name}: {e}")
            raise


def query_df(query: str, params: tuple = None) -> pd.DataFrame:
    """
    執行查詢並返回 DataFrame
    
    Args:
        query: SQL 查詢語句
        params: 查詢參數
        
    Returns:
        查詢結果的 DataFrame
    """
    with DatabaseConnection(USE_DUCKDB) as conn:
        try:
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
            
            print(f"[db] 查詢返回 {len(df)} 筆資料")
            return df
            
        except Exception as e:
            print(f"[db] 查詢失敗: {e}")
            print(f"[db] SQL: {query}")
            if params:
                print(f"[db] 參數: {params}")
            raise


def execute_sql(query: str, params: tuple = None) -> None:
    """
    執行 SQL 語句（非查詢）
    
    Args:
        query: SQL 語句
        params: 參數
    """
    with DatabaseConnection(USE_DUCKDB) as conn:
        try:
            if params:
                conn.execute(query, params)
            else:
                conn.execute(query)
            
            conn.commit()
            print(f"[db] SQL 執行成功")
            
        except Exception as e:
            print(f"[db] SQL 執行失敗: {e}")
            print(f"[db] SQL: {query}")
            if params:
                print(f"[db] 參數: {params}")
            raise


def create_minute_bars_table() -> None:
    """
    建立 minute_bars 資料表（用於 5 分鐘 K 線資料）
    """
    sql = """
    CREATE TABLE IF NOT EXISTS minute_bars (
        symbol VARCHAR NOT NULL,
        ts TIMESTAMP NOT NULL,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume BIGINT,
        vwap DOUBLE,
        PRIMARY KEY(symbol, ts)
    )
    """
    
    execute_sql(sql)
    print("[db] minute_bars 資料表已建立")


def migrate_sqlite_to_duckdb() -> None:
    """
    將 SQLite 資料遷移至 DuckDB
    """
    if not DUCKDB_AVAILABLE:
        raise ImportError("DuckDB not available for migration")
    
    if not os.path.exists(DB_PATH):
        print("[db] SQLite 檔案不存在，跳過遷移")
        return
    
    print("[db] 開始遷移 SQLite 資料至 DuckDB...")
    
    # 取得所有資料表名稱
    sqlite_conn = sqlite3.connect(DB_PATH)
    tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
    tables_df = pd.read_sql_query(tables_query, sqlite_conn)
    table_names = tables_df['name'].tolist()
    
    # 建立 DuckDB 連線
    duckdb_dir = os.path.dirname(DUCKDB_PATH)
    if not os.path.exists(duckdb_dir):
        os.makedirs(duckdb_dir)
    
    duckdb_conn = duckdb.connect(str(DUCKDB_PATH))
    
    try:
        for table_name in table_names:
            print(f"[db] 遷移資料表: {table_name}")
            
            # 從 SQLite 讀取資料
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)
            
            if not df.empty:
                # 寫入 DuckDB
                df.to_sql(table_name, duckdb_conn, if_exists='replace', index=False)
                print(f"[db] 已遷移 {len(df)} 筆資料至 {table_name}")
            else:
                print(f"[db] {table_name} 無資料，跳過")
    
    finally:
        sqlite_conn.close()
        duckdb_conn.close()
    
    print("[db] 資料遷移完成")


def get_database_info() -> dict:
    """
    取得資料庫資訊
    
    Returns:
        包含資料庫類型、路徑、表數量等資訊的字典
    """
    info = {
        'type': 'DuckDB' if (USE_DUCKDB and DUCKDB_AVAILABLE) else 'SQLite',
        'path': str(DUCKDB_PATH) if (USE_DUCKDB and DUCKDB_AVAILABLE) else DB_PATH,
        'duckdb_available': DUCKDB_AVAILABLE,
        'use_duckdb_setting': USE_DUCKDB
    }
    
    try:
        with DatabaseConnection(USE_DUCKDB) as conn:
            if USE_DUCKDB and DUCKDB_AVAILABLE:
                # DuckDB 查詢資料表
                tables_df = pd.read_sql_query(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'", 
                    conn
                )
            else:
                # SQLite 查詢資料表
                tables_df = pd.read_sql_query(
                    "SELECT name as table_name FROM sqlite_master WHERE type='table'", 
                    conn
                )
            
            info['tables'] = tables_df['table_name'].tolist()
            info['table_count'] = len(info['tables'])
            
    except Exception as e:
        info['error'] = str(e)
        info['tables'] = []
        info['table_count'] = 0
    
    return info


if __name__ == "__main__":
    # 測試腳本
    print("=== 資料庫連線測試 ===")
    
    # 顯示資料庫資訊
    db_info = get_database_info()
    print(f"資料庫類型: {db_info['type']}")
    print(f"資料庫路徑: {db_info['path']}")
    print(f"DuckDB 可用: {db_info['duckdb_available']}")
    print(f"使用 DuckDB 設定: {db_info['use_duckdb_setting']}")
    print(f"資料表數量: {db_info['table_count']}")
    
    if db_info['tables']:
        print("現有資料表:")
        for table in db_info['tables']:
            print(f"  - {table}")
    
    # 測試連線
    try:
        with get_conn() as conn:
            print("✅ 資料庫連線測試成功")
    except Exception as e:
        print(f"❌ 資料庫連線測試失敗: {e}")