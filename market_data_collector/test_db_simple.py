#!/usr/bin/env python3
"""
簡單的資料庫抽象層測試
"""
import sys
import os

# 確保可以 import utils 模組
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """測試模組導入"""
    try:
        from utils.config import USE_DUCKDB, DB_PATH, DUCKDB_PATH
        print(f"✅ Config 導入成功")
        print(f"   USE_DUCKDB: {USE_DUCKDB}")
        print(f"   DB_PATH: {DB_PATH}")
        print(f"   DUCKDB_PATH: {DUCKDB_PATH}")
        return True
    except Exception as e:
        print(f"❌ Config 導入失敗: {e}")
        return False

def test_db_module():
    """測試 db 模組"""
    try:
        from utils.db import get_conn, get_database_info
        print(f"✅ DB 模組導入成功")
        
        # 測試資料庫資訊
        db_info = get_database_info()
        print(f"   資料庫類型: {db_info.get('type', 'Unknown')}")
        print(f"   資料庫路徑: {db_info.get('path', 'Unknown')}")
        print(f"   資料表數量: {db_info.get('table_count', 0)}")
        
        if db_info.get('error'):
            print(f"   錯誤: {db_info['error']}")
        
        return True
    except Exception as e:
        print(f"❌ DB 模組測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_connection():
    """測試資料庫連線"""
    try:
        from utils.db import get_conn
        
        conn = get_conn()
        print(f"✅ 資料庫連線成功: {type(conn)}")
        
        # 測試簡單查詢
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        print(f"✅ 測試查詢成功: {result}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 資料庫連線失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 資料庫抽象層簡單測試 ===")
    
    # 測試導入
    if not test_imports():
        sys.exit(1)
    
    # 測試 DB 模組
    if not test_db_module():
        sys.exit(1)
    
    # 測試連線
    if not test_connection():
        sys.exit(1)
    
    print("✅ 所有測試通過！")