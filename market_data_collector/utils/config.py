# utils/config.py
import os
import random
from pathlib import Path

# Directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "stock_data.db")

# ============================
# DuckDB Settings (新增)
# ============================
DUCKDB_PATH = Path(__file__).resolve().parent.parent / "market.duckdb"
USE_DUCKDB = False  # 設為 True 啟用 DuckDB，False 使用 SQLite (先用 SQLite 測試)
MINUTE_START_DATE = "2020-03-02"  # 分鐘線資料起始日期 (Shioaji 最早可用日期)
# ============================
# FinMind API Settings
# ============================

# FinMind 帳號密碼設定
FINMIND_USER = "ghdhy034@gmail.com"
FINMIND_PASS = "@Ghdhy0930"

API_ENDPOINT = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockPrice"
FINANCIAL_BALANCE_SHEET_DATASET = "TaiwanStockBalanceSheet"
FINANCIAL_INCOME_STATEMENT_DATASET = "TaiwanStockFinancialStatements"
FINANCIAL_MONTH_REVENNUE = "TaiwanStockMonthRevenue"
MARGINPURCHASESHORTSALE = "TaiwanStockMarginPurchaseShortSale"
FINANCIAL_INVESTORSBUYSELL="TaiwanStockInstitutionalInvestorsBuySell"
FINANCIAL_PER="TaiwanStockPER"

# ============================
# Shioaji API Settings
# ============================
SHIOAJI_SIMULATION = False
SHIOAJI_USER = "FSpMKniHrQy7pPEMHbitgFvF5XHPkZMUZx3Y84akeWU6"
SHIOAJI_PASS = "FtMAFmt7fAw2485meZWxeUYkRqEuVDSr3QjehzMFGm3S"
SHIOAJI_CA_PATH = r"C:\Users\user\Desktop\stock\sino_API\credential\Sinopac.pfx"
SHIOAJI_CA_PASS = "O100643287"

# ============================
# Data Date Range
# ============================
START_DATE = "2020-03-02"
END_DATE = "2025-01-25"  # 可根據需求動態更新
MIN_START = "2020-03-02"  # 統一起始日期


# 候選股票列表 (這裡示範使用數字字串，實際應用請根據需求調整)
STOCK_IDS = [
    # Group A - 半導體‧電子供應鏈 (60支)
    "2330", "2317", "2454", "2303", "2408", "2412", "2382", "2357", "2379", "3034",
    "3008", "4938", "2449", "2383", "2356", "3006", "3661", "2324", "8046", "3017",
    "6121", "3037", "3014", "3035", "3062", "3030", "3529", "5443", "2337", "8150",
    "3293", "3596", "2344", "2428", "2345", "2338", "6202", "5347", "3673", "3105",
    "6231", "6669", "4961", "4967", "6668", "4960", "3528", "6147", "3526", "6547",
    "8047", "3227", "4968", "5274", "6415", "6414", "6770", "2331", "6290", "2342",
    # Group B - 傳產／原物料＆運輸 (60支)
    "2603", "2609", "2615", "2610", "2618", "2637", "2606", "2002", "2014", "2027",
    "2201", "1201", "1216", "1301", "1303", "1326", "1710", "1717", "1722", "1723",
    "1402", "1409", "1434", "1476", "2006", "2049", "2105", "2106", "2107", "1605",
    "1609", "1608", "1612", "2308", "1727", "1730", "1101", "1102", "1108", "1210",
    "1215", "1802", "1806", "1810", "1104", "1313", "1314", "1310", "5608", "5607",
    "8105", "8940", "5534", "5609", "5603", "2023", "2028", "2114", "9933", "2501",
    # Group C - 金融‧內需消費／綠能生技 (60支)
    "2880", "2881", "2882", "2883", "2884", "2885", "2886", "2887", "2888", "2890",
    "2891", "2892", "2812", "3665", "2834", "2850", "2801", "2836", "2845", "4807",
    "3702", "3706", "4560", "8478", "4142", "4133", "6525", "6548", "6843", "1513",
    "1514", "1516", "1521", "1522", "1524", "1533", "1708", "3019", "5904", "5906",
    "5902", "6505", "6806", "6510", "2207", "2204", "2231", "1736", "4105", "4108",
    "4162", "1909", "1702", "9917", "1217", "1218", "1737", "1783", "3708", "1795"
]
# [
#     "2330", "2317", "2382", "2449", "3711", "3037", "2308", "2395", "5347", "2486",
#     "4919", "6187", "2454", "2609", "2603", "3481", "3019", "2379", "3231", "8064",
#     "2356", "3665", "3227", "6223", "3029", "2618", "2383", "2368", "2472", "3596",
#     "2303", "3406", "8299", "3706", "1605", "6148", "2344", "2328", "2408", "2027",
#     "2409", "2606", "2002", "2637", "2371", "2610", "3260", "9945", "3162", "1101",
#     "3264", "5309", "1303", "1326", "2547", "8088", "2324", "5439", "2641", "8088"
    
# ]


