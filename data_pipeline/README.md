# Data Pipeline - 分鐘線資料下載器

## 概述

`data_pipeline` 模組負責從多個資料源下載台股分鐘線資料，並自動聚合為 5 分鐘 K 線資料。

## 主要功能

- **智慧路由**: 根據日期自動選擇最適合的資料源
- **多源整合**: 支援 FinMind、Shioaji 和代理資料生成
- **資料聚合**: 1 分鐘 → 5 分鐘 OHLCV + VWAP
- **速率限制**: FinMind API 200 req/min 限制
- **錯誤處理**: 3 次重試，指數退避

## 快速開始

### 1. 安裝依賴
```bash
pip install -r data_pipeline/requirements.txt
```

### 2. 設定 API 金鑰
編輯 `market_data_collector/utils/config.py`:
```python
TOKEN = "your_finmind_token"
SHIOAJI_USER = "your_shioaji_user"
SHIOAJI_PASS = "your_shioaji_pass"
```

### 3. 執行下載
```bash
# 下載單一股票
python -m data_pipeline.fetch_minute --date 2024-07-05 --symbols 2330

# 下載多檔股票
python -m data_pipeline.fetch_minute --date 2024-07-05 --symbols 2330 2603 2317
```

## 測試

### 快速驗證
```bash
cd data_pipeline
python smoke_test.py
```

### 完整測試
```bash
cd data_pipeline
python test_fetch_minute.py
```

### 使用範例
```bash
cd data_pipeline
python example_usage.py
```

## 資料源路由

| 日期範圍 | 資料源 | 說明 |
|---------|--------|------|
| < 2019-05-29 | 代理資料 | 基於次日開盤價生成近似 VWAP |
| 2019-05-29 ~ 2020-03-01 | FinMind | 歷史分鐘線資料 |
| >= 2020-03-02 | Shioaji | 即時和近期分鐘線資料 |

## 輸出格式

資料存入 `minute_bars` 資料表，包含以下欄位：
- `symbol`: 股票代號
- `ts`: 時間戳記 (5 分鐘間隔)
- `open`, `high`, `low`, `close`: OHLC 價格
- `volume`: 成交量
- `vwap`: 成交量加權平均價

## 注意事項

1. **API 限制**: FinMind 有 200 req/min 限制，系統會自動控制
2. **交易時間**: 僅下載 09:00-13:30 交易時段資料
3. **假日處理**: 非交易日會自動跳過
4. **資料品質**: 代理資料僅為近似值，請注意資料來源

## 故障排除

### 常見問題

1. **模組導入失敗**
   ```
   解決: 確保 market_data_collector 在正確路徑
   ```

2. **API 金鑰錯誤**
   ```
   解決: 檢查 config.py 中的 TOKEN 設定
   ```

3. **Shioaji 登入失敗**
   ```
   解決: 確認憑證檔案路徑和密碼正確
   ```

4. **資料庫連線失敗**
   ```
   解決: 執行 create_minute_bars_table() 建立資料表
   ```

## 效能建議

- 批次處理多檔股票以提高效率
- 避開 API 尖峰時段 (開盤前後)
- 使用 `--verbose` 模式監控下載進度
- 定期清理舊的日誌檔案