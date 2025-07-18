# 訓練模組 SSOT（Single Source of Truth）
> 最後更新：2025-01-15 (75維特徵配置統一)

本文件整合了 **訓練模組** 所需的 *股票劃分規則* 與 *資料庫結構*，並與先前確立的 4090 訓練流程、Optuna 搜尋機制串接，作為後續開發、測試與 CI/CD 的唯一依據。

## 1. 股票組別與資料切割

根據 `stock_split_config.json`：fileciteturn6file0

### 1.1 三大群組

- **group_A**：60 檔 (範例：2330, 2317, 2454, 2303, 2408…)

- **group_B**：60 檔 (範例：2603, 2609, 2615, 2610, 2618…)

- **group_C**：60 檔 (範例：2880, 2881, 2882, 2883, 2884…)


### 1.2 訓練/驗證/測試 切割

| Split | 股票數 | 說明 |
|-------|--------|------|

| **train** | 126 | 3673, 3035, 6415, 4938, 3661… |

| **validation** | 27 | 2331, 2454, 5443, 5347, 2412… |

| **test** | 27 | 6770, 2379, 2356, 3006, 2324… |


> **注意**：切割依照 *行業/權值分布均衡* 原則製定，請勿在訓練流程中改動。

## 2. 資料庫結構總覽

資料來源定義於 `db_structure.json`：fileciteturn6file1

| Table | Rows | Columns | 用途 |
|-------|------|---------|------|

| **candlesticks_min** | 11,467,227 | 10 | 價格序列 (min/daily) |

| **financials** | 3,770 | 20 | 財報原始欄 |

| **technical_indicators** | 233,560 | 21 | 技術指標 |

| **monthly_revenue** | 11,409 | 4 | 月營收 |

| **margin_purchase_shortsale** | 232,260 | 16 | 融資券餘額 |

| **institutional_investors_buy_sell** | 230,655 | 13 | 三大法人買賣超 |

| **financial_per** | 233,329 | 6 | 財報衍生指標 |

| **candlesticks_daily** | 233,560 | 9 | 價格序列 (min/daily) |


> 詳細欄位定義請參考旁錄，或用 `PRAGMA table_info({table_name})` 即時查詢。

## 3. 資料載入與特徵對應

1. **Arrow 分片**：由 `scripts/make_dataset.py` 根據表 2 中來源表格，匯出 `train.arrow` 等分片；切割依 1.2 split 規則。

2. **序列滑窗**：每檔股票以 `seq_len = 64`, `stride = 32` 產生樣本；避免洩漏 `forward_window = 15`。

3. **特徵映射**：`features_registry.py` 對應以下來源 → 模型輸入欄位

   **基本面特徵 (月/季度更新)**：
   - `monthly_revenue`: 1個特徵 (月營收，月更新)
   - `financials`: 14個特徵 (財報數據，季度更新)
   - 小計：15個基本面特徵

   **其他特徵 (每日更新)**：
   - `candlesticks_daily`: 5個特徵 (OHLCV)
   - `technical_indicators`: 17個特徵 (技術指標)
   - `margin_purchase_shortsale`: 13個特徵 (融資融券)
   - `institutional_investors_buy_sell`: 8個特徵 (法人進出，排除Foreign_Dealer_Self)
   - `financial_per`: 3個特徵 (本益比等估值指標)
   - `intraday_structure`: 5個特徵 (從5分K萃取的日內結構信號)
   - 小計：51個其他特徵

   **帳戶狀態**：4個特徵 (由環境即時計算)

   **總計**：15 + 51 + 4 = 70個特徵

## 4. 預設 TrainingConfig 抽要

```python
class TrainingConfig(BaseModel):
    device = 'cuda'          # 4090
    precision = 'fp16'
    seq_len = 64
    stride = 32
    
    # 特徵維度配置 (基於72維標準配置)
    fundamental_features = 15    # 基本面特徵 (月/季度更新)
    other_features = 53         # 其他特徵 (每日更新)
    account_features = 4        # 帳戶狀態特徵
    total_features = 72         # 總特徵數 (15+53+4) - 統一標準

    batch_size = 128
    accum_steps = 3
    num_epochs = 150
    lr = 3e-4
    lr_scheduler = 'cosine'
    warmup_epochs = 5
    early_stop_patience = 40
```

## 5. 特徵更新頻率說明

**基本面特徵 (15個)**：
- 更新頻率：月/季度 (`monthly_revenue`月更新，`financials`季度更新)
- 資料來源：`monthly_revenue` (1個) + `financials` (14個)
- 處理方式：智能時間對齊 (考慮發布延遲)
- 注意事項：需考慮財報發布時間延遲，月營收45天、財報120天時效性限制

**其他特徵 (53個)**：
- 更新頻率：每交易日
- 資料來源：價量、技術指標、籌碼面、估值指標、日內結構信號
- 處理方式：直接使用當日數據，日內結構信號需從5分K萃取
- 注意事項：確保資料時間對齊，candlesticks_min需特殊處理為結構信號

## 6. Optuna 搜索空間

- `lr`: LogUniform(1e-4, 8e-4)
- `d_model`: Categorical[512, 640, 768]
- `n_layer`: Int[4, 10]
- `dropout`: Uniform(0.1, 0.4)
- `k_slip`: Int[1, 4]  # 滑動偏移
- `alpha_beta`: Uniform(0.3, 0.7)  # 買/賣門檻基準

## 7. CI Smoke Tests

| 測試 | 條件 | 過關標準 |
|------|------|-----------|

| `test_train_smoke.py` | batch=8, epoch=2 | loss 下降 & 無 OOM |

| `test_env_agent_loop.py` | 100 steps | 無例外 |

| `test_cli_help.py` | `--help` | exit code 0 |
