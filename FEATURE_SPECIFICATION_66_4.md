# TSE Alpha 特徵規格說明 - 66維特徵配置

> **最後更新**: 2025-01-15  
> **基於**: db_structure.json 實際資料表結構  
> **總特徵數**: 66個 (帳戶特徵未來待加入)

## 📊 **特徵總覽**

| 類別 | 數量 | 來源 | 更新頻率 | 狀態 |
|------|------|------|----------|------|
| 基本面特徵 | 15個 | monthly_revenue + financials | 月/季度 | ✅ 使用中 |
| 其他特徵 | 51個 | 多個資料表 | 每日 | ✅ 使用中 |
| 帳戶特徵 | 4個 | 環境計算 | 即時 | 🔄 未來待加入 |
| **總計** | **66個** | - | - | **當前訓練** |

---

## 🔍 **詳細特徵清單**

### **1. 基本面特徵 (15個) - 月/季度更新**

#### **1.1 月營收特徵 (1個)**
**來源表**: `monthly_revenue`
**更新頻率**: 月度 (通常次月10日左右發布)

| 序號 | 欄位名稱 | 說明 |
|------|----------|------|
| 1 | `monthly_revenue` | 月營收 |

#### **1.2 財報特徵 (14個)**
**來源表**: `financials`
**更新頻率**: 季度 (季度結束後1-2個月發布)

| 序號 | 欄位名稱 | 說明 |
|------|----------|------|
| 2 | `cost_of_goods_sold` | 銷貨成本 |
| 3 | `eps` | 每股盈餘 |
| 4 | `equity_attributable_to_owners` | 歸屬母公司股東權益 |
| 5 | `gross_profit` | 毛利 |
| 6 | `income_after_taxes` | 稅後淨利 |
| 7 | `income_from_continuing_operations` | 繼續營業單位損益 |
| 8 | `operating_expenses` | 營業費用 |
| 9 | `operating_income` | 營業利益 |
| 10 | `other_comprehensive_income` | 其他綜合損益 |
| 11 | `pre_tax_income` | 稅前淨利 |
| 12 | `revenue` | 營收 |
| 13 | `tax` | 所得稅費用 |
| 14 | `total_profit` | 本期淨利 |
| 15 | `nonoperating_income_expense` | 營業外收支 |

**排除的financials欄位**: `pe_ratio`, `noncontrolling_interests`, `realized_gain` (不計入特徵)

---

### **2. 其他特徵 (51個) - 每日更新**

#### **2.1 價量特徵 (5個)**
**來源表**: `candlesticks_daily`
**更新頻率**: 每交易日

| 序號 | 欄位名稱 | 說明 |
|------|----------|------|
| 16 | `open` | 開盤價 |
| 17 | `high` | 最高價 |
| 18 | `low` | 最低價 |
| 19 | `close` | 收盤價 |
| 20 | `volume` | 成交量 |

#### **2.2 技術指標特徵 (17個)**
**來源表**: `technical_indicators`
**更新頻率**: 每交易日

| 序號 | 欄位名稱 | 說明 |
|------|----------|------|
| 21 | `sma_5` | 5日簡單移動平均 |
| 22 | `sma_20` | 20日簡單移動平均 |
| 23 | `sma_60` | 60日簡單移動平均 |
| 24 | `ema_12` | 12日指數移動平均 |
| 25 | `ema_26` | 26日指數移動平均 |
| 26 | `ema_50` | 50日指數移動平均 |
| 27 | `macd` | MACD線 |
| 28 | `macd_signal` | MACD信號線 |
| 29 | `macd_hist` | MACD柱狀圖 |
| 30 | `keltner_upper` | Keltner通道上軌 |
| 31 | `keltner_middle` | Keltner通道中軌 |
| 32 | `keltner_lower` | Keltner通道下軌 |
| 33 | `bollinger_upper` | 布林通道上軌 |
| 34 | `bollinger_middle` | 布林通道中軌 |
| 35 | `bollinger_lower` | 布林通道下軌 |
| 36 | `pct_b` | %B指標 |
| 37 | `bandwidth` | 布林通道寬度 |

#### **2.3 籌碼特徵 (21個)**

##### **2.3.1 融資融券特徵 (13個)**
**來源表**: `margin_purchase_shortsale`
**更新頻率**: 每交易日

| 序號 | 欄位名稱 | 說明 |
|------|----------|------|
| 38 | `MarginPurchaseBuy` | 融資買進 |
| 39 | `MarginPurchaseCashRepayment` | 融資現金償還 |
| 40 | `MarginPurchaseLimit` | 融資限額 |
| 41 | `MarginPurchaseSell` | 融資賣出 |
| 42 | `MarginPurchaseTodayBalance` | 融資今日餘額 |
| 43 | `MarginPurchaseYesterdayBalance` | 融資昨日餘額 |
| 44 | `OffsetLoanAndShort` | 資券相抵 |
| 45 | `ShortSaleBuy` | 融券買進 |
| 46 | `ShortSaleCashRepayment` | 融券現金償還 |
| 47 | `ShortSaleLimit` | 融券限額 |
| 48 | `ShortSaleSell` | 融券賣出 |
| 49 | `ShortSaleTodayBalance` | 融券今日餘額 |
| 50 | `ShortSaleYesterdayBalance` | 融券昨日餘額 |

##### **2.3.2 法人進出特徵 (8個)**
**來源表**: `institutional_investors_buy_sell`
**更新頻率**: 每交易日
**排除欄位**: `Foreign_Dealer_Self_buy`, `Foreign_Dealer_Self_sell`

| 序號 | 欄位名稱 | 說明 |
|------|----------|------|
| 51 | `Dealer_Hedging_buy` | 自營商避險買進 |
| 52 | `Dealer_self_buy` | 自營商自行買進 |
| 53 | `Foreign_Investor_buy` | 外資買進 |
| 54 | `Investment_Trust_buy` | 投信買進 |
| 55 | `Dealer_Hedging_sell` | 自營商避險賣出 |
| 56 | `Dealer_self_sell` | 自營商自行賣出 |
| 57 | `Foreign_Investor_sell` | 外資賣出 |
| 58 | `Investment_Trust_sell` | 投信賣出 |

#### **2.4 估值特徵 (3個)**
**來源表**: `financial_per`
**更新頻率**: 每交易日

| 序號 | 欄位名稱 | 說明 |
|------|----------|------|
| 59 | `dividend_yield` | 股息殖利率 |
| 60 | `PER` | 本益比 |
| 61 | `PBR` | 股價淨值比 |

#### **2.5 日內結構特徵 (5個)**
**來源表**: `candlesticks_min` (經處理)
**更新頻率**: 每交易日 (從5分K萃取)

| 序號 | 欄位名稱 | 說明 |
|------|----------|------|
| 62 | `volatility` | 日內波動率 |
| 63 | `vwap_deviation` | VWAP偏離度 |
| 64 | `volume_rhythm` | 成交量節奏 |
| 65 | `shadow_ratio` | 影線比率 |
| 66 | `noise_ratio` | 雜訊比率 |

---

### **3. 帳戶特徵 (4個) - 即時計算**

**來源**: 交易環境 (TSEAlphaEnv)
**更新頻率**: 每個交易步驟

| 序號 | 欄位名稱 | 說明 |
|------|----------|------|
| 67 | `nav` | 淨資產價值 |
| 68 | `cash_ratio` | 現金比率 |
| 69 | `exposure` | 市場曝險 |
| 70 | `max_drawdown` | 最大回撤 |

---

## 📋 **特徵分組摘要**

### **按更新頻率分組**
- **即時更新**: 4個 (帳戶特徵)
- **每日更新**: 51個 (價量 + 技術 + 籌碼 + 估值 + 日內)
- **月度更新**: 1個 (月營收)
- **季度更新**: 14個 (財報)

### **按資料來源分組**
- **candlesticks_daily**: 5個
- **technical_indicators**: 17個
- **margin_purchase_shortsale**: 13個
- **institutional_investors_buy_sell**: 8個
- **financial_per**: 3個
- **monthly_revenue**: 1個
- **financials**: 14個
- **candlesticks_min (處理後)**: 5個
- **環境計算**: 4個

### **按特徵類型分組**
- **基本面**: 15個 (財務健康度)
- **技術面**: 22個 (價量 + 技術指標)
- **籌碼面**: 21個 (融資融券 + 法人進出)
- **估值面**: 3個 (評價指標)
- **結構面**: 5個 (日內結構)
- **帳戶面**: 4個 (交易狀態)

---

## ⚠️ **重要注意事項**

### **資料完整性**
1. **基本面特徵**: 需要智能時間對齊 (考慮發布延遲)
2. **籌碼特徵**: 排除 `Foreign_Dealer_Self_buy/sell` (按需求)
3. **日內結構**: 需要從5分K資料萃取處理
4. **帳戶特徵**: 由環境動態計算，不存儲在資料庫

### **特徵維度檢查**
- **特徵工程輸出**: 66維 (不含帳戶特徵)
- **模型輸入**: 66維特徵工程 (4維帳戶特徵尚未計畫加入訓練)
- **SSOT標準**: 嚴格遵循此規格

### **向後相容性**
- 此規格取代之前的72維配置
- 基於實際資料庫結構，確保可實現性
- 所有相關文檔需要同步更新

---

**此文檔為TSE Alpha系統特徵配置的唯一權威來源 (Single Source of Truth)**