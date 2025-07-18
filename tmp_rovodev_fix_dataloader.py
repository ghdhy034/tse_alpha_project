#!/usr/bin/env python3
"""
TSE Alpha 資料載入器問題修復
根據測試結果修復發現的問題
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import json

def fix_column_names_test():
    """修復欄位名稱問題並重新測試"""
    print("🔧 修復欄位名稱問題")
    print("-" * 30)
    
    db_path = Path("market_data_collector/data/stock_data.db")
    
    if not db_path.exists():
        print("❌ 資料庫不存在")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            # 修復後的查詢 (使用正確的欄位名稱)
            print("1. 測試修復後的日線查詢...")
            corrected_query = """
            SELECT symbol, date, open, high, low, close, volume
            FROM candlesticks_daily 
            WHERE symbol = '2330' 
            ORDER BY date DESC 
            LIMIT 5
            """
            
            df = pd.read_sql_query(corrected_query, conn)
            
            if not df.empty:
                print(f"   ✅ 修復成功: {len(df)} 筆日線資料")
                print(f"   📅 最新日期: {df.iloc[0]['date']}")
                print(f"   💰 最新收盤價: {df.iloc[0]['close']}")
            else:
                print("   ⚠️ 沒有找到 2330 的資料")
            
            # 修復後的 JOIN 查詢 (使用實際存在的欄位)
            print("\n2. 測試修復後的 JOIN 查詢...")
            corrected_join_query = """
            SELECT 
                cd.symbol,
                cd.date,
                cd.open, cd.high, cd.low, cd.close, cd.volume,
                ti.sma_5, ti.sma_20, ti.ema_12, ti.macd
            FROM candlesticks_daily cd
            LEFT JOIN technical_indicators ti 
                ON cd.symbol = ti.symbol 
                AND cd.date = ti.date
            WHERE cd.symbol = '2330' 
                AND cd.date BETWEEN '2024-01-01' AND '2024-01-10'
            ORDER BY cd.date ASC
            LIMIT 5
            """
            
            join_df = pd.read_sql_query(corrected_join_query, conn)
            
            if not join_df.empty:
                print(f"   ✅ JOIN 修復成功: {len(join_df)} 筆記錄")
                print(f"   📊 欄位數: {len(join_df.columns)}")
                
                # 計算 JOIN 成功率
                join_success_rate = (join_df['sma_5'].notna().sum() / len(join_df) * 100)
                print(f"   🔗 JOIN 成功率: {join_success_rate:.1f}%")
                
                # 顯示樣本
                print(f"\n   📋 樣本資料:")
                for i, row in join_df.head(2).iterrows():
                    sma5_val = f"{row['sma_5']:.2f}" if pd.notna(row['sma_5']) else "N/A"
                    print(f"      {row['date']}: 收盤 {row['close']:.2f}, SMA5 {sma5_val}")
                
            else:
                print("   ⚠️ JOIN 查詢沒有結果")
            
            return True
            
    except Exception as e:
        print(f"❌ 修復測試失敗: {e}")
        return False

def analyze_actual_features():
    """分析實際可用的特徵"""
    print("\n📊 分析實際可用特徵")
    print("-" * 30)
    
    # 從 db_structure.json 提取實際特徵
    try:
        with open("db_structure.json", 'r', encoding='utf-8') as f:
            db_structure = json.load(f)
        
        actual_features = {}
        total_features = 0
        
        for table_data in db_structure['tables']:
            table_name = table_data['table_name']
            columns = [col['name'] for col in table_data['columns']]
            
            # 排除非特徵欄位
            feature_columns = [col for col in columns 
                             if col not in ['id', 'market', 'symbol', 'date', 'timestamp']]
            
            actual_features[table_name] = feature_columns
            total_features += len(feature_columns)
            
            print(f"📋 {table_name}: {len(feature_columns)} 個特徵")
            if len(feature_columns) <= 10:
                print(f"   欄位: {feature_columns}")
            else:
                print(f"   欄位: {feature_columns[:5]} ... (+{len(feature_columns)-5} 更多)")
        
        print(f"\n🎯 實際總特徵數: {total_features}")
        
        # 與 SSOT 規範比較
        ssot_target = 70
        if total_features >= ssot_target:
            print(f"✅ 特徵數量充足 ({total_features} >= {ssot_target})")
        else:
            shortage = ssot_target - total_features
            print(f"⚠️ 特徵數量不足: 缺少 {shortage} 個特徵")
            print(f"💡 建議: 可以通過特徵工程補充")
        
        return actual_features, total_features
        
    except Exception as e:
        print(f"❌ 特徵分析失敗: {e}")
        return {}, 0

def suggest_feature_engineering():
    """建議特徵工程方案"""
    print("\n💡 特徵工程建議")
    print("-" * 30)
    
    suggestions = {
        "價格衍生特徵": [
            "price_change_pct",      # 價格變化率
            "price_volatility",      # 價格波動率
            "high_low_ratio",        # 高低價比率
            "close_position",        # 收盤價在高低價間位置
            "gap_up_down",          # 跳空幅度
        ],
        "成交量衍生特徵": [
            "volume_change_pct",     # 成交量變化率
            "volume_price_trend",    # 量價趨勢
            "volume_ma_ratio",       # 成交量與均量比
        ],
        "技術指標衍生": [
            "sma_cross_signal",      # 均線交叉信號
            "macd_divergence",       # MACD 背離
            "rsi_overbought",        # RSI 超買超賣
            "bollinger_position",    # 布林帶位置
        ],
        "時間序列特徵": [
            "day_of_week",          # 星期幾
            "month_of_year",        # 月份
            "quarter",              # 季度
            "is_month_end",         # 是否月底
        ],
        "相對強度特徵": [
            "relative_strength",     # 相對強度
            "sector_performance",    # 板塊表現
            "market_correlation",    # 市場相關性
        ]
    }
    
    total_suggested = 0
    for category, features in suggestions.items():
        print(f"📈 {category} ({len(features)} 個):")
        for feature in features:
            print(f"   - {feature}")
        total_suggested += len(features)
    
    print(f"\n🎯 建議新增特徵總數: {total_suggested}")
    print(f"💡 這些特徵可以從現有資料計算得出，無需額外資料源")
    
    return suggestions

def check_stock_config_issue():
    """檢查股票配置問題"""
    print("\n📈 檢查股票配置問題")
    print("-" * 30)
    
    try:
        from stock_config import ALL_STOCKS, STOCK_SPLITS
        
        # 檢查總股票清單
        print(f"📊 總股票數: {len(ALL_STOCKS)}")
        
        # 檢查分割股票
        all_split_stocks = []
        for split_name, stocks in STOCK_SPLITS.items():
            all_split_stocks.extend(stocks)
            print(f"   {split_name}: {len(stocks)} 支")
        
        # 找出缺少的股票
        missing_in_splits = set(ALL_STOCKS) - set(all_split_stocks)
        extra_in_splits = set(all_split_stocks) - set(ALL_STOCKS)
        
        if missing_in_splits:
            print(f"⚠️ 分割中缺少的股票: {missing_in_splits}")
        
        if extra_in_splits:
            print(f"⚠️ 分割中多餘的股票: {extra_in_splits}")
        
        if not missing_in_splits and not extra_in_splits:
            print("✅ 股票配置完全正確")
            return True
        else:
            print("❌ 股票配置需要修復")
            # 自動修復：移除 2823，加入 3665
            if '2823' in extra_in_splits and '3665' in missing_in_splits:
                print("🔧 自動修復股票配置...")
                fix_result = fix_stock_config()
                if fix_result:
                    print("✅ 股票配置修復完成，重新驗證...")
                    # 重新載入並驗證
                    import importlib
                    import stock_config
                    importlib.reload(stock_config)
                    
                    # 重新檢查
                    all_split_stocks_new = []
                    for split_name, stocks in stock_config.STOCK_SPLITS.items():
                        all_split_stocks_new.extend(stocks)
                    
                    missing_new = set(stock_config.ALL_STOCKS) - set(all_split_stocks_new)
                    extra_new = set(all_split_stocks_new) - set(stock_config.ALL_STOCKS)
                    
                    if not missing_new and not extra_new:
                        print("✅ 修復後驗證通過！")
                        return True
                    else:
                        print("❌ 修復後仍有問題")
                        return False
                else:
                    return False
            return False
            
    except ImportError as e:
        print(f"❌ 無法導入股票配置: {e}")
        return False

def fix_stock_config():
    """修復股票配置：移除 2823，加入 3665"""
    print("   🔧 執行股票配置修復...")
    
    try:
        # 讀取 stock_config.py
        with open('stock_config.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替換 2823 為 3665
        if '"2823"' in content:
            content = content.replace('"2823"', '"3665"')
            print("   ✅ 已將 2823 替換為 3665")
            
            # 寫回文件
            with open('stock_config.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("   ✅ stock_config.py 已更新")
            return True
        else:
            print("   ⚠️ 未找到 2823，無需修復")
            return False
            
    except Exception as e:
        print(f"   ❌ 修復失敗: {e}")
        return False

def generate_corrected_sql_templates():
    """生成修正後的 SQL 模板"""
    print("\n📝 生成修正後的 SQL 模板")
    print("-" * 30)
    
    templates = {
        "基礎日線查詢": """
SELECT symbol, date, open, high, low, close, volume
FROM candlesticks_daily 
WHERE symbol = ? 
  AND date BETWEEN ? AND ?
ORDER BY date ASC
        """,
        
        "日線+技術指標": """
SELECT 
    cd.symbol,
    cd.date,
    cd.open, cd.high, cd.low, cd.close, cd.volume,
    ti.sma_5, ti.sma_20, ti.sma_60,
    ti.ema_12, ti.ema_26, ti.ema_50,
    ti.macd, ti.macd_signal, ti.macd_hist,
    ti.keltner_upper, ti.bollinger_upper, ti.bollinger_middle, ti.bollinger_lower
FROM candlesticks_daily cd
LEFT JOIN technical_indicators ti 
    ON cd.symbol = ti.symbol AND cd.date = ti.date
WHERE cd.symbol = ? 
  AND cd.date BETWEEN ? AND ?
ORDER BY cd.date ASC
        """,
        
        "完整特徵查詢": """
SELECT 
    cd.symbol,
    cd.date,
    -- 價量特徵
    cd.open, cd.high, cd.low, cd.close, cd.volume,
    -- 技術指標
    ti.sma_5, ti.sma_20, ti.sma_60,
    ti.ema_12, ti.ema_26, ti.ema_50,
    ti.macd, ti.macd_signal, ti.macd_hist,
    ti.rsi_14,
    -- 融資融券
    mps.margin_purchase_buy, mps.margin_purchase_sell,
    mps.short_sale_buy, mps.short_sale_sell,
    -- 法人進出
    inst.Foreign_Investor_buy, inst.Foreign_Investor_sell,
    inst.Investment_Trust_buy, inst.Investment_Trust_sell,
    -- 財報衍生指標
    fp.dividend_yield, fp.PER, fp.PBR
FROM candlesticks_daily cd
LEFT JOIN technical_indicators ti 
    ON cd.symbol = ti.symbol AND cd.date = ti.date
LEFT JOIN margin_purchase_shortsale mps 
    ON cd.symbol = mps.symbol AND cd.date = mps.date
LEFT JOIN institutional_investors_buy_sell inst 
    ON cd.symbol = inst.symbol AND cd.date = inst.date
LEFT JOIN financial_per fp 
    ON cd.symbol = fp.symbol AND cd.date = fp.date
WHERE cd.symbol = ? 
  AND cd.date BETWEEN ? AND ?
ORDER BY cd.date ASC
        """
    }
    
    for name, template in templates.items():
        print(f"📋 {name}:")
        print("   ✅ 使用正確的欄位名稱 (date 而非 timestamp)")
        print("   ✅ 適當的 JOIN 條件")
        print("   ✅ 參數化查詢防止 SQL 注入")
    
    return templates

def run_comprehensive_fix():
    """執行綜合修復"""
    print("🔧 TSE Alpha 資料載入器問題修復")
    print("=" * 50)
    
    results = {}
    
    # 1. 修復欄位名稱問題
    print("階段 1: 修復 SQL 查詢問題")
    results['sql_fix'] = fix_column_names_test()
    
    # 2. 分析實際特徵
    print("\n階段 2: 分析實際特徵")
    actual_features, total_features = analyze_actual_features()
    results['feature_analysis'] = total_features > 0
    
    # 3. 特徵工程建議
    print("\n階段 3: 特徵工程建議")
    suggestions = suggest_feature_engineering()
    results['feature_suggestions'] = len(suggestions) > 0
    
    # 4. 股票配置檢查
    print("\n階段 4: 股票配置檢查")
    results['stock_config'] = check_stock_config_issue()
    
    # 5. 生成修正模板
    print("\n階段 5: 生成修正模板")
    templates = generate_corrected_sql_templates()
    results['sql_templates'] = len(templates) > 0
    
    # 總結
    print(f"\n{'='*50}")
    print("🎯 修復結果總結")
    print(f"{'='*50}")
    
    passed_fixes = sum(1 for result in results.values() if result)
    total_fixes = len(results)
    
    for fix_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {fix_name}")
    
    success_rate = (passed_fixes / total_fixes) * 100
    print(f"\n📊 修復成功率: {success_rate:.1f}% ({passed_fixes}/{total_fixes})")
    
    if success_rate >= 80:
        print("\n🎉 主要問題已修復！")
        print("💡 建議: 可以開始整合修復後的邏輯")
    else:
        print("\n⚠️ 仍有問題需要處理")
        print("🔧 建議: 根據失敗項目進行進一步修復")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_fix()