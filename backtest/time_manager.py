# backtest/time_manager.py
"""
時間序列管理器 - 實作 References.txt 建議
包含 Walk-forward 邏輯和剪尾處理
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import logging

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .config import BacktestConfig, Period
except ImportError:
    from config import BacktestConfig, Period

logger = logging.getLogger(__name__)


class TimeSeriesManager:
    """時間序列管理器 - 實作 Walk-forward 和剪尾邏輯"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trading_calendar = None
        self._load_trading_calendar()
    
    def _load_trading_calendar(self):
        """載入交易日曆"""
        try:
            # 嘗試從資料庫載入交易日曆
            from market_data_collector.utils.db import query_df
            
            query = """
            SELECT DISTINCT date 
            FROM candlesticks_daily 
            WHERE date >= '2020-01-01' 
            ORDER BY date ASC
            """
            df = query_df(query)
            
            if not df.empty:
                self.trading_calendar = pd.to_datetime(df['date']).dt.date.tolist()
                logger.info(f"載入交易日曆: {len(self.trading_calendar)} 個交易日")
            else:
                self._create_dummy_calendar()
                
        except Exception as e:
            logger.warning(f"無法載入交易日曆: {e}，使用虛擬日曆")
            self._create_dummy_calendar()
    
    def _create_dummy_calendar(self):
        """創建虛擬交易日曆 (排除週末)"""
        start_date = date(2020, 1, 1)
        end_date = date(2025, 12, 31)
        
        calendar = []
        current_date = start_date
        
        while current_date <= end_date:
            # 排除週末
            if current_date.weekday() < 5:
                calendar.append(current_date)
            current_date += timedelta(days=1)
        
        self.trading_calendar = calendar
        logger.info(f"創建虛擬交易日曆: {len(self.trading_calendar)} 個交易日")
    
    def generate_walk_forward_periods(self, 
                                    start_date: date, 
                                    end_date: date) -> List[Period]:
        """
        生成 Walk-forward 週期
        
        Args:
            start_date: 回測開始日期
            end_date: 回測結束日期
            
        Returns:
            週期列表
        """
        periods = []
        
        # 計算第一個測試期間的開始日期
        first_test_start = start_date + relativedelta(months=self.config.train_window_months)
        
        current_test_start = first_test_start
        period_id = 1
        
        while current_test_start < end_date:
            # 計算測試期間結束日期
            test_end = current_test_start + relativedelta(months=self.config.test_window_months)
            
            # 如果測試期間超出結束日期，調整為結束日期
            if test_end > end_date:
                test_end = end_date
            
            # 計算對應的訓練期間
            train_end = current_test_start - timedelta(days=1)
            train_start = train_end - relativedelta(months=self.config.train_window_months) + timedelta(days=1)
            
            # 確保訓練開始日期不早於回測開始日期
            if train_start < start_date:
                train_start = start_date
            
            # 創建週期
            period = Period(
                train_start=train_start,
                train_end=train_end,
                test_start=current_test_start,
                test_end=test_end,
                period_id=f"P{period_id:03d}"
            )
            
            # References.txt 建議：剪尾處理
            period = self._trim_tail(period, self.config.horizon_days)
            
            periods.append(period)
            
            # 移動到下一個測試期間
            current_test_start += relativedelta(months=self.config.step_size_months)
            period_id += 1
            
            # 如果當前測試開始日期已經超出結束日期，停止
            if current_test_start >= end_date:
                break
        
        logger.info(f"生成 {len(periods)} 個 Walk-forward 週期")
        return periods
    
    def _trim_tail(self, period: Period, horizon: int) -> Period:
        """
        剪尾處理 - References.txt 建議
        剪除訓練期間最後 horizon 天的樣本，避免標籤外溢
        
        Args:
            period: 原始週期
            horizon: 預測視野天數
            
        Returns:
            剪尾後的週期
        """
        # 計算剪尾後的訓練結束日期
        trimmed_train_end = period.train_end - timedelta(days=horizon)
        
        # 確保剪尾後仍有足夠的訓練資料
        min_train_days = 30  # 最少需要 30 天訓練資料
        if (trimmed_train_end - period.train_start).days < min_train_days:
            logger.warning(f"週期 {period.period_id} 剪尾後訓練資料不足，保持原始週期")
            return period
        
        trimmed_period = Period(
            train_start=period.train_start,
            train_end=trimmed_train_end,
            test_start=period.test_start,
            test_end=period.test_end,
            period_id=period.period_id
        )
        
        logger.debug(f"週期 {period.period_id} 剪尾: {period.train_end} -> {trimmed_train_end}")
        return trimmed_period
    
    def get_trading_dates_in_range(self, start_date: date, end_date: date) -> List[date]:
        """獲取指定範圍內的交易日期"""
        if self.trading_calendar is None:
            return []
        
        return [d for d in self.trading_calendar if start_date <= d <= end_date]
    
    def get_previous_trading_date(self, target_date: date) -> Optional[date]:
        """獲取指定日期的前一個交易日"""
        if self.trading_calendar is None:
            return None
        
        for d in reversed(self.trading_calendar):
            if d < target_date:
                return d
        return None
    
    def get_next_trading_date(self, target_date: date) -> Optional[date]:
        """獲取指定日期的下一個交易日"""
        if self.trading_calendar is None:
            return None
        
        for d in self.trading_calendar:
            if d > target_date:
                return d
        return None
    
    def validate_periods(self, periods: List[Period]) -> bool:
        """
        驗證週期列表 - References.txt 建議的單元測試
        確保最後一筆樣本的標籤不跨 period
        """
        for i, period in enumerate(periods):
            # 檢查週期內部一致性
            if period.train_start >= period.train_end:
                logger.error(f"週期 {period.period_id} 訓練日期範圍無效")
                return False
            
            if period.test_start >= period.test_end:
                logger.error(f"週期 {period.period_id} 測試日期範圍無效")
                return False
            
            if period.train_end >= period.test_start:
                logger.error(f"週期 {period.period_id} 訓練和測試期間重疊")
                return False
            
            # 檢查標籤外溢 - 訓練期間最後一筆樣本的標籤不應跨入測試期間
            label_horizon = self.config.horizon_days
            last_sample_date = period.train_end
            last_label_date = last_sample_date + timedelta(days=label_horizon)
            
            if last_label_date > period.test_start:
                logger.error(f"週期 {period.period_id} 存在標籤外溢: "
                           f"最後樣本 {last_sample_date} 的標籤延伸到 {last_label_date}，"
                           f"跨入測試期間 {period.test_start}")
                return False
            
            # 檢查週期間連續性
            if i > 0:
                prev_period = periods[i-1]
                gap_days = (period.test_start - prev_period.test_end).days
                
                if gap_days > 7:  # 允許最多 7 天間隔
                    logger.warning(f"週期 {prev_period.period_id} 和 {period.period_id} "
                                 f"間隔過大: {gap_days} 天")
        
        logger.info(f"週期驗證通過: {len(periods)} 個週期")
        return True
    
    def get_period_summary(self, periods: List[Period]) -> dict:
        """獲取週期摘要統計"""
        if not periods:
            return {}
        
        train_days = [p.train_days for p in periods]
        test_days = [p.test_days for p in periods]
        
        summary = {
            'total_periods': len(periods),
            'date_range': f"{periods[0].train_start} ~ {periods[-1].test_end}",
            'avg_train_days': sum(train_days) / len(train_days),
            'avg_test_days': sum(test_days) / len(test_days),
            'min_train_days': min(train_days),
            'max_train_days': max(train_days),
            'min_test_days': min(test_days),
            'max_test_days': max(test_days),
            'total_train_days': sum(train_days),
            'total_test_days': sum(test_days)
        }
        
        return summary


def test_time_manager():
    """測試時間序列管理器"""
    print("=== 測試時間序列管理器 ===")
    
    # 創建測試配置
    from config import create_smoke_test_config
    config = create_smoke_test_config()
    
    # 創建時間管理器
    print("1. 創建時間管理器...")
    try:
        time_manager = TimeSeriesManager(config)
        print(f"   ✅ 時間管理器創建成功")
        print(f"   交易日曆: {len(time_manager.trading_calendar)} 個交易日")
    except Exception as e:
        print(f"   ❌ 時間管理器創建失敗: {e}")
        return False
    
    # 測試 Walk-forward 週期生成
    print("2. 測試 Walk-forward 週期生成...")
    try:
        start_date = date(2024, 1, 1)
        end_date = date(2024, 6, 30)
        
        periods = time_manager.generate_walk_forward_periods(start_date, end_date)
        
        print(f"   ✅ 生成 {len(periods)} 個週期")
        for period in periods[:3]:  # 顯示前 3 個週期
            print(f"   {period}")
    except Exception as e:
        print(f"   ❌ 週期生成失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 測試週期驗證
    print("3. 測試週期驗證...")
    try:
        is_valid = time_manager.validate_periods(periods)
        if is_valid:
            print("   ✅ 週期驗證通過")
        else:
            print("   ❌ 週期驗證失敗")
            return False
    except Exception as e:
        print(f"   ❌ 週期驗證異常: {e}")
        return False
    
    # 測試週期摘要
    print("4. 測試週期摘要...")
    try:
        summary = time_manager.get_period_summary(periods)
        print("   週期摘要:")
        for key, value in summary.items():
            print(f"     {key}: {value}")
        print("   ✅ 週期摘要生成成功")
    except Exception as e:
        print(f"   ❌ 週期摘要生成失敗: {e}")
        return False
    
    # 測試交易日期查詢
    print("5. 測試交易日期查詢...")
    try:
        test_date = date(2024, 3, 15)
        prev_date = time_manager.get_previous_trading_date(test_date)
        next_date = time_manager.get_next_trading_date(test_date)
        
        print(f"   測試日期: {test_date}")
        print(f"   前一交易日: {prev_date}")
        print(f"   下一交易日: {next_date}")
        print("   ✅ 交易日期查詢成功")
    except Exception as e:
        print(f"   ❌ 交易日期查詢失敗: {e}")
        return False
    
    print("✅ 時間序列管理器測試完成")
    return True


if __name__ == "__main__":
    test_time_manager()