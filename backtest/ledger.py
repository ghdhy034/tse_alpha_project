# backtest/ledger.py
"""
帳本系統 - 使用 DuckDB 記錄交易和持倉
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import logging

# 添加路徑
sys.path.append(str(Path(__file__).parent.parent / "market_data_collector"))

try:
    from market_data_collector.utils.db import get_conn, insert_df, query_df, execute_sql
except ImportError as e:
    print(f"警告: 無法導入資料庫模組: {e}")

logger = logging.getLogger(__name__)


class TradingLedger:
    """交易帳本 - 記錄所有交易和持倉變化"""
    
    def __init__(self, account_id: str = "default"):
        self.account_id = account_id
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 確保資料表存在
        self._create_tables()
        
    def _create_tables(self):
        """建立帳本相關資料表"""
        try:
            # 交易記錄表
            execute_sql("""
                CREATE TABLE IF NOT EXISTS trade_records (
                    id INTEGER PRIMARY KEY,
                    account_id VARCHAR NOT NULL,
                    session_id VARCHAR NOT NULL,
                    trade_date DATE NOT NULL,
                    symbol VARCHAR NOT NULL,
                    action VARCHAR NOT NULL,  -- 'BUY' or 'SELL'
                    quantity INTEGER NOT NULL,
                    price DECIMAL(10,2) NOT NULL,
                    trade_value DECIMAL(15,2) NOT NULL,
                    commission DECIMAL(10,2) NOT NULL,
                    tax DECIMAL(10,2) NOT NULL,
                    total_cost DECIMAL(15,2) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 持倉快照表
            execute_sql("""
                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY,
                    account_id VARCHAR NOT NULL,
                    session_id VARCHAR NOT NULL,
                    snapshot_date DATE NOT NULL,
                    symbol VARCHAR NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_price DECIMAL(10,2) NOT NULL,
                    market_price DECIMAL(10,2) NOT NULL,
                    market_value DECIMAL(15,2) NOT NULL,
                    unrealized_pnl DECIMAL(15,2) NOT NULL,
                    days_held INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 帳戶快照表
            execute_sql("""
                CREATE TABLE IF NOT EXISTS account_snapshots (
                    id INTEGER PRIMARY KEY,
                    account_id VARCHAR NOT NULL,
                    session_id VARCHAR NOT NULL,
                    snapshot_date DATE NOT NULL,
                    cash DECIMAL(15,2) NOT NULL,
                    position_value DECIMAL(15,2) NOT NULL,
                    total_nav DECIMAL(15,2) NOT NULL,
                    daily_pnl DECIMAL(15,2),
                    cumulative_pnl DECIMAL(15,2),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 績效指標表
            execute_sql("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY,
                    account_id VARCHAR NOT NULL,
                    session_id VARCHAR NOT NULL,
                    metric_date DATE NOT NULL,
                    total_return DECIMAL(10,6),
                    sharpe_ratio DECIMAL(10,6),
                    max_drawdown DECIMAL(10,6),
                    win_rate DECIMAL(10,6),
                    avg_win DECIMAL(10,6),
                    avg_loss DECIMAL(10,6),
                    total_trades INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            logger.info("帳本資料表建立完成")
            
        except Exception as e:
            logger.error(f"建立帳本資料表失敗: {e}")
            raise
    
    def record_trade(self, 
                    trade_date: date,
                    symbol: str,
                    action: str,
                    quantity: int,
                    price: float,
                    commission_rate: float = 0.001425,
                    tax_rate: float = 0.003) -> Dict[str, Any]:
        """
        記錄交易
        
        Args:
            trade_date: 交易日期
            symbol: 股票代號
            action: 'BUY' or 'SELL'
            quantity: 交易數量
            price: 交易價格
            commission_rate: 手續費率
            tax_rate: 交易稅率 (僅賣出時收取)
            
        Returns:
            交易成本明細
        """
        try:
            # 使用 Decimal 進行精確計算
            price_decimal = Decimal(str(price))
            quantity_decimal = Decimal(str(quantity))
            
            # 計算交易金額
            trade_value = price_decimal * quantity_decimal
            
            # 計算手續費 (買賣都收，最低 20 元)
            commission = max(trade_value * Decimal(str(commission_rate)), Decimal('20'))
            commission = commission.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            # 計算交易稅 (僅賣出時收取)
            if action.upper() == 'SELL':
                tax = trade_value * Decimal(str(tax_rate))
                tax = tax.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            else:
                tax = Decimal('0')
            
            # 計算總成本
            if action.upper() == 'BUY':
                total_cost = trade_value + commission + tax
            else:  # SELL
                total_cost = trade_value - commission - tax
            
            # 準備交易記錄
            trade_record = {
                'account_id': self.account_id,
                'session_id': self.session_id,
                'trade_date': trade_date,
                'symbol': symbol,
                'action': action.upper(),
                'quantity': quantity,
                'price': float(price_decimal),
                'trade_value': float(trade_value),
                'commission': float(commission),
                'tax': float(tax),
                'total_cost': float(total_cost)
            }
            
            # 插入資料庫
            df = pd.DataFrame([trade_record])
            insert_df('trade_records', df, if_exists='append')
            
            logger.info(f"記錄交易: {symbol} {action} {quantity} @ {price}")
            
            return {
                'trade_value': float(trade_value),
                'commission': float(commission),
                'tax': float(tax),
                'total_cost': float(total_cost)
            }
            
        except Exception as e:
            logger.error(f"記錄交易失敗: {e}")
            raise
    
    def update_positions(self, 
                        snapshot_date: date,
                        positions: Dict[str, Dict],
                        current_prices: Dict[str, float]):
        """
        更新持倉快照
        
        Args:
            snapshot_date: 快照日期
            positions: 持倉字典 {symbol: {'qty': int, 'avg_price': float, 'days_held': int}}
            current_prices: 當前價格字典
        """
        try:
            position_records = []
            
            for symbol, pos in positions.items():
                if symbol in current_prices:
                    market_price = current_prices[symbol]
                    market_value = pos['qty'] * market_price
                    cost_value = pos['qty'] * pos['avg_price']
                    unrealized_pnl = market_value - cost_value
                    
                    position_records.append({
                        'account_id': self.account_id,
                        'session_id': self.session_id,
                        'snapshot_date': snapshot_date,
                        'symbol': symbol,
                        'quantity': pos['qty'],
                        'avg_price': pos['avg_price'],
                        'market_price': market_price,
                        'market_value': market_value,
                        'unrealized_pnl': unrealized_pnl,
                        'days_held': pos['days_held']
                    })
            
            if position_records:
                df = pd.DataFrame(position_records)
                insert_df('position_snapshots', df, if_exists='append')
                logger.debug(f"更新 {len(position_records)} 筆持倉快照")
            
        except Exception as e:
            logger.error(f"更新持倉快照失敗: {e}")
            raise
    
    def update_account(self,
                      snapshot_date: date,
                      cash: float,
                      position_value: float,
                      total_nav: float,
                      daily_pnl: Optional[float] = None,
                      cumulative_pnl: Optional[float] = None):
        """
        更新帳戶快照
        
        Args:
            snapshot_date: 快照日期
            cash: 現金餘額
            position_value: 持倉市值
            total_nav: 總淨值
            daily_pnl: 當日損益
            cumulative_pnl: 累積損益
        """
        try:
            account_record = {
                'account_id': self.account_id,
                'session_id': self.session_id,
                'snapshot_date': snapshot_date,
                'cash': cash,
                'position_value': position_value,
                'total_nav': total_nav,
                'daily_pnl': daily_pnl,
                'cumulative_pnl': cumulative_pnl
            }
            
            df = pd.DataFrame([account_record])
            insert_df('account_snapshots', df, if_exists='append')
            
            logger.debug(f"更新帳戶快照: NAV={total_nav:,.2f}")
            
        except Exception as e:
            logger.error(f"更新帳戶快照失敗: {e}")
            raise
    
    def calculate_performance_metrics(self, end_date: Optional[date] = None) -> Dict[str, float]:
        """
        計算績效指標
        
        Args:
            end_date: 計算截止日期
            
        Returns:
            績效指標字典
        """
        try:
            # 獲取帳戶快照
            query = """
            SELECT snapshot_date, total_nav, daily_pnl
            FROM account_snapshots
            WHERE account_id = ? AND session_id = ?
            ORDER BY snapshot_date ASC
            """
            
            df_account = query_df(query, (self.account_id, self.session_id))
            
            if df_account.empty:
                return {}
            
            # 計算總收益率
            initial_nav = df_account.iloc[0]['total_nav']
            final_nav = df_account.iloc[-1]['total_nav']
            total_return = (final_nav - initial_nav) / initial_nav
            
            # 計算 Sharpe 比率
            daily_returns = df_account['daily_pnl'] / df_account['total_nav'].shift(1)
            daily_returns = daily_returns.dropna()
            
            if len(daily_returns) > 1:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252 ** 0.5)
            else:
                sharpe_ratio = 0.0
            
            # 計算最大回撤
            nav_series = df_account['total_nav']
            peak = nav_series.expanding().max()
            drawdown = (nav_series - peak) / peak
            max_drawdown = drawdown.min()
            
            # 獲取交易統計
            trade_query = """
            SELECT symbol, action, quantity, price, total_cost
            FROM trade_records
            WHERE account_id = ? AND session_id = ?
            ORDER BY trade_date ASC
            """
            
            df_trades = query_df(trade_query, (self.account_id, self.session_id))
            
            # 計算交易統計
            total_trades = len(df_trades)
            
            # 計算勝率 (簡化版本，基於完整的買賣對)
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            
            if not df_trades.empty:
                # 這裡可以實作更複雜的交易對分析
                # 暫時使用簡化版本
                profits = daily_returns[daily_returns > 0]
                losses = daily_returns[daily_returns < 0]
                
                if len(daily_returns) > 0:
                    win_rate = len(profits) / len(daily_returns)
                
                if len(profits) > 0:
                    avg_win = profits.mean()
                
                if len(losses) > 0:
                    avg_loss = abs(losses.mean())
            
            metrics = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': abs(max_drawdown),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_trades': total_trades
            }
            
            # 記錄績效指標
            metric_date = end_date or date.today()
            metric_record = {
                'account_id': self.account_id,
                'session_id': self.session_id,
                'metric_date': metric_date,
                **metrics
            }
            
            df_metrics = pd.DataFrame([metric_record])
            insert_df('performance_metrics', df_metrics, if_exists='append')
            
            logger.info(f"績效指標計算完成: 總收益 {total_return:.2%}, Sharpe {sharpe_ratio:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"計算績效指標失敗: {e}")
            return {}
    
    def get_trade_history(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """獲取交易歷史"""
        try:
            if symbol:
                query = """
                SELECT * FROM trade_records
                WHERE account_id = ? AND session_id = ? AND symbol = ?
                ORDER BY trade_date ASC, timestamp ASC
                """
                return query_df(query, (self.account_id, self.session_id, symbol))
            else:
                query = """
                SELECT * FROM trade_records
                WHERE account_id = ? AND session_id = ?
                ORDER BY trade_date ASC, timestamp ASC
                """
                return query_df(query, (self.account_id, self.session_id))
                
        except Exception as e:
            logger.error(f"獲取交易歷史失敗: {e}")
            return pd.DataFrame()
    
    def get_nav_history(self) -> pd.DataFrame:
        """獲取 NAV 歷史"""
        try:
            query = """
            SELECT snapshot_date, total_nav, daily_pnl, cumulative_pnl
            FROM account_snapshots
            WHERE account_id = ? AND session_id = ?
            ORDER BY snapshot_date ASC
            """
            return query_df(query, (self.account_id, self.session_id))
            
        except Exception as e:
            logger.error(f"獲取 NAV 歷史失敗: {e}")
            return pd.DataFrame()
    
    def cleanup_session(self):
        """清理當前 session 的資料"""
        try:
            tables = ['trade_records', 'position_snapshots', 'account_snapshots', 'performance_metrics']
            
            for table in tables:
                execute_sql(f"""
                    DELETE FROM {table}
                    WHERE account_id = ? AND session_id = ?
                """, (self.account_id, self.session_id))
            
            logger.info(f"清理 session {self.session_id} 完成")
            
        except Exception as e:
            logger.error(f"清理 session 失敗: {e}")


def test_ledger():
    """測試帳本系統"""
    print("=== 測試交易帳本系統 ===")
    
    try:
        # 建立帳本
        ledger = TradingLedger("test_account")
        
        # 清理測試資料
        ledger.cleanup_session()
        
        # 測試交易記錄
        print("1. 測試交易記錄...")
        trade_cost = ledger.record_trade(
            trade_date=date(2024, 1, 2),
            symbol='2330',
            action='BUY',
            quantity=100,
            price=580.0
        )
        print(f"   買入成本: {trade_cost}")
        
        trade_cost = ledger.record_trade(
            trade_date=date(2024, 1, 3),
            symbol='2330',
            action='SELL',
            quantity=50,
            price=590.0
        )
        print(f"   賣出收入: {trade_cost}")
        
        # 測試持倉更新
        print("2. 測試持倉更新...")
        positions = {
            '2330': {'qty': 50, 'avg_price': 580.0, 'days_held': 1}
        }
        current_prices = {'2330': 590.0}
        
        ledger.update_positions(
            snapshot_date=date(2024, 1, 3),
            positions=positions,
            current_prices=current_prices
        )
        
        # 測試帳戶更新
        print("3. 測試帳戶更新...")
        ledger.update_account(
            snapshot_date=date(2024, 1, 3),
            cash=970000.0,  # 扣除買入成本後的現金
            position_value=29500.0,  # 50 股 * 590
            total_nav=999500.0,
            daily_pnl=500.0,
            cumulative_pnl=500.0
        )
        
        # 測試績效計算
        print("4. 測試績效計算...")
        metrics = ledger.calculate_performance_metrics()
        print(f"   績效指標: {metrics}")
        
        # 測試歷史查詢
        print("5. 測試歷史查詢...")
        trade_history = ledger.get_trade_history()
        print(f"   交易記錄數: {len(trade_history)}")
        
        nav_history = ledger.get_nav_history()
        print(f"   NAV 記錄數: {len(nav_history)}")
        
        print("✅ 帳本系統測試完成")
        
        # 清理測試資料
        ledger.cleanup_session()
        
    except Exception as e:
        print(f"❌ 帳本系統測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ledger()