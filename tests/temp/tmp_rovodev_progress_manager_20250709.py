#!/usr/bin/env python3
"""
進度管理器 - 查看和管理資料收集進度
"""
import json
import os
from datetime import datetime
from typing import Dict, List

class ProgressManager:
    """進度管理器"""
    
    def __init__(self, progress_file: str = "data_collection_progress.json"):
        self.progress_file = progress_file
    
    def load_progress(self) -> Dict:
        """載入進度資料"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"載入進度檔案失敗: {e}")
        
        return {}
    
    def show_progress(self):
        """顯示進度狀況"""
        progress = self.load_progress()
        
        if not progress:
            print("❌ 沒有找到進度檔案")
            return
        
        print("=== 資料收集進度報告 ===")
        print(f"最後更新: {progress.get('last_update', 'N/A')}")
        
        # 統計資料
        stats = progress.get('statistics', {})
        print(f"\n📊 統計資料:")
        print(f"  總請求數: {stats.get('total_requests', 0)}")
        print(f"  成功請求: {stats.get('successful_requests', 0)}")
        print(f"  失敗請求: {stats.get('failed_requests', 0)}")
        
        if stats.get('total_requests', 0) > 0:
            success_rate = stats.get('successful_requests', 0) / stats.get('total_requests', 1) * 100
            print(f"  成功率: {success_rate:.1f}%")
        
        # 完成狀況
        completed = progress.get('completed_symbols', [])
        failed = progress.get('failed_symbols', [])
        
        print(f"\n✅ 已完成: {len(completed)} 個任務")
        print(f"❌ 失敗: {len(failed)} 個任務")
        
        # 按資料類型分組統計
        data_types = {}
        for item in completed:
            if '_' in item:
                symbol, data_type = item.rsplit('_', 1)
                if data_type not in data_types:
                    data_types[data_type] = 0
                data_types[data_type] += 1
        
        print(f"\n📋 按資料類型統計:")
        for data_type, count in data_types.items():
            print(f"  {data_type}: {count} 支股票")
        
        # 失敗項目詳情
        if failed:
            print(f"\n❌ 失敗項目 (最近10個):")
            for item in failed[-10:]:
                if isinstance(item, dict):
                    print(f"  {item.get('key', 'N/A')}: {item.get('error', 'N/A')}")
                else:
                    print(f"  {item}")
    
    def reset_progress(self):
        """重置進度"""
        if os.path.exists(self.progress_file):
            backup_file = f"{self.progress_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(self.progress_file, backup_file)
            print(f"進度檔案已備份為: {backup_file}")
        
        print("進度已重置")
    
    def get_remaining_tasks(self, all_symbols: List[str] = None) -> List[str]:
        """獲取剩餘任務"""
        if all_symbols is None:
            # 預設180支股票清單
            all_symbols = [f"{i:04d}" for i in range(1101, 1281)]  # 示例
        
        progress = self.load_progress()
        completed = set(progress.get('completed_symbols', []))
        
        data_types = ["daily_price", "margin_shortsale", "institutional", "financial", "balance_sheet", "monthly_revenue"]
        
        remaining = []
        for symbol in all_symbols:
            for data_type in data_types:
                task_key = f"{symbol}_{data_type}"
                if task_key not in completed:
                    remaining.append(task_key)
        
        return remaining


def main():
    """主函數"""
    manager = ProgressManager()
    
    print("=== 進度管理器 ===")
    print("1. 顯示進度")
    print("2. 重置進度")
    print("3. 查看剩餘任務")
    
    choice = input("請選擇操作 (1-3): ").strip()
    
    if choice == "1":
        manager.show_progress()
    elif choice == "2":
        confirm = input("確定要重置進度嗎？(y/N): ").strip().lower()
        if confirm == 'y':
            manager.reset_progress()
    elif choice == "3":
        remaining = manager.get_remaining_tasks()
        print(f"剩餘任務數: {len(remaining)}")
        if remaining:
            print("前10個剩餘任務:")
            for task in remaining[:10]:
                print(f"  {task}")
    else:
        print("無效選擇")


if __name__ == "__main__":
    main()