#!/usr/bin/env python3
"""
股票分組管理器 - 處理三組別股票的平均分配
"""
import sys
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# 添加路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "market_data_collector"))

class StockGroupsManager:
    """股票分組管理器"""
    
    def __init__(self, stock_file: str = "stock_id.txt"):
        self.stock_file = stock_file
        self.groups = {}
        self.all_stocks = []
        self.load_stock_groups()
    
    def load_stock_groups(self):
        """從檔案載入股票分組"""
        try:
            with open(self.stock_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析三個組別 - 根據實際檔案格式
            lines = content.strip().split('\n')
            current_group = None
            in_group_definition = False
            
            for line in lines:
                line = line.strip()
                
                # 跳過空行
                if not line:
                    continue
                
                # 檢查組別標題
                if line.startswith('#') and ('group_A' in line or 'A.' in line or '半導體' in line):
                    current_group = 'group_A'
                    in_group_definition = False
                elif line.startswith('#') and ('group_B' in line or 'B.' in line or '傳產' in line):
                    current_group = 'group_B'
                    in_group_definition = False
                elif line.startswith('#') and ('group_C' in line or 'C.' in line or '金融' in line):
                    current_group = 'group_C'
                    in_group_definition = False
                
                # 檢查組別定義開始
                elif line.startswith('group_A =') or line.startswith('group_B =') or line.startswith('group_C ='):
                    if 'group_A' in line:
                        current_group = 'group_A'
                    elif 'group_B' in line:
                        current_group = 'group_B'
                    elif 'group_C' in line:
                        current_group = 'group_C'
                    
                    if current_group not in self.groups:
                        self.groups[current_group] = []
                    
                    in_group_definition = True
                    
                    # 處理同一行的股票代號
                    if '[' in line:
                        stock_part = line.split('[')[1] if '[' in line else line
                        self._extract_stocks_from_line(stock_part, current_group)
                
                # 處理組別內的股票代號
                elif in_group_definition and current_group:
                    self._extract_stocks_from_line(line, current_group)
                    
                    # 檢查是否結束
                    if ']' in line:
                        in_group_definition = False
            
            # 移除重複並統計
            for group in self.groups:
                self.groups[group] = list(set(self.groups[group]))
                self.all_stocks.extend(self.groups[group])
            
            self.all_stocks = list(set(self.all_stocks))
            
            print(f"✅ 載入股票分組:")
            for group, stocks in self.groups.items():
                group_name = {
                    'group_A': 'A. 半導體‧電子供應鏈',
                    'group_B': 'B. 傳產／原物料＆運輸', 
                    'group_C': 'C. 金融‧內需消費／綠能生技'
                }.get(group, group)
                print(f"   {group_name}: {len(stocks)} 支股票")
                print(f"     前10支: {', '.join(stocks[:10])}")
            print(f"   總計: {len(self.all_stocks)} 支股票")
            
        except Exception as e:
            print(f"❌ 載入股票分組失敗: {e}")
            import traceback
            traceback.print_exc()
            # 使用預設分組
            self._create_default_groups()
    
    def _extract_stocks_from_line(self, line: str, current_group: str):
        """從行中提取股票代號"""
        # 移除括號、引號等符號
        line = line.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
        line = line.replace('(', '').replace(')', '').replace('#', '')
        
        # 分割股票代號
        stocks = line.replace(',', ' ').split()
        
        for stock in stocks:
            # 清理股票代號
            clean_stock = ''.join(filter(str.isdigit, stock))
            if len(clean_stock) == 4:  # 台股代號是4位數
                self.groups[current_group].append(clean_stock)
    
    def _create_default_groups(self):
        """創建預設分組（如果檔案讀取失敗）"""
        print("使用預設股票分組...")
        
        # 根據 stock_id.txt 的實際內容創建預設分組
        self.groups = {
            'group_A': [
                "2330","2317","2454","2303","2408","2412","2382","2357","2379","3034",
                "3008","4938","2449","2383","2356","3006","3661","2324","8046","3017",
                "6121","3037","3014","3035","3062","3030","3529","5443","2337","8150",
                "3293","3596","2344","2428","2345","2338","6202","5347","3673","3105",
                "6231","6669","4961","4967","6668","4960","3528","6147","3526","6547",
                "8047","3227","4968","5274","6415","6414","6770","2331","6290","2342"
            ],
            'group_B': [
                "2603","2609","2615","2610","2618","2637","2606","2002","2014","2027",
                "2201","1201","1216","1301","1303","1326","1710","1717","1722","1723",
                "1402","1409","1434","1476","2006","2049","2105","2106","2107","1605",
                "1609","1608","1612","2308","1727","1730","1101","1102","1108","1210",
                "1215","1802","1806","1810","1104","1313","1314","1310","5608","5607",
                "8105","8940","5534","5609","5603","2023","2028","2114","9933","2501"
            ],
            'group_C': [
                "2880","2881","2882","2883","2884","2885","2886","2887","2888","2890",
                "2891","2892","2812","2823","2834","2850","2801","2836","2845","4807",
                "3702","3706","4560","8478","4142","4133","6525","6548","6843","1513",
                "1514","1516","1521","1522","1524","1533","1708","3019","5904","5906",
                "5902","6505","6806","6510","2207","2204","2231","1736","4105","4108",
                "4162","1909","1702","9917","1217","1218","1737","1783","3708","1795"
            ]
        }
        
        self.all_stocks = []
        for stocks in self.groups.values():
            self.all_stocks.extend(stocks)
    
    def get_balanced_split(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, 
                          random_seed: int = 42) -> Dict[str, List[str]]:
        """
        獲取平衡的訓練/驗證/測試集分割
        確保每個組別的股票在三個集合中平均分布
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 檢查比例總和
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("訓練、驗證、測試比例總和必須為1.0")
        
        train_stocks = []
        val_stocks = []
        test_stocks = []
        
        print(f"📊 分割比例: 訓練 {train_ratio:.1%}, 驗證 {val_ratio:.1%}, 測試 {test_ratio:.1%}")
        
        for group_name, stocks in self.groups.items():
            # 隨機打亂該組股票
            group_stocks = stocks.copy()
            random.shuffle(group_stocks)
            
            n_stocks = len(group_stocks)
            n_train = int(n_stocks * train_ratio)
            n_val = int(n_stocks * val_ratio)
            n_test = n_stocks - n_train - n_val  # 剩餘的分配給測試集
            
            # 分配股票
            train_stocks.extend(group_stocks[:n_train])
            val_stocks.extend(group_stocks[n_train:n_train + n_val])
            test_stocks.extend(group_stocks[n_train + n_val:])
            
            print(f"   {group_name}: {n_train} 訓練, {n_val} 驗證, {n_test} 測試 (總計 {n_stocks})")
        
        # 再次隨機打亂各集合
        random.shuffle(train_stocks)
        random.shuffle(val_stocks)
        random.shuffle(test_stocks)
        
        result = {
            'train': train_stocks,
            'validation': val_stocks,
            'test': test_stocks
        }
        
        print(f"\n✅ 分割結果:")
        print(f"   訓練集: {len(train_stocks)} 支股票")
        print(f"   驗證集: {len(val_stocks)} 支股票")
        print(f"   測試集: {len(test_stocks)} 支股票")
        print(f"   總計: {len(train_stocks) + len(val_stocks) + len(test_stocks)} 支股票")
        
        return result
    
    def verify_group_balance(self, split_result: Dict[str, List[str]]) -> Dict:
        """驗證分割結果中各組別的平衡性"""
        balance_report = {}
        
        for split_name, stocks in split_result.items():
            group_counts = {group: 0 for group in self.groups.keys()}
            
            for stock in stocks:
                for group_name, group_stocks in self.groups.items():
                    if stock in group_stocks:
                        group_counts[group_name] += 1
                        break
            
            balance_report[split_name] = group_counts
        
        print(f"\n📊 組別平衡性驗證:")
        for split_name, group_counts in balance_report.items():
            total = sum(group_counts.values())
            print(f"   {split_name}:")
            for group, count in group_counts.items():
                percentage = count / total * 100 if total > 0 else 0
                print(f"     {group}: {count} 支 ({percentage:.1f}%)")
        
        return balance_report
    
    def save_split_config(self, split_result: Dict[str, List[str]], filename: str = "stock_split_config.json"):
        """儲存分割配置"""
        config = {
            'groups': self.groups,
            'split': split_result,
            'metadata': {
                'total_stocks': len(self.all_stocks),
                'group_counts': {group: len(stocks) for group, stocks in self.groups.items()},
                'split_counts': {split: len(stocks) for split, stocks in split_result.items()}
            }
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 分割配置已儲存到: {filename}")
            
        except Exception as e:
            print(f"❌ 儲存配置失敗: {e}")
    
    def load_split_config(self, filename: str = "stock_split_config.json") -> Dict:
        """載入分割配置"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"✅ 從 {filename} 載入分割配置")
            return config
            
        except Exception as e:
            print(f"❌ 載入配置失敗: {e}")
            return {}
    
    def update_data_collector_config(self):
        """更新資料收集器配置以使用新的股票清單"""
        try:
            # 更新增強版資料收集器
            collector_file = "tmp_rovodev_enhanced_data_collector.py"
            
            if os.path.exists(collector_file):
                with open(collector_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 替換股票清單
                new_stock_list = str(self.all_stocks)
                
                # 找到並替換 get_full_stock_list 方法中的股票清單
                import re
                pattern = r'top_180_stocks = \[.*?\]'
                replacement = f'top_180_stocks = {new_stock_list}'
                
                if re.search(pattern, content, re.DOTALL):
                    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                    
                    with open(collector_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"✅ 已更新資料收集器股票清單: {len(self.all_stocks)} 支股票")
                else:
                    print("⚠️  未找到股票清單定義，請手動更新")
            
            # 更新配置檔案
            config_file = Path("market_data_collector/utils/config.py")
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 替換 STOCK_IDS
                pattern = r'STOCK_IDS\s*=\s*\[.*?\]'
                replacement = f'STOCK_IDS = {new_stock_list}'
                
                if re.search(pattern, content, re.DOTALL):
                    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                else:
                    content += f"\n\n# 更新的股票清單\nSTOCK_IDS = {new_stock_list}\n"
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ 已更新配置檔案股票清單")
            
        except Exception as e:
            print(f"❌ 更新配置失敗: {e}")


def main():
    """主函數"""
    print("=== 股票分組管理器 ===")
    
    # 創建管理器
    manager = StockGroupsManager()
    
    if not manager.groups:
        print("❌ 無法載入股票分組")
        return
    
    # 顯示分組資訊
    print(f"\n📋 股票分組詳情:")
    for group_name, stocks in manager.groups.items():
        print(f"\n{group_name}:")
        print(f"  股票數量: {len(stocks)}")
        print(f"  股票清單: {', '.join(stocks[:10])}" + ("..." if len(stocks) > 10 else ""))
    
    # 生成平衡分割
    print(f"\n🔄 生成平衡分割...")
    split_result = manager.get_balanced_split(
        train_ratio=0.7,
        val_ratio=0.15, 
        test_ratio=0.15,
        random_seed=42
    )
    
    # 驗證平衡性
    manager.verify_group_balance(split_result)
    
    # 儲存配置
    manager.save_split_config(split_result)
    
    # 更新相關配置檔案
    print(f"\n🔧 更新系統配置...")
    manager.update_data_collector_config()
    
    print(f"\n🎉 股票分組管理完成！")
    print(f"💡 後續訓練時請使用 stock_split_config.json 中的分割配置")


if __name__ == "__main__":
    main()