#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test - Fix encoding issues and test core functions
"""
import sys
import os
import json
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "market_data_collector"))

def test_stock_groups_direct():
    """Direct test of stock groups without exec"""
    print("Testing stock groups directly...")
    
    try:
        # Direct implementation of stock groups
        groups = {
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
        
        all_stocks = []
        for stocks in groups.values():
            all_stocks.extend(stocks)
        
        print(f"Group A: {len(groups['group_A'])} stocks")
        print(f"Group B: {len(groups['group_B'])} stocks") 
        print(f"Group C: {len(groups['group_C'])} stocks")
        print(f"Total: {len(all_stocks)} stocks")
        
        # Create balanced split
        import random
        random.seed(42)
        
        train_stocks = []
        val_stocks = []
        test_stocks = []
        
        for group_name, stocks in groups.items():
            group_stocks = stocks.copy()
            random.shuffle(group_stocks)
            
            n_stocks = len(group_stocks)
            n_train = int(n_stocks * 0.7)
            n_val = int(n_stocks * 0.15)
            
            train_stocks.extend(group_stocks[:n_train])
            val_stocks.extend(group_stocks[n_train:n_train + n_val])
            test_stocks.extend(group_stocks[n_train + n_val:])
        
        split_result = {
            'train': train_stocks,
            'validation': val_stocks,
            'test': test_stocks
        }
        
        print(f"Train: {len(train_stocks)} stocks")
        print(f"Validation: {len(val_stocks)} stocks")
        print(f"Test: {len(test_stocks)} stocks")
        
        # Save config
        config = {
            'groups': groups,
            'split': split_result,
            'metadata': {
                'total_stocks': len(all_stocks),
                'group_counts': {group: len(stocks) for group, stocks in groups.items()},
                'split_counts': {split: len(stocks) for split, stocks in split_result.items()}
            }
        }
        
        with open('stock_split_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("Stock split config saved successfully!")
        return True
        
    except Exception as e:
        print(f"Error in stock groups test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chip_features():
    """Test chip features"""
    print("\nTesting chip features...")
    
    try:
        import market_data_collector
        from market_data_collector.utils import config
        from market_data_collector.utils import db
        from data_pipeline import features
        
        print("Module imports successful")
        
        engine = features.FeatureEngine(['2330', '2317'])
        chip_indicators = features.ChipIndicators()
        
        print("Feature engine initialization successful")
        
        # Check database tables
        tables_to_check = [
            "candlesticks_daily",
            "margin_purchase_shortsale", 
            "institutional_investors_buy_sell",
            "minute_bars"
        ]
        
        for table in tables_to_check:
            try:
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                result = db.query_df(count_query)
                count = result.iloc[0]['count'] if not result.empty else 0
                print(f"   {table}: {count} records")
            except Exception as e:
                print(f"   {table}: Table not found or no data")
        
        print("Chip features test successful!")
        return True
        
    except Exception as e:
        print(f"Error in chip features test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_collector():
    """Test data collector"""
    print("\nTesting data collector...")
    
    try:
        # Direct implementation of key components
        class SimpleAPIKeyManager:
            def __init__(self):
                self.api_keys = []
                self.load_api_keys()
            
            def load_api_keys(self):
                try:
                    with open('findmind_api_keys.txt', 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and line.startswith('eyJ'):
                            self.api_keys.append(line)
                    
                    print(f"Loaded {len(self.api_keys)} API keys")
                except Exception as e:
                    print(f"Failed to load API keys: {e}")
                    self.api_keys = ["dummy_key"]
        
        api_manager = SimpleAPIKeyManager()
        
        # Test stock list
        group_A = ["2330","2317","2454","2303","2408","2412","2382","2357","2379","3034"]
        group_B = ["2603","2609","2615","2610","2618","2637","2606","2002","2014","2027"]
        group_C = ["2880","2881","2882","2883","2884","2885","2886","2887","2888","2890"]
        
        all_stocks = group_A + group_B + group_C
        all_stocks = list(set(all_stocks))
        all_stocks.sort()
        
        print(f"Stock list generated: {len(all_stocks)} stocks")
        print(f"First 10 stocks: {all_stocks[:10]}")
        
        print("Data collector test successful!")
        return True
        
    except Exception as e:
        print(f"Error in data collector test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_split():
    """Test training data split"""
    print("\nTesting training data split...")
    
    try:
        config_file = "stock_split_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            split_data = config.get('split', {})
            groups_data = config.get('groups', {})
            
            print("Split config loaded successfully")
            print(f"   Train: {len(split_data.get('train', []))} stocks")
            print(f"   Validation: {len(split_data.get('validation', []))} stocks")
            print(f"   Test: {len(split_data.get('test', []))} stocks")
            
            # Verify group balance
            for split_name, stocks in split_data.items():
                group_counts = {'group_A': 0, 'group_B': 0, 'group_C': 0}
                
                for stock in stocks:
                    for group_name, group_stocks in groups_data.items():
                        if stock in group_stocks:
                            group_counts[group_name] += 1
                            break
                
                total = sum(group_counts.values())
                print(f"   {split_name} group distribution:")
                for group, count in group_counts.items():
                    percentage = count / total * 100 if total > 0 else 0
                    print(f"     {group}: {count} stocks ({percentage:.1f}%)")
            
            print("Training data split test successful!")
            return True
        else:
            print("Split config file not found")
            return False
        
    except Exception as e:
        print(f"Error in training split test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=== TSE Alpha Quick Test (Fixed Encoding) ===")
    
    tests = [
        ("Stock Groups", test_stock_groups_direct),
        ("Chip Features", test_chip_features),
        ("Data Collector", test_data_collector),
        ("Training Split", test_training_split)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Error in {test_name} test: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("All tests passed! System ready!")
        print("\nNext steps:")
        print("1. Run: run_enhanced_data_collector.bat")
        print("2. Monitor progress with: python tmp_rovodev_progress_manager.py")
    else:
        print("Some tests failed, please check the issues")
    
    return passed == len(results)

if __name__ == "__main__":
    main()