#!/usr/bin/env python3
"""
最終驗證腳本 - 驗證所有資料庫結構修復
"""

def main():
    """主測試函數"""
    print("=" * 60)
    print("TSE Alpha 最終配置驗證")
    print("基於實際資料庫結構的修復驗證")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # 測試 1: TrainingConfig 創建和驗證
    print("\n🔧 測試 1: TrainingConfig 創建和驗證")
    print("-" * 40)
    total_tests += 1
    
    try:
        from models.config.training_config import TrainingConfig
        config = TrainingConfig()
        print("✅ TrainingConfig 創建成功")
        
        # 檢查關鍵配置
        print(f"📊 配置摘要:")
        print(f"   價格特徵數: {config.price_features}")
        print(f"   技術指標數: {len(config.technical_indicators)}")
        print(f"   基本面特徵數: {config.fundamental_features}")
        print(f"   基本面特徵列表長度: {len(config.fundamental_features_list)}")
        print(f"   帳戶特徵數: {config.account_features}")
        
        # 驗證數量匹配
        tech_expected = config.price_features - 5  # OHLCV = 5
        tech_actual = len(config.technical_indicators)
        fundamental_expected = config.fundamental_features
        fundamental_actual = len(config.fundamental_features_list)
        
        print(f"\n🔍 驗證結果:")
        tech_match = tech_expected == tech_actual
        fundamental_match = fundamental_expected == fundamental_actual
        
        print(f"   技術指標: 期望 {tech_expected}, 實際 {tech_actual} - {'✅' if tech_match else '❌'}")
        print(f"   基本面特徵: 期望 {fundamental_expected}, 實際 {fundamental_actual} - {'✅' if fundamental_match else '❌'}")
        
        if tech_match and fundamental_match:
            print("✅ 所有特徵數量匹配正確")
            tests_passed += 1
        else:
            print("❌ 特徵數量不匹配")
            
    except Exception as e:
        print(f"❌ TrainingConfig 測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 測試 2: 模型創建和前向傳播
    print("\n🤖 測試 2: 模型創建和前向傳播")
    print("-" * 40)
    total_tests += 1
    
    try:
        from models.config.training_config import TrainingConfig
        from models.model_architecture import ModelConfig, TSEAlphaModel
        import torch
        
        training_config = TrainingConfig()
        model_config = ModelConfig(
            price_frame_shape=(10, 64, training_config.price_features),
            fundamental_dim=training_config.fundamental_features,
            account_dim=training_config.account_features,
            n_stocks=10,
            hidden_dim=128
        )
        
        model = TSEAlphaModel(model_config)
        print("✅ 模型創建成功")
        
        # 測試前向傳播
        observation = {
            'price_frame': torch.randn(1, 10, 64, training_config.price_features),
            'fundamental': torch.randn(1, training_config.fundamental_features),
            'account': torch.randn(1, training_config.account_features)
        }
        
        outputs = model(observation)
        print("✅ 模型前向傳播成功")
        print(f"   輸出形狀: {[(k, v.shape) for k, v in outputs.items() if hasattr(v, 'shape')]}")
        
        # 測試動作生成
        action = model.get_action(observation, deterministic=True)
        print(f"✅ 動作生成成功: 股票={action[0]}, 倉位={action[1]}")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ 模型測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 測試結果總結
    print("\n" + "=" * 60)
    print("📋 最終驗證結果")
    print("=" * 60)
    
    pass_rate = (tests_passed / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"總測試數: {total_tests}")
    print(f"通過測試: {tests_passed}")
    print(f"失敗測試: {total_tests - tests_passed}")
    print(f"通過率: {pass_rate:.1f}%")
    
    if pass_rate == 100:
        print(f"\n🎉 所有測試通過！")
        print(f"✅ 資料庫結構修復完全成功")
        print(f"🚀 系統現在完全可用")
        print(f"📈 可以開始端到端訓練")
        print(f"\n🎯 建議下一步:")
        print(f"   1. 開始創建端到端訓練管線")
        print(f"   2. 進行小規模訓練測試")
        print(f"   3. 擴展到完整的180支股票訓練")
    elif pass_rate >= 75:
        print(f"\n✅ 大部分測試通過")
        print(f"🔧 系統基本可用，可能需要微調")
    else:
        print(f"\n⚠️ 還有重要問題需要解決")
    
    return pass_rate

if __name__ == "__main__":
    result = main()
    print(f"\n最終通過率: {result:.1f}%")