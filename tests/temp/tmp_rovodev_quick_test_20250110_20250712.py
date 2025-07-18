#!/usr/bin/env python3
"""
快速測試腳本 - 驗證配置修復
"""

def quick_test():
    """快速測試配置是否正確"""
    
    print("=== TSE Alpha 快速配置測試 ===")
    
    try:
        # 測試 TrainingConfig
        from models.config.training_config import TrainingConfig
        config = TrainingConfig()
        print("✅ TrainingConfig 創建成功")
        
        print(f"📊 配置:")
        print(f"   技術指標數: {len(config.technical_indicators)}")
        print(f"   基本面特徵數: {config.fundamental_features}")
        print(f"   基本面列表長度: {len(config.fundamental_features_list)}")
        print(f"   價格特徵數: {config.price_features}")
        
        # 驗證匹配
        tech_match = len(config.technical_indicators) == (config.price_features - 5)
        fundamental_match = config.fundamental_features == len(config.fundamental_features_list)
        
        print(f"\n🔍 驗證:")
        print(f"   技術指標匹配: {'✅' if tech_match else '❌'}")
        print(f"   基本面特徵匹配: {'✅' if fundamental_match else '❌'}")
        
        if tech_match and fundamental_match:
            print(f"\n🎉 配置完全正確！")
            
            # 測試模型創建
            from models.model_architecture import ModelConfig, TSEAlphaModel
            import torch
            
            model_config = ModelConfig(
                price_frame_shape=(5, 32, config.price_features),
                fundamental_dim=config.fundamental_features,
                account_dim=config.account_features,
                n_stocks=5,
                hidden_dim=64
            )
            
            model = TSEAlphaModel(model_config)
            print("✅ 模型創建成功")
            
            # 測試前向傳播
            observation = {
                'price_frame': torch.randn(1, 5, 32, config.price_features),
                'fundamental': torch.randn(1, config.fundamental_features),
                'account': torch.randn(1, config.account_features)
            }
            
            outputs = model(observation)
            action = model.get_action(observation)
            
            print("✅ 模型運行成功")
            print(f"✅ 動作生成成功: {action}")
            print(f"\n🚀 系統完全可用！可以開始端到端訓練")
            return True
        else:
            print(f"\n❌ 配置仍有問題")
            return False
            
    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print(f"\n✅ 所有測試通過 - 系統就緒！")
    else:
        print(f"\n❌ 測試失敗 - 需要進一步調試")