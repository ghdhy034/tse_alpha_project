"""
TSE Alpha 雙硬體環境配置
自動檢測硬體並選擇對應配置
"""
import torch
from dataclasses import dataclass
from typing import Dict, Any
import psutil
import logging

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """硬體配置基類"""
    name: str
    gpu_memory_gb: float
    batch_size: int
    seq_len: int
    n_stocks: int
    num_epochs: int
    accumulate_grad_batches: int
    precision: str
    data_subset_ratio: float
    num_workers: int
    pin_memory: bool


class HardwareDetector:
    """硬體檢測器"""
    
    @staticmethod
    def detect_gpu() -> Dict[str, Any]:
        """檢測 GPU 配置"""
        if not torch.cuda.is_available():
            return {
                'available': False,
                'name': 'CPU',
                'memory_gb': 0,
                'compute_capability': None
            }
        
        gpu_props = torch.cuda.get_device_properties(0)
        return {
            'available': True,
            'name': gpu_props.name,
            'memory_gb': gpu_props.total_memory / 1e9,
            'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
        }
    
    @staticmethod
    def detect_system_memory() -> float:
        """檢測系統記憶體 (GB)"""
        return psutil.virtual_memory().total / 1e9
    
    @staticmethod
    def get_hardware_profile() -> str:
        """獲取硬體配置檔案"""
        gpu_info = HardwareDetector.detect_gpu()
        
        if not gpu_info['available']:
            return 'cpu'
        
        gpu_memory = gpu_info['memory_gb']
        gpu_name = gpu_info['name'].lower()
        
        # RTX 4090 (24GB) - 生產環境
        if '4090' in gpu_name or gpu_memory > 20:
            return 'rtx4090'
        
        # GTX 1660 Ti (6GB) - 開發環境
        elif '1660' in gpu_name or (4 < gpu_memory <= 8):
            return 'gtx1660ti'
        
        # 其他 GPU
        elif gpu_memory > 8:
            return 'high_end'
        else:
            return 'low_end'


class ConfigManager:
    """配置管理器"""
    
    # 預定義配置
    CONFIGS = {
        'gtx1660ti': HardwareConfig(
            name='GTX 1660 Ti (開發/測試)',
            gpu_memory_gb=6.0,
            batch_size=8,                    # 保守配置
            seq_len=32,                      # 減半序列長度
            n_stocks=10,                     # 少量股票快速驗證
            num_epochs=5,                    # 煙霧測試輪數
            accumulate_grad_batches=8,       # 等效 batch=64
            precision='16-mixed',            # 混合精度節省 VRAM
            data_subset_ratio=0.2,           # 20% 資料子集
            num_workers=2,                   # 較少工作進程
            pin_memory=False                 # 節省記憶體
        ),
        
        'rtx4090': HardwareConfig(
            name='RTX 4090 (生產/訓練)',
            gpu_memory_gb=24.0,
            batch_size=128,                  # 最大化批次
            seq_len=64,                      # 完整序列長度
            n_stocks=180,                    # 完整股票池
            num_epochs=150,                  # 充分訓練
            accumulate_grad_batches=1,       # 無需梯度累積
            precision='16-mixed',            # 混合精度優化性能
            data_subset_ratio=1.0,           # 完整資料集
            num_workers=8,                   # 充分利用 CPU
            pin_memory=True                  # 加速資料傳輸
        ),
        
        'high_end': HardwareConfig(
            name='高階 GPU (通用)',
            gpu_memory_gb=12.0,
            batch_size=64,
            seq_len=64,
            n_stocks=100,
            num_epochs=100,
            accumulate_grad_batches=2,
            precision='16-mixed',
            data_subset_ratio=0.8,
            num_workers=4,
            pin_memory=True
        ),
        
        'low_end': HardwareConfig(
            name='低階 GPU (相容)',
            gpu_memory_gb=4.0,
            batch_size=4,
            seq_len=32,
            n_stocks=5,
            num_epochs=3,
            accumulate_grad_batches=16,
            precision='16-mixed',
            data_subset_ratio=0.1,
            num_workers=1,
            pin_memory=False
        ),
        
        'cpu': HardwareConfig(
            name='CPU Only (調試)',
            gpu_memory_gb=0.0,
            batch_size=2,
            seq_len=16,
            n_stocks=3,
            num_epochs=1,
            accumulate_grad_batches=32,
            precision='32-true',
            data_subset_ratio=0.05,
            num_workers=1,
            pin_memory=False
        )
    }
    
    @classmethod
    def get_auto_config(cls) -> HardwareConfig:
        """自動檢測並返回適合的配置"""
        profile = HardwareDetector.get_hardware_profile()
        config = cls.CONFIGS[profile]
        
        # 記錄檢測結果
        gpu_info = HardwareDetector.detect_gpu()
        system_memory = HardwareDetector.detect_system_memory()
        
        logger.info(f"硬體檢測結果:")
        logger.info(f"  GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
        logger.info(f"  系統記憶體: {system_memory:.1f}GB")
        logger.info(f"  選擇配置: {config.name}")
        logger.info(f"  批次大小: {config.batch_size}")
        logger.info(f"  序列長度: {config.seq_len}")
        logger.info(f"  股票數量: {config.n_stocks}")
        
        return config
    
    @classmethod
    def get_config(cls, profile: str) -> HardwareConfig:
        """獲取指定配置"""
        if profile not in cls.CONFIGS:
            raise ValueError(f"未知配置檔案: {profile}. 可用: {list(cls.CONFIGS.keys())}")
        return cls.CONFIGS[profile]
    
    @classmethod
    def list_configs(cls) -> Dict[str, str]:
        """列出所有可用配置"""
        return {profile: config.name for profile, config in cls.CONFIGS.items()}


def create_training_config(hardware_profile: str = 'auto') -> Dict[str, Any]:
    """創建訓練配置"""
    if hardware_profile == 'auto':
        hw_config = ConfigManager.get_auto_config()
    else:
        hw_config = ConfigManager.get_config(hardware_profile)
    
    return {
        # 硬體配置
        'hardware_profile': hw_config.name,
        'batch_size': hw_config.batch_size,
        'accumulate_grad_batches': hw_config.accumulate_grad_batches,
        'precision': hw_config.precision,
        'num_workers': hw_config.num_workers,
        'pin_memory': hw_config.pin_memory,
        
        # 模型配置
        'sequence_length': hw_config.seq_len,
        'n_stocks': hw_config.n_stocks,
        
        # 訓練配置
        'num_epochs': hw_config.num_epochs,
        'data_subset_ratio': hw_config.data_subset_ratio,
        
        # 學習率 (根據批次大小調整)
        'learning_rate': 1e-4 * (hw_config.batch_size * hw_config.accumulate_grad_batches / 64),
        
        # 其他配置
        'early_stopping_patience': max(5, hw_config.num_epochs // 10),
        'save_top_k': 3 if hw_config.gpu_memory_gb > 10 else 1,
        'log_every_n_steps': max(1, hw_config.batch_size // 8)
    }


def create_smoke_test_config() -> Dict[str, Any]:
    """創建煙霧測試專用配置 (強制低配置)"""
    return {
        'hardware_profile': '煙霧測試',
        'batch_size': 4,
        'accumulate_grad_batches': 2,
        'precision': '16-mixed',
        'num_workers': 1,
        'pin_memory': False,
        'sequence_length': 16,
        'n_stocks': 3,
        'num_epochs': 2,
        'data_subset_ratio': 0.01,  # 1% 資料
        'learning_rate': 1e-3,      # 較高學習率快速收斂
        'early_stopping_patience': 1,
        'save_top_k': 1,
        'log_every_n_steps': 1
    }


if __name__ == "__main__":
    # 測試硬體檢測
    print("=== TSE Alpha 硬體配置檢測 ===")
    
    # 檢測硬體
    gpu_info = HardwareDetector.detect_gpu()
    profile = HardwareDetector.get_hardware_profile()
    
    print(f"GPU: {gpu_info}")
    print(f"檢測到配置檔案: {profile}")
    
    # 顯示所有可用配置
    print("\n可用配置:")
    for profile, name in ConfigManager.list_configs().items():
        print(f"  {profile}: {name}")
    
    # 自動配置
    print("\n自動選擇配置:")
    auto_config = ConfigManager.get_auto_config()
    print(f"  {auto_config}")
    
    # 訓練配置
    print("\n訓練配置:")
    train_config = create_training_config()
    for key, value in train_config.items():
        print(f"  {key}: {value}")
    
    # 煙霧測試配置
    print("\n煙霧測試配置:")
    smoke_config = create_smoke_test_config()
    for key, value in smoke_config.items():
        print(f"  {key}: {value}")