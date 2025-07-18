# gym_env/__init__.py
"""
TSE Alpha Gymnasium Trading Environment
台股量化交易環境模組
"""
__version__ = "1.0.0"

from gymnasium.envs.registration import register

# 註冊自定義環境
register(
    id='TSEAlpha-v0',
    entry_point='gym_env.env:TSEAlphaEnv',
    max_episode_steps=10000,  # 最大步數限制
    kwargs={
        'max_holding_days': 15,
        'max_position_per_stock': 300,
        'daily_max_loss_pct': 0.02,
        'rolling_max_dd_pct': 0.10,
    }
)