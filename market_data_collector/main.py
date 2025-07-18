# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.config import *

def update_data():
    print("=== 資料更新開始 ===")
    import fetch_data.fetch_historical_data as fd
    fd.main()
    print("=== 資料更新完成 ===")



if __name__ == "__main__":
    print("========== 主流程開始 ==========")
    update_data()
    print("========== 主流程結束 ==========")
