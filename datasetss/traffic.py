import os
import numpy as np
import pandas as pd
import datetime as dt
import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 辅助类与函数
# ==========================================

class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float() # 回归任务
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.x.shape[0]

def load_trace(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    
    # 1. Holiday Feature
    # UCI数据集里非节假日标记为字符串 "None"。
    # 原代码逻辑 (df["holiday"].values == None) 比较模糊，这里修正为：如果不是 "None"，则是节假日(1.0)
    holiday = (df["holiday"].values != 'None').astype(np.float32)
    
    # 2. Temp Feature
    temp = df["temp"].values.astype(np.float32)
    temp -= np.mean(temp) # 减去均值 (Centering)
    
    # 3. Weather Features
    rain = df["rain_1h"].values.astype(np.float32)
    snow = df["snow_1h"].values.astype(np.float32)
    clouds = df["clouds_all"].values.astype(np.float32)
    
    # 4. Time Features
    date_time_str = df["date_time"].values
    # 解析时间: 2012-10-02 13:00:00
    # 列表推导式解析时间可能较慢，但保持原逻辑
    date_time = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_time_str]
    
    weekday = np.array([d.weekday() for d in date_time]).astype(np.float32)
    
    # Noon feature: sin(hour * pi / 24)
    # 0点=0, 12点=1, 24点=0
    hour = np.array([d.hour for d in date_time]).astype(np.float32)
    noon = np.sin(hour * np.pi / 24)

    # Stack Features [Samples, 7]
    features = np.stack([holiday, temp, rain, snow, clouds, weekday, noon], axis=-1)

    # 5. Target (Traffic Volume)
    traffic_volume = df["traffic_volume"].values.astype(np.float32)
    traffic_volume -= np.mean(traffic_volume)
    traffic_volume /= (np.std(traffic_volume) + 1e-8)

    return features, traffic_volume

def cut_in_sequences(x, y, seq_len, inc=1):
    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    if len(sequences_x) == 0:
        return np.array([]), np.array([])

    # 修改点：原代码 axis=1 (Time-Major), 这里改为 axis=0 (Batch-First)
    return np.stack(sequences_x, axis=0), np.stack(sequences_y, axis=0)

# ==========================================
# 2. TrafficData 类
# ==========================================

class TrafficData:
    def __init__(self, seq_len=32, batch_size=64):
        data_path = "data/traffic/Metro_Interstate_Traffic_Volume.csv"
        
        # 1. 加载和预处理
        raw_x, raw_y = load_trace(data_path)
        
        # 2. 切分序列
        # 原代码 inc=4
        self.train_x, self.train_y = cut_in_sequences(raw_x, raw_y, seq_len, inc=4)
        
        # 由于 Target 是 1D array，切分后 y shape 为 [Batch, SeqLen]，
        # 为了保持维度一致性，通常 reshape 为 [Batch, SeqLen, 1]
        if len(self.train_y.shape) == 2:
            self.train_y = self.train_y.reshape(self.train_y.shape[0], self.train_y.shape[1], 1)

        print(f"Traffic Data Loaded:")
        print(f"X shape: {self.train_x.shape}") # Expected: [Batch, SeqLen, 7]
        print(f"Y shape: {self.train_y.shape}") # Expected: [Batch, SeqLen, 1]

        total_seqs = self.train_x.shape[0]
        print(f"Total number of training sequences: {total_seqs}")
        
        # 3. 数据划分
        # 固定种子 23489
        rng = np.random.RandomState(23489)
        permutation = rng.permutation(total_seqs)
        
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)
        
        # 索引切片
        valid_indices = permutation[:valid_size]
        test_indices = permutation[valid_size:valid_size+test_size]
        train_indices = permutation[valid_size+test_size:]

        self.valid_x = self.train_x[valid_indices]
        self.valid_y = self.train_y[valid_indices]
        
        self.test_x = self.train_x[test_indices]
        self.test_y = self.train_y[test_indices]
        
        self.train_x = self.train_x[train_indices]
        self.train_y = self.train_y[train_indices]

        print(f"Train size: {self.train_x.shape[0]}")
        print(f"Valid size: {self.valid_x.shape[0]}")
        print(f"Test  size: {self.test_x.shape[0]}")

        # 4. 创建 DataLoader
        self.train_loader = DataLoader(
            SequenceDataset(self.train_x, self.train_y), 
            batch_size=batch_size, 
            shuffle=True
        )
        self.valid_loader = DataLoader(
            SequenceDataset(self.valid_x, self.valid_y), 
            batch_size=batch_size, 
            shuffle=False
        )
        self.test_loader = DataLoader(
            SequenceDataset(self.test_x, self.test_y), 
            batch_size=batch_size, 
            shuffle=False
        )

if __name__ == "__main__":
    # 测试代码
    try:
        # 确保数据文件存在
        if os.path.exists("data/traffic/Metro_Interstate_Traffic_Volume.csv"):
            data = TrafficData(seq_len=32, batch_size=1)
            for x, y in data.train_loader:
                print(f"Batch X shape: {x.shape}") # 预期: [32, 32, 7]
                print(f"Batch Y shape: {y.shape}") # 预期: [32, 32, 1]
                print(f"{y}")
                break
        else:
            print("Data file not found, skipping.")
    except Exception as e:
        print(f"Error: {e}")