import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 辅助类与函数
# ==========================================

class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.x.shape[0]
    
class SequenceDatasetRadom(Dataset):
    def __init__(self, x, y, random_ratio=0.0):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.random_ratio = random_ratio
        
    def __getitem__(self, index):
        # 随机用-1mask掉random_ratio比例seq_len的元素
        # if self.random_ratio > 0:
        #     mask = torch.rand(self.x[index].shape[0]) < self.random_ratio
        #     self.x[index][mask] = -1
        # return self.x[index], self.y[index]
        x = self.x[index].clone()
        if self.random_ratio > 0:
            mask = torch.rand(x.shape[0]) < self.random_ratio
            x[mask] = -1
        return x, self.y[index]
        # return x, self.y[index], self.x[index]
    # def __getitem__(self, index):
    #     x = self.x[index].clone()   # [seq_len, feat]
    #     seq_len = x.shape[0]

    #     if self.random_ratio > 0:
    #         # 1. 总共要 mask 的长度
    #         total_mask_len = int(seq_len * self.random_ratio)
    #         total_mask_len = max(1, total_mask_len)

    #         # 2. 每段长度范围（你可以调）
    #         min_span = max(1, total_mask_len // 4)
    #         max_span = max(min_span, total_mask_len // 2)

    #         masked = 0
    #         mask = torch.zeros(seq_len, dtype=torch.bool)

    #         # 3. 生成若干连续区间
    #         while masked < total_mask_len:
    #             span_len = torch.randint(min_span, max_span + 1, (1,)).item()
    #             start = torch.randint(0, seq_len - span_len + 1, (1,)).item()

    #             end = min(start + span_len, seq_len)
    #             newly_masked = (~mask[start:end]).sum().item()

    #             mask[start:end] = True
    #             masked += newly_masked

    #         # 4. 整段 mask
    #         x[mask] = -1

    #     return x, self.y[index], self.x[index]
    def __len__(self):
        return self.x.shape[0]
    
def read_file(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    df = pd.read_csv(filename)
    
    # 提取特征
    data_x = np.stack([
        df['Temperature'].values,
        df['Humidity'].values,
        df['Light'].values,
        df['CO2'].values,
        df['HumidityRatio'].values,
    ], axis=-1)
    
    # 提取标签 (Occupancy 0 or 1)
    data_y = df['Occupancy'].values.astype(np.int64)
    
    return data_x, data_y

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
    return np.stack(sequences_x, axis=0), np.stack(sequences_y, axis=0)[:, -1]

# ==========================================
# 2. OccupancyData 类
# ==========================================
class OccupancyDataRandom:
    def __init__(self, seq_len=16, batch_size=64, random_ratio=0.0):
        # 基础路径
        data_dir = "data/occupancy"
        print("OccupancyDataRandom",random_ratio)
        # 1. 读取原始数据
        train_x, train_y = read_file(os.path.join(data_dir, "datatraining.txt"))
        test0_x, test0_y = read_file(os.path.join(data_dir, "datatest.txt"))
        test1_x, test1_y = read_file(os.path.join(data_dir, "datatest2.txt"))

        # 2. 归一化 (基于训练集统计量)
        mean_x = np.mean(train_x, axis=0)
        std_x = np.std(train_x, axis=0)
        
        train_x = (train_x - mean_x) / (std_x + 1e-8)
        test0_x = (test0_x - mean_x) / (std_x + 1e-8)
        test1_x = (test1_x - mean_x) / (std_x + 1e-8)

        # 3. 切分序列
        # 注意：Train inc=1, Test inc=8 (保持原逻辑)
        train_x, train_y = cut_in_sequences(train_x, train_y, seq_len,  inc=1)
        test0_x, test0_y = cut_in_sequences(test0_x, test0_y, seq_len,  inc=8)
        test1_x, test1_y = cut_in_sequences(test1_x, test1_y, seq_len,  inc=8)

        print(f"Total number of training sequences: {train_x.shape[0]}")

        # 4. 划分 Train/Valid
        # 使用相同的种子 893429
        # 注意：现在 shape[0] 是 batch 维，shape[1] 是 seq_len
        total_train = train_x.shape[0]
        permutation = np.random.RandomState(893429).permutation(total_train)
        valid_size = int(0.1 * total_train)
        
        print(f"Validation split: {valid_size}, Training split: {total_train - valid_size}")

        self.valid_x = train_x[permutation[:valid_size]]
        self.valid_y = train_y[permutation[:valid_size]]
        self.train_x = train_x[permutation[valid_size:]]
        self.train_y = train_y[permutation[valid_size:]]

        # 5. 合并测试集
        # 注意：Batch-First 拼接使用 axis=0
        self.test_x = np.concatenate([test0_x, test1_x], axis=0)
        self.test_y = np.concatenate([test0_y, test1_y], axis=0)
        
        print(f"Total number of test sequences: {self.test_x.shape[0]}")

        # 6. 创建 DataLoader
        self.train_loader = DataLoader(
            SequenceDatasetRadom(self.train_x, self.train_y, random_ratio=random_ratio), 
            batch_size=batch_size, 
            shuffle=True
        )
        self.valid_loader = DataLoader(
            SequenceDatasetRadom(self.valid_x, self.valid_y, random_ratio=random_ratio), 
            batch_size=batch_size, 
            shuffle=False
        )
        self.test_loader = DataLoader(
            SequenceDatasetRadom(self.test_x, self.test_y, random_ratio=random_ratio), 
            batch_size=batch_size, 
            shuffle=False
        )
        
        
class OccupancyData:
    def __init__(self, seq_len=16, batch_size=64):
        # 基础路径
        data_dir = "data/occupancy"
        
        # 1. 读取原始数据
        train_x, train_y = read_file(os.path.join(data_dir, "datatraining.txt"))
        test0_x, test0_y = read_file(os.path.join(data_dir, "datatest.txt"))
        test1_x, test1_y = read_file(os.path.join(data_dir, "datatest2.txt"))

        # 2. 归一化 (基于训练集统计量)
        mean_x = np.mean(train_x, axis=0)
        std_x = np.std(train_x, axis=0)
        
        train_x = (train_x - mean_x) / (std_x + 1e-8)
        test0_x = (test0_x - mean_x) / (std_x + 1e-8)
        test1_x = (test1_x - mean_x) / (std_x + 1e-8)

        # 3. 切分序列
        # 注意：Train inc=1, Test inc=8 (保持原逻辑)
        train_x, train_y = cut_in_sequences(train_x, train_y, seq_len, inc=1)
        test0_x, test0_y = cut_in_sequences(test0_x, test0_y, seq_len, inc=8)
        test1_x, test1_y = cut_in_sequences(test1_x, test1_y, seq_len, inc=8)

        print(f"Total number of training sequences: {train_x.shape[0]}")

        # 4. 划分 Train/Valid
        # 使用相同的种子 893429
        # 注意：现在 shape[0] 是 batch 维，shape[1] 是 seq_len
        total_train = train_x.shape[0]
        permutation = np.random.RandomState(893429).permutation(total_train)
        valid_size = int(0.1 * total_train)
        
        print(f"Validation split: {valid_size}, Training split: {total_train - valid_size}")

        self.valid_x = train_x[permutation[:valid_size]]
        self.valid_y = train_y[permutation[:valid_size]]
        self.train_x = train_x[permutation[valid_size:]]
        self.train_y = train_y[permutation[valid_size:]]

        # 5. 合并测试集
        # 注意：Batch-First 拼接使用 axis=0
        self.test_x = np.concatenate([test0_x, test1_x], axis=0)
        self.test_y = np.concatenate([test0_y, test1_y], axis=0)
        
        print(f"Total number of test sequences: {self.test_x.shape[0]}")

        # 6. 创建 DataLoader
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
        data = OccupancyData(seq_len=1600, batch_size=32)
        for x, y in data.train_loader:
            print(f"Batch X shape: {x.shape}") # 预期: [32, 16, 5] (Batch, Seq, Feats)
            print(f"Batch Y shape: {y.shape}") # 预期: [32, 16]
            print(f"Unique labels: {y.unique()}{y}")
            break
    except FileNotFoundError as e:
        print(f"Error: {e}")