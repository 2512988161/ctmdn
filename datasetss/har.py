import os
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

def load_raw_data(prefix):
    """
    辅助函数：加载 UCI HAR 的 txt 文件
    """
    x_path = os.path.join("data/har/UCI HAR Dataset", prefix, f"X_{prefix}.txt")
    y_path = os.path.join("data/har/UCI HAR Dataset", prefix, f"y_{prefix}.txt")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Data files not found: {x_path} or {y_path}")

    print(f"Loading {prefix} data...")
    x = np.loadtxt(x_path)
    # 原始标签是 1-6，转换为 0-5 以适配 PyTorch
    y = (np.loadtxt(y_path) - 1).astype(np.int64) 
    
    return x, y

def cut_in_sequences(x, y, seq_len, inc=1):
    sequences_x = []
    sequences_y = []

    # 原始逻辑：只要长度够 seq_len 就切一片
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
# 2. HarData 类
# ==========================================

class HarData:
    def __init__(self, seq_len=16, batch_size=64):
        # 1. 加载原始数据
        # 这一步可能会比较慢，因为 np.loadtxt 读取大文本文件较慢
        raw_train_x, raw_train_y = load_raw_data("train")
        raw_test_x, raw_test_y = load_raw_data("test")

        # 2. 切分序列
        # 训练集 inc=1 (滑动窗口步长1)
        # 测试集 inc=8 (滑动窗口步长8，保留原代码逻辑)
        train_x, train_y = cut_in_sequences(raw_train_x, raw_train_y, seq_len, inc=1)
        test_x, test_y = cut_in_sequences(raw_test_x, raw_test_y, seq_len, inc=8)
        
        print(f"Total number of training sequences: {train_x.shape[0]}")

        # 3. 划分 Train/Valid
        # 使用原代码固定的随机种子 893429
        total_train = train_x.shape[0]
        rng = np.random.RandomState(893429)
        permutation = rng.permutation(total_train)
        
        valid_size = int(0.1 * total_train)
        print(f"Validation split: {valid_size}, Training split: {total_train - valid_size}")

        # 切片 (Batch-First 维度是 0)
        self.valid_x = train_x[permutation[:valid_size]]
        self.valid_y = train_y[permutation[:valid_size]]
        
        self.train_x = train_x[permutation[valid_size:]]
        self.train_y = train_y[permutation[valid_size:]]

        self.test_x = test_x
        self.test_y = test_y
        
        print(f"Total number of test sequences: {self.test_x.shape[0]}")

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
        # 检查数据目录是否存在，避免直接运行报错
        if os.path.exists("data/har/UCI HAR Dataset"):
            data = HarData(seq_len=32, batch_size=32)
            for x, y in data.train_loader:
                print(f"Batch X shape: {x.shape}") # 预期: [32, 16, 561] (Batch, Seq, Feats)
                print(f"Batch Y shape: {y.shape}") # 预期: [32, 16] (Dense labels)
                print(f"Unique labels in this batch: {torch.unique(y)}{y}")# 6类 
                break
        else:
            print("HAR Dataset path not found, skipping.")
    except Exception as e:
        print(f"Error: {e}")