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
        self.y = torch.from_numpy(y).float() # 回归任务，Y也是float
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.x.shape[0]

def convert_to_floats(feature_col, memory):
    """
    处理缺失值：如果是 '?' 或空值，使用 memory 中的前一时刻值填充
    """
    for i in range(len(feature_col)):
        if feature_col[i] == "?" or feature_col[i] == "\n" or feature_col[i].strip() == "":
            feature_col[i] = memory[i]
        else:
            val = float(feature_col[i])
            feature_col[i] = val
            memory[i] = val
    return feature_col, memory

def load_raw_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    all_x = []
    
    # 原始逻辑：逐行读取并处理缺失值
    with open(file_path, "r") as f:
        lineno = -1
        # 初始 memory，假设前7列都有值（原代码逻辑如此）
        memory = [0.0 for _ in range(7)] 
        
        for line in f:
            lineno += 1
            if lineno == 0: # 跳过 Header
                continue
                
            arr = line.split(";")
            if len(arr) < 8:
                continue
            
            # 取第2列及之后的数据 (Date, Time 被忽略)
            feature_col = arr[2:]
            
            # 类型转换与缺失值填充
            feature_col, memory = convert_to_floats(feature_col, memory)
            
            all_x.append(np.array(feature_col, dtype=np.float32))

    all_x = np.stack(all_x, axis=0)

    # 归一化 (Normalization)
    # 注意：原代码是对所有列（包含 label）一起做归一化的
    all_x -= np.mean(all_x, axis=0)
    all_x /= (np.std(all_x, axis=0) + 1e-8)

    # 拆分 X 和 Y
    # column 0 is 'Global_active_power' (Target)
    all_y = all_x[:, 0].reshape([-1, 1])
    # columns 1-6 are features
    all_x = all_x[:, 1:]

    return all_x, all_y

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
# 2. PowerData 类
# ==========================================

class PowerData:
    def __init__(self, seq_len=32, batch_size=64):
        data_path = "data/power/household_power_consumption.txt"
        
        # 1. 加载、清洗、归一化
        # all_x: [TotalSteps, 6], all_y: [TotalSteps, 1]
        raw_x, raw_y = load_raw_data(data_path)
        
        # 2. 切分序列
        # 原代码 inc=seq_len，意味着是不重叠的窗口 (Non-overlapping windows)
        self.train_x, self.train_y = cut_in_sequences(raw_x, raw_y, seq_len, inc=seq_len)

        print(f"Power Data Loaded:")
        print(f"X shape: {self.train_x.shape}") # [Batch, SeqLen, 6]
        print(f"Y shape: {self.train_y.shape}") # [Batch, SeqLen, 1]

        total_seqs = self.train_x.shape[0]
        print(f"Total number of training sequences: {total_seqs}")
        
        # 3. 数据划分 (Train/Valid/Test)
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
        if os.path.exists("data/power/household_power_consumption.txt"):
            data = PowerData(seq_len=32, batch_size=16)
            for x, y in data.train_loader:
                print(f"Batch X shape: {x.shape}") # 预期: [16, 32, 6]
                print(f"Batch Y shape: {y.shape}") # 预期: [16, 32, 1]
                print(f"{y}")
                break
        else:
            print("Data file not found, skipping.")
    except Exception as e:
        print(f"Error: {e}")