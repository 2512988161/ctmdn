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

def to_float(v):
    if v == "?" or v.strip() == "":
        return 0.0
    else:
        return float(v)

def load_trace(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    all_x = []
    all_y = []
    
    miss = 0
    total = 0

    with open(file_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(',')
            
            # Ozone 数据集格式: Date, [Feats...], Label
            # 校验特征长度 (原代码检查74列)
            if len(parts) != 74:
                continue

            total += 1
            
            # 检查缺失值 '?'
            has_missing = False
            for i in range(1, len(parts)-1):
                if parts[i] == "?":
                    miss += 1
                    has_missing = True
                    break # 原代码统计逻辑是只要有缺失就算miss

            # 提取特征 (跳过第0列Date，取到倒数第2列)
            feats = [to_float(parts[i]) for i in range(1, len(parts)-1)]
            
            # 提取标签 (最后一列)
            label = int(float(parts[-1]))

            all_x.append(np.array(feats, dtype=np.float32))
            all_y.append(label)

    print(f"Missing features in {miss} out of {total} samples ({100*miss/total if total>0 else 0:.2f}%)")
    print(f"Read {len(all_x)} lines")
    
    all_x = np.stack(all_x, axis=0)
    all_y = np.array(all_y, dtype=np.int64)

    # 打印类别不平衡率
    print(f"Imbalance: {100*np.mean(all_y):.2f}% (Positive class ratio)")
    
    # 归一化 (Standardization)
    # 原代码使用 np.mean(all_x) (全局标量)，这里改为 axis=0 (按特征列归一化)，效果通常更好
    mean_x = np.mean(all_x, axis=0)
    std_x = np.std(all_x, axis=0)
    all_x = (all_x - mean_x) / (std_x + 1e-8)

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

    # 返回 Batch-First: [Batch, SeqLen, Feats]
    return np.stack(sequences_x, axis=0), np.stack(sequences_y, axis=0)

# ==========================================
# 2. OzoneData 类
# ==========================================

class OzoneData:
    def __init__(self, seq_len=32, batch_size=64):
        data_path = "data/ozone/eighthr.data"
        
        # 1. 加载和预处理
        raw_x, raw_y = load_trace(data_path)
        
        # 2. 切分序列
        # 原代码 inc=4
        all_x, all_y = cut_in_sequences(raw_x, raw_y, seq_len, inc=4)
        
        total_seqs = all_x.shape[0]
        print(f"Total number of training sequences: {total_seqs}")
        
        # 3. 数据划分
        # 使用固定种子 23489
        rng = np.random.RandomState(23489)
        permutation = rng.permutation(total_seqs)
        
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)
        
        # 索引切片
        valid_indices = permutation[:valid_size]
        test_indices = permutation[valid_size:valid_size+test_size]
        train_indices = permutation[valid_size+test_size:]

        self.valid_x = all_x[valid_indices]
        self.valid_y = all_y[valid_indices]
        
        self.test_x = all_x[test_indices]
        self.test_y = all_y[test_indices]
        
        self.train_x = all_x[train_indices]
        self.train_y = all_y[train_indices]

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
        # 注意：需要确保 data/ozone/eighthr.data 存在
        # 如果没有数据，这里会报错
        if os.path.exists("data/ozone/eighthr.data"):
            data = OzoneData(seq_len=32, batch_size=32)
            for x, y in data.train_loader:
                print(f"Batch X shape: {x.shape}") # 预期: [32, 32, 72] (Batch, Seq, Feats)
                print(f"Batch Y shape: {y.shape}") # 预期: [32, 32]
                print(f"Batch Y unique values: {np.unique(y)}{y}") # 预期: [0, 1]
                break
        else:
            print("Data file not found, skipping execution.")
    except Exception as e:
        print(f"Error: {e}")