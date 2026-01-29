import os

from tensorflow import data
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
# ==========================================
# 1. 数据预处理部分 (保持与原脚本逻辑一致)
# ==========================================

class_map = {
    'lying down': 0, 'lying': 0,
    'sitting down': 1, 'sitting': 1,
    'standing up from lying': 2, 'standing up from sitting': 2, 'standing up from sitting on the ground': 2,
    "walking": 3,
    "falling": 4,
    'on all fours': 5,
    'sitting on the ground': 6,
}

sensor_ids = {
    "010-000-024-033": 0,
    "010-000-030-096": 1,
    "020-000-033-111": 2,
    "020-000-032-221": 3
}

def one_hot(x, n):
    y = np.zeros(n, dtype=np.float32)
    y[x] = 1
    return y

def load_crappy_formated_csv():
    # 路径根据实际情况修改
    file_path = "data/person/ConfLongDemo_JSI.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

    all_x = []
    all_y = []

    series_x = []
    series_y = []

    all_feats = []
    
    with open(file_path, "r") as f:
        current_person = "A01"
        for line in f:
            arr = line.split(",")
            if len(arr) < 6:
                break
            if arr[0] != current_person:
                if len(series_x) > 0:
                    series_x = np.stack(series_x, axis=0)
                    series_y = np.array(series_y, dtype=np.int64) # PyTorch labels use Long/Int64
                    all_x.append(series_x)
                    all_y.append(series_y)
                series_x = []
                series_y = []
                current_person = arr[0]
            
            sensor_id = sensor_ids[arr[1]]
            label_col = class_map[arr[7].replace("\n", "")]
            feature_col_2 = np.array(arr[4:7], dtype=np.float32)

            feature_col_1 = np.zeros(4, dtype=np.float32)
            feature_col_1[sensor_id] = 1

            feature_col = np.concatenate([feature_col_1, feature_col_2])
            
            series_x.append(feature_col)
            all_feats.append(feature_col)
            series_y.append(label_col)

    # Handle last person
    if len(series_x) > 0:
        series_x = np.stack(series_x, axis=0)
        series_y = np.array(series_y, dtype=np.int64)
        all_x.append(series_x)
        all_y.append(series_y)

    all_feats = np.stack(all_feats, axis=0)
    
    # Normalization logic
    all_mean = np.mean(all_feats, axis=0)
    all_std = np.std(all_feats, axis=0)
    # Don't normalize one-hot part (indices 0-3)
    all_mean[0:4] = 0 
    all_std[0:4] = 1
    
    # Normalize
    for i in range(len(all_x)):
        all_x[i] = (all_x[i] - all_mean) / (all_std + 1e-8)

    return all_x, all_y

def cut_in_sequences(all_x, all_y, seq_len, inc=1):
    sequences_x = []
    sequences_y = []

    for i in range(len(all_x)):
        x, y = all_x[i], all_y[i]
        for s in range(0, x.shape[0] - seq_len, inc):
            start = s
            end = start + seq_len
            sequences_x.append(x[start:end])
            sequences_y.append(y[start:end])

    # 原代码返回 shape [SeqLen, Batch, Feats] (Time-Major)
    # 为了配合 PyTorch Dataset，我们这里改为 [Batch, SeqLen, Feats]
    return np.stack(sequences_x, axis=0), np.stack(sequences_y, axis=0)

# ==========================================
# 2. PyTorch Dataset 封装
# ==========================================

class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.x.shape[0]

class PersonData:
    def __init__(self, seq_len=32, batch_size=64):
        all_x, all_y = load_crappy_formated_csv()
        # inc=seq_len//2 means 50% overlap
        all_x, all_y = cut_in_sequences(all_x, all_y, seq_len=seq_len, inc=seq_len//2)
        
        total_seqs = all_x.shape[0]
        print(f"Total number of training sequences: {total_seqs}")
        
        # Shuffle
        rng = np.random.RandomState(27731)
        permutation = rng.permutation(total_seqs)
        
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)
        
        # Slicing based on permutation
        def get_subset(indices):
            return all_x[indices], all_y[indices]

        valid_x, valid_y = get_subset(permutation[:valid_size])
        test_x, test_y = get_subset(permutation[valid_size:valid_size+test_size])
        train_x, train_y = get_subset(permutation[valid_size+test_size:])
        
        self.train_loader = DataLoader(SequenceDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(SequenceDataset(valid_x, valid_y), batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(SequenceDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
        
        print(f"Test sequences: {test_x.shape[0]}")


class SequenceDatasetRandom(Dataset):
    def __init__(self, x, y, random_ratio=0.0):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.random_ratio = random_ratio
        
    def __getitem__(self, index):
        # 随机用-1mask掉random_ratio比例seq_len的元素
        if self.random_ratio > 0:
            mask = torch.rand(self.x[index].shape[0]) < self.random_ratio
            self.x[index][mask] = -1
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.x.shape[0]

class PersonDataRandom:
    def __init__(self, seq_len=32, batch_size=64, random_ratio=0.0):
        all_x, all_y = load_crappy_formated_csv()
        # inc=seq_len//2 means 50% overlap
        all_x, all_y = cut_in_sequences(all_x, all_y, seq_len=seq_len, inc=seq_len//2)
        
        total_seqs = all_x.shape[0]
        print(f"Total number of training sequences: {total_seqs}")
        
        # Shuffle
        rng = np.random.RandomState(27731)
        permutation = rng.permutation(total_seqs)
        
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)
        
        # Slicing based on permutation
        def get_subset(indices):
            return all_x[indices], all_y[indices]

        valid_x, valid_y = get_subset(permutation[:valid_size])
        test_x, test_y = get_subset(permutation[valid_size:valid_size+test_size])
        train_x, train_y = get_subset(permutation[valid_size+test_size:])
        
        self.train_loader = DataLoader(SequenceDatasetRandom(train_x, train_y, random_ratio=random_ratio), batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(SequenceDatasetRandom(valid_x, valid_y, random_ratio=random_ratio), batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(SequenceDatasetRandom(test_x, test_y, random_ratio=random_ratio), batch_size=batch_size, shuffle=False)
        
        print(f"Test sequences: {test_x.shape[0]}")
        
        
if __name__ == "__main__":
    # data = PersonData(seq_len=3200, batch_size=64)
    
    # train_x, train_y = next(iter(data.train_loader))
    # valid_x, valid_y = next(iter(data.valid_loader))
    # test_x, test_y = next(iter(data.test_loader))
    # # [Batch, SeqLen, Feats]
    # print(train_x.shape, train_y.shape)
    # print(valid_x.shape, valid_y.shape)
    # print(test_x.shape, test_y.shape, test_y.unique())
    
    data_random = PersonDataRandom(seq_len=1024, batch_size=64, random_ratio=0.5)
    train_x, train_y = next(iter(data_random.train_loader))
    valid_x, valid_y = next(iter(data_random.valid_loader))
    test_x, test_y = next(iter(data_random.test_loader))
    # [Batch, SeqLen, Feats]
    print(train_x.shape, train_y.shape)
    print(valid_x.shape, valid_y.shape)
    print(test_x.shape, test_y.shape, test_y.unique())