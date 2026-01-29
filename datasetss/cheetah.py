import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 辅助函数与 Dataset
# ==========================================

def cut_in_sequences_cheetah(x, seq_len, inc=1):
    """
    切分序列用于下一时刻预测任务 (Next-step prediction)
    输入 x: [Time, Feats]
    输出: 
        sequences_x: [Batch, SeqLen, Feats]
        sequences_y: [Batch, SeqLen, Feats] (向后偏移一个时间步)
    """
    sequences_x = []
    sequences_y = []

    # 保证 y 能取到 start+1 到 end+1
    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        
        # 边界检查，防止越界
        if end + 1 > x.shape[0]:
            break
            
        sequences_x.append(x[start:end])
        sequences_y.append(x[start+1:end+1])

    if len(sequences_x) == 0:
        return np.array([]), np.array([])
        
    # Stack 为 [Batch, SeqLen, Feats]
    return np.stack(sequences_x, axis=0), np.stack(sequences_y, axis=0)

class RegressionDataset(Dataset):
    """
    用于回归任务的数据集 (y 是 float)
    """
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.x.shape[0]

# ==========================================
# 2. CheetahData 类
# ==========================================

class CheetahData:
    def __init__(self, seq_len=32, batch_size=64):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.obs_size = 17  # Cheetah 数据的特征维度
        
        data_dir = "data/cheetah"
        if not os.path.exists(data_dir):
             raise FileNotFoundError(f"Data directory not found at {data_dir}")

        # 获取所有 .npy 文件并排序
        all_files = sorted([
            os.path.join(data_dir, d) 
            for d in os.listdir(data_dir) 
            if d.endswith(".npy")
        ])

        # 按照原脚本逻辑划分文件
        # 注意：原脚本 train=15:25, test=5:15, valid=0:5
        train_files = all_files[15:25]
        test_files = all_files[5:15]
        valid_files = all_files[:5]

        # 加载并处理数据
        # 原脚本中 stride (inc) 设置为 10
        train_x, train_y = self._load_and_process(train_files, inc=10)
        valid_x, valid_y = self._load_and_process(valid_files, inc=10)
        test_x, test_y = self._load_and_process(test_files, inc=10)

        print(f"Cheetah Data Loaded:")
        print(f"Train sequences: {train_x.shape[0]}")
        print(f"Valid sequences: {valid_x.shape[0]}")
        print(f"Test  sequences: {test_x.shape[0]}")
        print(f"Feature dim: {train_x.shape[-1]}")

        # 创建 DataLoader
        self.train_loader = DataLoader(
            RegressionDataset(train_x, train_y), 
            batch_size=batch_size, 
            shuffle=True
        )
        self.valid_loader = DataLoader(
            RegressionDataset(valid_x, valid_y), 
            batch_size=batch_size, 
            shuffle=False
        )
        self.test_loader = DataLoader(
            RegressionDataset(test_x, test_y), 
            batch_size=batch_size, 
            shuffle=False
        )

    def _load_and_process(self, files, inc):
        all_x = []
        all_y = []
        
        for f in files:
            arr = np.load(f)
            arr = arr.astype(np.float32)
            
            # 切分序列
            x, y = cut_in_sequences_cheetah(arr, self.seq_len, inc=inc)
            
            if len(x) > 0:
                all_x.append(x)
                all_y.append(y)
        
        # 将列表中的 array 拼接起来
        # x 这里的形状是 [Batch, SeqLen, Feats]
        if len(all_x) > 0:
            return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)
        else:
            return np.array([]), np.array([])

if __name__ == "__main__":
    # 测试代码
    try:
        data = CheetahData(seq_len=32, batch_size=1)
        for x, y in data.train_loader:
            print("Batch X shape:", x.shape) # 预期: [16, 32, 17]
            print("Batch Y shape:", y.shape) # 预期: [16, 32, 17]
            # print(f"{y}")
            break
    except FileNotFoundError as e:
        print(e)