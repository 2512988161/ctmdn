import os
import numpy as np
import pandas as pd
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

def load_trace(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
        
    df = pd.read_csv(filename, header=0)
    
    str_y = df["Phase"].values
    convert = {"D": 0, "P": 1, "S": 2, "H": 3, "R": 4}
    
    y = np.empty(str_y.shape[0], dtype=np.int64) # PyTorch labels use int64
    for i in range(str_y.shape[0]):
        y[i] = convert[str_y[i]]
       
    x = df.values[:, :-1].astype(np.float32)
    
    return x, y

def cut_in_sequences(tup, seq_len, interleaved=False):
    x, y = tup
    num_sequences = x.shape[0] // seq_len
    sequences_x = []
    sequences_y = []

    for s in range(num_sequences):
        start = seq_len * s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

        if interleaved and s < num_sequences - 1:
            start += seq_len // 2
            end = start + seq_len
            sequences_x.append(x[start:end])
            sequences_y.append(y[start:end])

    return sequences_x, sequences_y

# ==========================================
# 2. GestureData 类
# ==========================================

class GestureData:
    def __init__(self, seq_len=32, batch_size=64):
        training_files = [
            "a3_va3.csv", "b1_va3.csv", "b3_va3.csv",
            "c1_va3.csv", "c3_va3.csv", "a2_va3.csv", "a1_va3.csv",
        ]
        data_dir = "data/gesture"
        
        all_x_list = []
        all_y_list = []

        # 1. 加载并切分数据
        # 原逻辑：默认对训练文件开启 interleaved
        interleaved_train = True 
        for f in training_files:
            file_path = os.path.join(data_dir, f)
            try:
                raw_x, raw_y = load_trace(file_path)
                seq_x, seq_y = cut_in_sequences((raw_x, raw_y), seq_len, interleaved=interleaved_train)
                all_x_list.extend(seq_x)
                all_y_list.extend(seq_y)
            except FileNotFoundError as e:
                print(f"Warning: {e}")

        if not all_x_list:
            raise RuntimeError("No data loaded. Check data path.")

        # 2. 堆叠数据
        # 原代码使用 axis=1 (Time-Major: [Seq, Batch, Feat])
        # 这里改为 axis=0 (Batch-First: [Batch, Seq, Feat])
        all_x = np.stack(all_x_list, axis=0)
        all_y = np.stack(all_y_list, axis=0)

        # 3. 归一化 (Normalization)
        # 计算全局均值和方差 (在 Flatten 后计算)
        flat_x = all_x.reshape([-1, all_x.shape[-1]])
        mean_x = np.mean(flat_x, axis=0)
        std_x = np.std(flat_x, axis=0)
        all_x = (all_x - mean_x) / (std_x + 1e-8)

        total_seqs = all_x.shape[0]
        print(f"Total number of sequences: {total_seqs}")

        # 4. 数据集划分 (Train/Valid/Test)
        # 使用固定种子保证可复现性
        rng = np.random.RandomState(23489)
        permutation = rng.permutation(total_seqs)
        
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)
        
        # 根据 permutation 切分
        valid_indices = permutation[:valid_size]
        test_indices = permutation[valid_size:valid_size+test_size]
        train_indices = permutation[valid_size+test_size:]

        self.train_x = all_x[train_indices]
        self.train_y = all_y[train_indices]
        
        self.valid_x = all_x[valid_indices]
        self.valid_y = all_y[valid_indices]
        
        self.test_x = all_x[test_indices]
        self.test_y = all_y[test_indices]

        print(f"Train size: {self.train_x.shape[0]}")
        print(f"Valid size: {self.valid_x.shape[0]}")
        print(f"Test  size: {self.test_x.shape[0]}")

        # 5. 创建 DataLoader
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
        
        
import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_compact(x_batch):
    """
    x_batch: shape [Batch, Seq_Len, Feats] -> [16, 30, 32]
    """
    # 取第一个样本: [30, 32]
    data = x_batch[0].numpy()
    seq_len, num_feats = data.shape

    # 8行4列
    rows, cols = 8, 4
    
    # figsize 调整为瘦长型以适应 8行
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10)) 
    
    # 展平以便遍历
    axes_flat = axes.flatten()

    for i in range(32):
        ax = axes_flat[i]
        # 绘制曲线 (黑色线条，简洁)
        ax.plot(data[:, i], color='black', linewidth=1.2)
        # === 极简风格设置 ===
        # 移除X和Y轴的刻度值 (数字)
        ax.set_xticks([])
        ax.set_yticks([])
        # 移除轴的标签 (label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        # 保留边框 (Spines)，如果连边框都不要，可以用 ax.axis('off')
        # 这里保留边框作为"框框"
        for spine in ax.spines.values():
            spine.set_linewidth(0.5) # 边框线弄细一点
        # # 可选：在图内左上角标注特征序号，方便对应
        # ax.text(0.02, 0.85, f'{i}', transform=ax.transAxes, fontsize=8, color='gray')

    # === 紧凑布局 ===
    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig("sample_features.png", dpi=300)

if __name__ == "__main__":
    # 测试代码
    try:
        # 注意：确保你的 data/gesture 目录下有数据文件，否则会报错
        data = GestureData(seq_len=512, batch_size=16)
        # 获取一个 Batch
        for x, y in data.train_loader:
            print(f"Batch X shape: {x.shape}") # [16, 30, 32]
            print(f"Batch Y shape: {y.shape}") # [16, 30]
            
            # === 调用可视化函数 ===
            visualize_compact(x)
            
            break # 只看第一个 batch 即可
            
    except Exception as e:
        print(f"Error: {e}")
        

# if __name__ == "__main__":
#     # 测试代码
#     try:
#         data = GestureData(seq_len=30, batch_size=16)
#         for x, y in data.train_loader:
#             print(f"Batch X shape: {x.shape}") # 预期: [16, 32, Feats]
#             print(f"Batch Y shape: {y.shape}") # 预期: [16, 32]
#             print(f"Unique labels: {y.unique()}")
#             break
#     except Exception as e:
#         print(f"Error: {e}")