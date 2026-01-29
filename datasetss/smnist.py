import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import numpy as np

# ==========================================
# 1. Dataset 封装 (保持简洁，仅负责 getitem)
# ==========================================
class SequenceDataset(Dataset):
    """
    通用序列数据集封装
    x: [Batch, SeqLen, Feats]
    y: [Batch]
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.x.shape[0]

# ==========================================
# 2. SMnistData 数据管理类
# ==========================================
class SMnistData:
    def __init__(self, batch_size=64, data_root="./data"):
        """
        初始化 SMNIST 数据
        SMNIST 将 28x28 的图像视为: 序列长度 28, 特征维度 28
        """
        # 1. 下载并加载原始数据
        # 训练集 (60000张)
        train_set = datasets.MNIST(
            root=data_root,
            train=True,
            download=True
        )
        # 测试集 (10000张)
        test_set = datasets.MNIST(
            root=data_root,
            train=False,
            download=True
        )

        # 2. 预处理数据
        # MNIST.data 是 uint8 [N, 28, 28], 转换为 float 并归一化到 [0, 1]
        x_train_all = train_set.data.float() / 255.0
        y_train_all = train_set.targets.long()

        x_test = test_set.data.float() / 255.0
        y_test = test_set.targets.long()

        # 3. 划分训练集和验证集 (从 60000 张训练集中划分)
        total_train_samples = x_train_all.shape[0]
        val_split = int(0.1 * total_train_samples) # 10% 用于验证 (6000张)
        train_split = total_train_samples - val_split # 90% 用于训练 (54000张)

        # 为了保证可复现性，固定种子进行打乱 (可选，MNIST通常不需要大乱序，但这里保持与PersonData逻辑类似)
        indices = torch.randperm(total_train_samples)
        
        train_indices = indices[:train_split]
        valid_indices = indices[train_split:]

        x_train = x_train_all[train_indices]
        y_train = y_train_all[train_indices]

        x_valid = x_train_all[valid_indices]
        y_valid = y_train_all[valid_indices]

        # 4. 打印统计信息
        print(f"SMNIST Data Loaded:")
        print(f"Train sequences: {x_train.shape[0]} (Shape: {x_train.shape[1:]})")
        print(f"Valid sequences: {x_valid.shape[0]}")
        print(f"Test  sequences: {x_test.shape[0]}")

        # 5. 创建 DataLoader
        # PyTorch RNN 默认接受 batch_first=True, 如果需要 time-major，需要在模型里处理或这里 permute
        # 这里保持 [Batch, SeqLen, Feats] 格式
        self.train_loader = DataLoader(
            SequenceDataset(x_train, y_train), 
            batch_size=batch_size, 
            shuffle=True
        )
        
        self.valid_loader = DataLoader(
            SequenceDataset(x_valid, y_valid), 
            batch_size=batch_size, 
            shuffle=False
        )
        
        self.test_loader = DataLoader(
            SequenceDataset(x_test, y_test), 
            batch_size=batch_size, 
            shuffle=False
        )

# ==========================================
# 3. 测试代码
# ==========================================
if __name__ == "__main__":
    # 实例化
    smnist = SMnistData(batch_size=64)
    
    # 验证输出形状
    for x, y in smnist.train_loader:
        print(f"Batch X shape: {x.shape}") # 预期: [64, 28, 28] (Batch, Time, Feats)
        print(f"Batch Y shape: {y.shape}") # 预期: [64] (Batch)
        print(f"Unique labels: {y.unique()}")
        break
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, transforms
# import numpy as np
# class SMnistDataset(Dataset):
#     def __init__(self, train=True):
#         mnist = datasets.MNIST(
#             root="./data",
#             train=train,
#             download=True,
#             transform=transforms.ToTensor()
#         )

#         xs = mnist.data.float() / 255.0  # [N,28,28]
#         ys = mnist.targets

#         if train:
#             split = int(0.9 * len(xs))
#             self.x = xs[:split]
#             self.y = ys[:split]
#             self.valid_x = xs[split:]
#             self.valid_y = ys[split:]
#         else:
#             self.x = xs
#             self.y = ys

#     def get_train(self):
#         # [T,B,D]
#         x = self.x.permute(1, 0, 2)
#         return x, self.y

#     def get_valid(self):
#         x = self.valid_x.permute(1, 0, 2)
#         return x, self.valid_y
    
# if __name__ == "__main__":
#     train_data = SMnistDataset(train=True)
#     test_data = SMnistDataset(train=False)

#     train_x, train_y = train_data.get_train()
#     valid_x, valid_y = train_data.get_valid()
#     test_x, test_y = test_data.get_train()

#     print(train_x.shape, train_y.shape)
#     print(valid_x.shape, valid_y.shape)
#     print(test_x.shape, test_y.shape)