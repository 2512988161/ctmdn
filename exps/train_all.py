import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import argparse
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models.build import build_model
import models.utils as utils

from datasetss.person import PersonData, PersonDataRandom
from datasetss.smnist import SMnistData
from datasetss.occupancy import OccupancyData, OccupancyDataRandom
from datasetss.gesture import GestureData
from datasetss.ozone import OzoneData
from datasetss.har import HarData

# 回归任务
from datasetss.power import PowerData
from datasetss.traffic import TrafficData
# 多维度回归 
from datasetss.cheetah import CheetahData

import time

# torch seed setup 
seed=42
torch.manual_seed(seed)

# ==========================================
# 训练逻辑
# ==========================================

def train_model(args):
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' else "cpu")
    print(f"Using device: {device}")

    # Data
    if args.task == "person":
        data = PersonData(seq_len=args.seq_len, batch_size=64) # [Batch, SeqLen, Feats] = [64, 32, 4+3]
        input_size = 4 + 3 # 4 one-hot sensors + 3 features 7维特征
        output_size = 7 # 7种类别
    elif args.task in ["occupancy-64-0.0",
                       "occupancy-128-0.0",
                       "occupancy-32-0.1",
                       "occupancy-32-0.2",
                       "occupancy-32-0.3",
                       "occupancy-32-0.4",
                       "occupancy-32-0.5",
                       "occupancy-32-0.6",
                       "occupancy-256-0.0",
                       "occupancy-512-0.0",
                       "occupancy-1024-0.0",
                       "occupancy-256-0.1",
                       "occupancy-256-0.2",
                       "occupancy-256-0.3",
                       "occupancy-256-0.4",
                       "occupancy-256-0.5",
                       "occupancy-256-0.6",
                       "occupancy-256-0.7",
                       "occupancy-256-0.8",
                       "occupancy-256-0.9",
                       ]:
        seq_len = int(args.task.split("-")[1])
        random_ratio = float(args.task.split("-")[2])
        print(f"seq_len",seq_len,"random_ratio",random_ratio)
        data = OccupancyDataRandom(seq_len=seq_len, batch_size=64, random_ratio=random_ratio) # [Batch, SeqLen, Feats] = [64, 32, 5]
        input_size = 5 # 5维特征
        output_size = 2 # 2种类别
        
    elif args.task == "smnist":
        data = SMnistData(batch_size=64) # BTD = [64, 28, 28]
        input_size = 28 # 28x28 images (feat_dim)
        output_size = 10 # 10种类别
    elif args.task == "occupancy":
        data = OccupancyData(seq_len=args.seq_len, batch_size=64) # [Batch, SeqLen, Feats] = [64, 32, 5]
        input_size = 5 # 5维特征
        output_size = 2 # 2种类别
    elif args.task == "gesture":
        data = GestureData(seq_len=args.seq_len, batch_size=64) # [Batch, SeqLen, Feats] = [64, 32, 32]
        input_size = 32 # 6维特征
        output_size = 5 # 5种类别
    elif args.task == "ozone":
        data = OzoneData(seq_len=args.seq_len, batch_size=64) # [Batch, SeqLen, Feats] = [64, 32, 72]
        input_size = 72 # 72维特征
        output_size = 2 # 2种类别
    elif args.task == "har":
        data = HarData(seq_len=args.seq_len, batch_size=64) # [Batch, SeqLen, Feats] = [64, 32, 561]
        input_size = 561 # 561维特征
        output_size = 6 # 6种类别
    elif args.task == "power":
        data = PowerData(seq_len=args.seq_len, batch_size=64) # [Batch, SeqLen, Feats] = [64, 32, 6]
        input_size = 6 # 6维特征
        output_size = 1 # 1维输出
    elif args.task == "traffic":
        data = TrafficData(seq_len=args.seq_len, batch_size=64) # [Batch, SeqLen, Feats] = [64, 32, 1]
        input_size = 7 # 1维特征
        output_size = 1 # 1维输出
    elif args.task == "cheetah":
        data = CheetahData(seq_len=args.seq_len, batch_size=64) # [Batch, SeqLen, Feats] = [64, 32, 17]
        input_size = 17 # 17维特征
        output_size = 17 # 17维输出    
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    model = build_model(args, input_size, output_size, device)
    model_params, model_flops,model_macs, model_latency = utils.print_model_stats(
            model, 
            batch_size=1, 
            seq_len=args.size, 
            input_size=input_size, 
            device=device
        )
    
    # Optimizer (Use higher LR for LTC as per original script logic)
    lr = 0.01
    # lr = 0.01 if args.model.startswith("ltc") else 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
     # 1. 定义任务类型列表
    REGRESSION_TASKS = ["power", "traffic", "cheetah"]
    
    # 2. 选择 Loss Function
    if args.task in REGRESSION_TASKS:
        criterion = nn.MSELoss()  # 回归用均方误差
    else:
        criterion = nn.CrossEntropyLoss() # 分类用交叉熵

    # Logging
    result_dir = os.path.join("results", args.task)
    
    os.makedirs(result_dir, exist_ok=True)
    if args.model == "ctmdnattention":
        logdir = os.path.join(result_dir, f"{args.model}_{args.size}_fold{args.unfolds}_tick{args.ticks}")
    else:
        logdir = os.path.join(result_dir, f"{args.model}_{args.size}_fold{args.unfolds}")
    os.makedirs(logdir,     exist_ok=True)
    # logging model stats to txt
    stats_txt = os.path.join(logdir, "model_stats.txt")
    with open(stats_txt, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Model: {str(model)}\n")
        f.write(f"Input size: {input_size}\n")
        f.write(f"Seq len: {args.size}\n")
        f.write(f"Batch size: 1\n")
        f.write(f"Unfolds: {args.unfolds}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Params: {model_params}\n")
        f.write(f"FLOPs: {model_flops}\n")
        f.write(f"MACs: {model_macs}\n")
        f.write(f"Latency (ms): {model_latency}\n")
    
    if args.task in REGRESSION_TASKS:
        best_valid_acc = float('inf')
    else:
        best_valid_acc = 0.0
        
    best_stats = None
    
    patience_limit = getattr(args, 'patience', 20) 
    patience_counter = 0
    
    print("Entering training loop")
    
    
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        train_accs = []
        
        # Training Loop
        start_time = time.time()
        mean_steps = 0
        # for x, y,x_true in data.train_loader:
        for x, y in data.train_loader:
            # x, y,x_true = x.to(device), y.to(device),x_true.to(device)
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            if args.model in ["ctmdnxadapt","nmodectmdnxadapt"]:
                logits, steps = model(x)
                # logits, steps = model(x,x_true)
                mean_steps += steps
            else:
                logits = model(x)
            
            if args.task in REGRESSION_TASKS:
                # 回归任务处理
                # 序列到序列回归，通常不需要取最后一个时间步
                # y.view(-1, output_size) 确保维度匹配
                loss = criterion(logits, y.float()) 
                # 回归没有 "Accuracy"，计算 MAE
                acc = nn.functional.l1_loss(logits, y.float()).item()
            else:
                # 分类任务处理 (保持原样)
                 # y shape: [Batch, SeqLen], logits: [Batch, SeqLen, Classes]
                if args.task not in ["person", "gesture", "ozone"]:
                    logits = logits[:, -1, :] # 取最后一个时间步的logits
                    
                loss = criterion(logits.view(-1, output_size), y.view(-1))
                # 计算 Accuracy
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == y).float().mean().item()
                
                
            loss.backward()
            
            # utils.plot_ode_gradient_flow(model.cell)
            # breakpoint()
            optimizer.step()
            # Apply Constraints
            if args.sparsity > 0:
                model.apply_sparsity()
            if args.model.startswith("ltc"):
                model.cell.apply_weight_constraints()
            # Metrics
            train_losses.append(loss.item())
            train_accs.append(acc)
        mean_steps /= len(data.train_loader)
        end_time = time.time()
        train_time = end_time - start_time
        
        # Validation
        if epoch % args.log == 0 or epoch == args.epochs - 1:
            model.eval()
            
            def evaluate(loader):
                losses = []
                accs = []
                with torch.no_grad():
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        if args.model in ["ctmdnxadapt","nmodectmdnxadapt"]:
                            logits, steps = model(x)
                        else:
                            logits = model(x)
                        if args.task in REGRESSION_TASKS:
                            # 回归任务处理
                            # 序列到序列回归，通常不需要取最后一个时间步
                            # y.view(-1, output_size) 确保维度匹配
                            loss = criterion(logits, y.float()) 
                            # 回归没有 "Accuracy"，计算 MAE
                            acc = nn.functional.l1_loss(logits, y.float()).item()
                        else:
                            # 分类任务处理 (保持原样)
                            # y shape: [Batch, SeqLen], logits: [Batch, SeqLen, Classes]
                            if args.task not in ["person", "gesture", "ozone"]:
                                logits = logits[:, -1, :] # 取最后一个时间步的logits
                            loss = criterion(logits.view(-1, output_size), y.view(-1))
                            # 计算 Accuracy
                            preds = torch.argmax(logits, dim=-1)
                            acc = (preds == y).float().mean().item()
                            
                        # if args.task not in ["person", "gesture", "ozone"]:
                        #     logits = logits[:, -1, :] # 取最后一个时间步的logits
                        # loss = criterion(logits.view(-1, output_size), y.view(-1))
                        # preds = torch.argmax(logits, dim=-1)
                        # acc = (preds == y).float().mean().item()
                        losses.append(loss.item())
                        accs.append(acc)
                return np.mean(losses), np.mean(accs)
            
            valid_loss, valid_acc = evaluate(data.valid_loader)
            test_loss, test_acc = evaluate(data.test_loader)
            
            train_loss_mean = np.mean(train_losses)
            train_acc_mean = np.mean(train_accs)
            
            if args.model in ["ctmdnxadapt","nmodectmdnxadapt"]:
                msg = ( f"Epoch {epoch:03d} | "
                    f"Train Loss: {train_loss_mean:.2f} Acc: {train_acc_mean*100:.2f}% | "
                    f"Valid Loss: {valid_loss:.2f} Acc: {valid_acc*100:.2f}% | "
                    f"Test Loss: {test_loss:.2f} Acc: {test_acc*100:.2f}% | "
                    f"Train Time: {train_time:.2f}s | "
                    f"Mean Steps: {mean_steps:.2f}\n")
            else:
                msg = ( f"Epoch {epoch:03d} | "
                    f"Train Loss: {train_loss_mean:.2f} Acc: {train_acc_mean*100:.2f}% | "
                    f"Valid Loss: {valid_loss:.2f} Acc: {valid_acc*100:.2f}% | "
                    f"Test Loss: {test_loss:.2f} Acc: {test_acc*100:.2f}% | "
                    f"Train Time: {train_time:.2f}s\n")
            
            print(msg)
            log_file = os.path.join(logdir, "train.log")
            with open(log_file, "a") as f:
                f.write(msg)
            if args.task in REGRESSION_TASKS:
                if valid_acc < best_valid_acc:
                    best_valid_acc = valid_acc
                    best_stats = (epoch, train_loss_mean, train_acc_mean, valid_loss, valid_acc, test_loss, test_acc)
                    # 保存最佳模型
                    torch.save(model.state_dict(), os.path.join(logdir, "best_model.pth"))
                    # 如果性能提升了，重置耐心计数器
                    patience_counter = 0 
                    # print(f"New best model found at epoch {epoch}")
                else:
                    # 如果性能没有提升，增加计数器
                    patience_counter += 1
                    # 触发早停
                    if patience_counter >= patience_limit:
                        print(f"Early stopping triggered at epoch {epoch}. No improvement for {patience_limit} check points.")
                        break
            else:
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_stats = (epoch, train_loss_mean, train_acc_mean, valid_loss, valid_acc, test_loss, test_acc)
                    # 保存最佳模型
                    torch.save(model.state_dict(), os.path.join(logdir, "best_model.pth"))
                    # 如果性能提升了，重置耐心计数器
                    patience_counter = 0 
                    # print(f"New best model found at epoch {epoch}")
                else:
                    # 如果性能没有提升，增加计数器
                    patience_counter += 1
                    # 触发早停
                    if patience_counter >= patience_limit:
                        print(f"Early stopping triggered at epoch {epoch}. No improvement for {patience_limit} check points.")
                        break
    # Final Record
    if best_stats:
        ep, tl, ta, vl, va, tsl, tsa = best_stats
        print(f"Best Epoch {ep:03d}: Valid Acc {va*100:.2f}%, Test Acc {tsa*100:.2f}%")
        with open(log_file, "a") as f:
            f.write(f"Best Epoch {ep:03d}: Valid Acc {va*100:.2f}%, Test Acc {tsa*100:.2f}%")
            f.write("best epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")
            f.write(f"{ep}, {tl:.2f}, {ta*100:.2f}, {vl:.2f}, {va*100:.2f}, {tsl:.2f}, {tsa*100:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="lstm", help="Model type: lstm, ltc, node, ctgru, ctrnn, ctmdn, ctmdnx, selfattention, ctmdnselfattention")
    parser.add_argument('--log', default=1, type=int, help="Log interval")
    parser.add_argument('--seq_len', default=32, type=int, help="seq_len of data")
    parser.add_argument('--size', default=32, type=int, help="Hidden size / sequence length logic (reused from original)")
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--unfolds', default=6, type=int)
    parser.add_argument('--ticks', default=6, type=int)
    parser.add_argument('--sparsity', default=0.0, type=float)
    
    parser.add_argument('--task', default="person", help="Task type: person")
    
    args = parser.parse_args()
    train_model(args)