import torch
import time
import numpy as np

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
 
import matplotlib.pyplot as plt
import torch
import numpy as np

def gaussian_smooth(y, window_size=5, sigma=2.0):
    """
    使用 NumPy 实现简单的一维高斯平滑，不依赖 scipy
    """
    # 生成高斯核
    x = np.arange(-window_size // 2 + 1, window_size // 2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    # 卷积平滑 (mode='same' 保持长度一致)
    y_smooth = np.convolve(y, kernel, mode='same')
    
    # 处理边缘效应 (简单地用原始值填充边缘，避免卷积导致的下降)
    edge = window_size // 2
    y_smooth[:edge] = y[:edge]
    y_smooth[-edge:] = y[-edge:]
    return y_smooth

def visualize_compact_with_heat(x_batch, h_batch, save_name="sample_features_heat.png"):
    """
    x_batch: [Batch, Seq_Len, Feats]
    h_batch: [Batch, Seq_Len, Hidden]
    """
    # 1. 数据提取
    if isinstance(x_batch, torch.Tensor):
        x_data = x_batch[0].detach().cpu().numpy()
    else:
        x_data = x_batch[0]
        
    if isinstance(h_batch, torch.Tensor):
        h_data = h_batch[0].detach().cpu().numpy()
    else:
        h_data = h_batch[0]

    seq_len, num_feats = x_data.shape
    _, num_hidden = h_data.shape
    
    num_plots = min(num_feats, num_hidden, 32)
    rows, cols = 8, 4
    
    # === 1. 统一热力图范围 (Global Scale) ===
    # 这样所有子图颜色的深浅是可比的
    global_abs_max = np.max(np.abs(h_data)) + 1e-6
    vmin, vmax = -global_abs_max, global_abs_max

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes_flat = axes.flatten()

    for i in range(rows * cols):
        ax = axes_flat[i]
        
        if i < num_plots:
            raw_signal = x_data[:, i]
            heat = h_data[:, i]
            
            # === 2. 生成平滑曲线 ===
            # window_size 越大越平滑，根据 seq_len 调整
            smooth_signal = gaussian_smooth(raw_signal, window_size=7, sigma=3.0)
            
            # 计算 Y 轴范围
            y_min, y_max = raw_signal.min(), raw_signal.max()
            if y_max - y_min < 1e-6:
                y_min -= 0.5
                y_max += 0.5
            margin = (y_max - y_min) * 0.15 # 稍微多留点边距
            y_min -= margin
            y_max += margin
            
            # === 3. 绘制热力背景 (插值平滑 + 统一范围) ===
            im = ax.imshow(
                heat[None, :], 
                cmap='seismic',       # 红白蓝
                aspect='auto',
                extent=[0, seq_len, y_min, y_max],
                vmin=vmin, vmax=vmax, # 统一量纲
                interpolation='bicubic', # 极其平滑的过渡
                alpha=0.4            # 稍微不透明一点，为了衬托白线
            )
            
            # === 4. 绘制曲线 (Highlight & Smooth) ===
            # (A) 绘制平滑曲线 (作为光晕/底色)：白色，粗线条
            ax.plot(np.arange(seq_len), smooth_signal, 
                    color='brown', linewidth=2.5, alpha=0.9, label='Trend')
            
            # (B) 绘制原始曲线 (作为细节)：黑色，细线条，叠在上方
            ax.plot(np.arange(seq_len), raw_signal, 
                    color='black', linewidth=1.0, alpha=0.8, label='Raw')
            
            # 锁定坐标轴，裁掉边缘
            ax.set_xlim(0, seq_len)
            ax.set_ylim(y_min, y_max)
        
        # 去除所有边框和刻度，打造极简风格
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off') # 直接关闭坐标轴显示

    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def visualize_mask_filling(x_batch, h_aggregated, save_name="mask_filling_imputation.png"):
    """
    可视化：利用 Hidden State 的趋势来视觉填补 Input 的 Mask 区域。
    
    Args:
        x_batch: [Batch, Seq_Len, Feats]. 输入数据，含有 -1 的 Mask。
        h_aggregated: [Batch, Seq_Len]. 已经在外部做过 mean(1).mean(-1) 的 Hidden 状态。
        save_name: 保存文件名。
    """
    # 1. 提取第一个样本 (Batch 0)
    if isinstance(x_batch, torch.Tensor):
        x_data = x_batch[0].detach().cpu().numpy() # [Seq, Feat]
    else:
        x_data = x_batch[0]
        
    if isinstance(h_aggregated, torch.Tensor):
        h_data = h_aggregated[0].detach().cpu().numpy() # [Seq]
    else:
        h_data = h_aggregated[0]

    seq_len, num_feats = x_data.shape
    
    # 2. 识别 Mask 区域
    # 假设 Mask 是所有特征同时为 -1
    # mask_indices 为 bool 数组，True 表示该时刻被 Mask
    mask_bool = np.abs(x_data[:, 0] + 1.0) < 1e-4  # check if close to -1
    
    # 如果没有 Mask，为了演示，强制认为中间 20% 是逻辑上的 Mask (可选)
    if not np.any(mask_bool):
        print("Warning: No -1 mask detected in x_cpu. Plotting strictly raw data.")
        
    valid_bool = ~mask_bool
    
    # 3. 设置绘图布局
    # 限制最多画 16 个特征，避免太拥挤
    plots_to_show = min(num_feats, 4)
    cols = 2
    rows = (plots_to_show + cols - 1) // cols
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 10 * rows))
    axes_flat = axes.flatten()

    # 生成时间轴
    t = np.arange(seq_len)

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        
        if i < plots_to_show:
            # 当前特征序列
            feat_series = x_data[:, i].copy()
            
            # --- 核心逻辑：统计对齐 (Statistical Alignment) ---
            # 我们想看 h_data 在 mask 区域长什么样，但 h 的数值范围和 feat 不一样。
            # 所以我们将 h 映射到 feat 的分布上。
            
            if np.sum(valid_bool) > 1: # 确保有足够数据计算统计量
                mu_x = np.mean(feat_series[valid_bool])
                std_x = np.std(feat_series[valid_bool]) + 1e-6
                
                mu_h = np.mean(h_data) # 使用全局 h 统计量，反映整体波动
                std_h = np.std(h_data) + 1e-6
                
                # 线性变换: h_projected = (h - mu_h) * (std_x / std_h) + mu_x
                h_projected = (h_data - mu_h) * (std_x / std_h) + mu_x
            else:
                h_projected = h_data # 无法归一化，直接画

            # --- 绘图 ---
            
            # 1. 绘制 Mask 背景带 (灰色区域)
            # 找到 mask 的连续区间并填充
            y_min, y_max = np.min(h_projected), np.max(h_projected)
            if np.any(valid_bool):
                y_min = min(y_min, np.min(feat_series[valid_bool]))
                y_max = max(y_max, np.max(feat_series[valid_bool]))
            
            margin = (y_max - y_min) * 0.1
            
            # 填充 Mask 区域背景
            ax.fill_between(t, y_min - margin, y_max + margin, 
                            where=mask_bool, color='lightgray', alpha=0.4, label='Masked Region')

            # 2. 绘制 "脑补" 曲线 (Projected Hidden State)
            # 我们在全区间绘制淡色的 h 曲线，展示趋势
            ax.plot(t, h_projected, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='Hidden Trend')
            
            # 3. 在 Mask 区域高亮 "脑补" 结果
            h_masked_part = h_projected.copy()
            h_masked_part[valid_bool] = np.nan # 非 Mask 区域变透明
            ax.plot(t, h_masked_part, color='red', linewidth=2.0, label='Imputed (Filled)')

            # 4. 绘制原始真实数据 (Valid Data)
            # 将 mask 区域的数据设为 nan 以断开线条
            feat_plot = feat_series.copy()
            feat_plot[mask_bool] = np.nan
            ax.plot(t, feat_plot, color='navy', linewidth=2.0, label='True Signal')

            # 美化
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_title(f'Feature {i}', fontsize=10, pad=3)
            
            # 只在第一个图显示图例
            # if i == 0:
            #     ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

        # 去除刻度，保持极简
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 如果是空白子图，隐藏边框
        if i >= plots_to_show:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Visualization saved to {save_name}")

def visualize_mask_filling_comparison(x_masked_batch, x_true_batch, h_aggregated, save_name="imputation_comparison.png"):
    """
    可视化对比：
    1. 蓝色实线: 模型看到的输入 (Visible Input)
    2. 绿色虚线: 被遮挡的真实值 (Ground Truth in Mask)
    3. 红色实线: 模型隐藏状态的投射 (Model Imputation)
    
    Args:
        x_masked_batch: [Batch, Seq_Len, Feats] or [Batch, Seq_Len, 1]. 含有 -1 Mask。
        x_true_batch:   [Batch, Seq_Len, Feats] or [Batch, Seq_Len, 1]. 无 Mask 的真值。
        h_aggregated:   [Batch, Seq_Len]. 隐藏状态趋势 (通常是 hidden 的均值)。
    """
    # --- 1. 数据提取与预处理 ---
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x[0].detach().cpu().numpy()
        return x[0]

    x_m_data = to_numpy(x_masked_batch) # [Seq, Feat]
    x_t_data = to_numpy(x_true_batch)   # [Seq, Feat]
    h_data = to_numpy(h_aggregated)     # [Seq]

    seq_len, num_feats = x_m_data.shape
    
    # --- 2. 识别 Mask 区域 ---
    # 只要由 -1 (或接近 -1) 组成，就认为是 mask
    mask_bool = np.abs(x_m_data[:, 0] + 1.0) < 1e-4
    valid_bool = ~mask_bool
    
    if not np.any(mask_bool):
        print("Warning: No -1 mask detected. Comparison might be trivial.")

    # --- 3. 绘图设置 ---

    
    
    plots_to_show = min(num_feats, 4)
    cols = 2
    rows = (plots_to_show + cols - 1) // cols
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 10 * rows))
    axes_flat = axes.flatten()
    t = np.arange(seq_len)

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        
        if i < plots_to_show:
            # 提取当前特征
            series_masked = x_m_data[:, i].copy()
            series_true = x_t_data[:, i].copy()
            
            # --- 核心逻辑：统计对齐 (Statistical Alignment) ---
            # 我们利用“可见部分”的统计特性，将 h 映射到 x 的值域
            if np.sum(valid_bool) > 1:
                mu_x = np.mean(series_masked[valid_bool])
                std_x = np.std(series_masked[valid_bool]) + 1e-6
                
                mu_h = np.mean(h_data)
                std_h = np.std(h_data) + 1e-6
                
                # 线性变换: 将 Hidden 的波动映射到 Input 的幅度
                h_projected = (h_data - mu_h) * (std_x / std_h) + mu_x
            else:
                h_projected = h_data

            # --- 准备绘图数据 (利用 NaN 断开线条) ---
            
            # A. 可见输入 (Blue)
            plot_visible = series_masked.copy()
            plot_visible[mask_bool] = np.nan 
            
            # B. 缺失的真值 (Green)
            plot_truth_missing = series_true.copy()
            plot_truth_missing[valid_bool] = np.nan # 只保留 mask 区域
            
            # C. 模型的填补 (Red)
            plot_imputed = h_projected.copy()
            plot_imputed[valid_bool] = np.nan # 只高亮 mask 区域
            
            # --- 绘制图层 ---

            # 1. 灰色背景带 (Mask Zone)
            y_all = np.concatenate([series_true, h_projected])
            y_min, y_max = np.min(y_all), np.max(y_all)
            margin = (y_max - y_min) * 0.15
            
            ax.fill_between(t, y_min - margin, y_max + margin, 
                            where=mask_bool, color='whitesmoke', hatch='//', alpha=0.5)

            # 2. 全局趋势 (Orange, faint) - 展示模型整体是否平滑
            ax.plot(t, h_projected, color='orange', linestyle='-', linewidth=1.0, alpha=0.3, label='Global Trend (H)')

            # 3. 真实缺失值 (Green Dotted) - "正确答案"
            ax.plot(t, plot_truth_missing, color='green', linestyle=':', linewidth=2.5,  alpha=0.8, label='Ground Truth')

            # 4. 模型填补值 (Red Solid) - "模型预测"
            ax.plot(t, plot_imputed, color='red', linestyle='-', linewidth=2.0, alpha=0.9, label='Model Fill')

            # 5. 可见输入 (Navy Solid) - "已知条件"
            ax.plot(t, plot_visible, color='navy', linewidth=2.0, alpha=0.8, label='Visible Input')

            # 限制范围
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_xlim(0, seq_len)
            
            # 标题与图例
            ax.set_title(f'Feat {i} Comparison', fontsize=10, pad=4)
            if i == 0:
                ax.legend(loc='best', fontsize=8, framealpha=0.9)
        
        # 极简风格：去刻度
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= plots_to_show:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Comparison visualization saved to {save_name}")


def plot_ode_gradient_flow(model, save_name="ode_grad_flow.png"):
    """
    绘制 NODE 内部 unfolding 的梯度流
    """
    steps, grads = model.get_gradient_flow()
    
    if len(grads) == 0:
        print("Warning: No gradients found. Make sure you called loss.backward() before plotting.")
        return

    plt.figure(figsize=(10, 6))
    # 绘制折线图
    plt.plot(steps, grads, marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=8, label='Gradient Norm')
    # 添加填充，增加视觉效果
    plt.fill_between(steps, grads, color='royalblue', alpha=0.1)
    # 标注数值
    # for x, y in zip(steps, grads):
    #     plt.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=9)

    plt.title(f'Gradient Flow across ODE Unfolds (Steps={len(steps)})', fontsize=14)
    plt.xlabel('Unfold Step (Time Integration)', fontsize=12)
    plt.ylabel('Average Gradient L2 Norm', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.xticks(steps) # 强制显示整数步长
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Gradient flow plot saved to {save_name}")
    
    
def count_parameters(model):
    """
    计算模型的可训练参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_flops(model, input_shape, device):
    """
    使用 thop 计算 FLOPs (Floating Point Operations)
    需要安装: pip install thop
    """
    try:
        from thop import profile
        import logging
        
        # 抑制 thop 的详细打印
        logging.getLogger('thop').setLevel(logging.WARNING)
        
        # 构造 dummy input
        # input_shape 应为 (Batch, SeqLen, Feats)
        inputs = torch.randn(input_shape).to(device)
        
        # 将模型设为 eval 模式进行统计
        model_mode = model.training
        model.eval()
        
        # thop 需要输入是一个 tuple
        macs, params = profile(model, inputs=(inputs, ), verbose=False)
        
        # 恢复模型原有状态
        model.train(model_mode)
        
        # 1 MAC (Multiply-Accumulate) 通常被视为 2 FLOPs
        flops = macs * 2
        return flops, macs
        
    except ImportError:
        print("[Warning] 'thop' library not found. FLOPs calculation skipped. (pip install thop)")
        return 0.0, 0.0
    except Exception as e:
        print(f"[Warning] Failed to calculate FLOPs: {e}")
        return 0.0, 0.0

def measure_inference_speed(model, input_shape, device, iterations=100, warmup=10):
    """
    测量推理速度 (Latency & Throughput)
    """
    inputs = torch.randn(input_shape).to(device)
    model_mode = model.training
    model.eval()
    
    # 预热 (Warmup)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inputs)
    
    # 同步 CUDA (如果是 GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(inputs)
            # 在循环内不频繁同步，模拟真实高吞吐场景，或者在循环后同步测总时间
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_batch = total_time / iterations # 秒
    
    # 恢复状态
    model.train(model_mode)
    
    return avg_time_per_batch

def print_model_stats(model, batch_size, seq_len, input_size, device):
    """
    综合打印并返回所有指标
    """
    # 1. 参数量
    params = count_parameters(model)
    
    # 2. FLOPs (使用 Batch=1 进行标准衡量)
    input_shape_flops = (1, seq_len, input_size)
    flops, macs = compute_flops(model, input_shape_flops, device)
    
    # 3. 速度 (使用实际 BatchSize 进行吞吐量衡量，或者 Batch=1 衡量延迟)
    # 这里我们测量 Batch=1 的延迟 (Latency)
    input_shape_speed = (1, seq_len, input_size)
    latency = measure_inference_speed(model, input_shape_speed, device)
    
    print("="*40)
    print(f"Model Statistics:")
    print(f"  - Parameters : {params / 1e3:.2f} K")
    print(f"  - FLOPs      : {flops / 1e6:.2f} M (Batch=1)")
    print(f"  - MACs       : {macs / 1e6:.2f} M (Batch=1)")
    print(f"  - Latency    : {latency * 1000:.2f} ms/sample (Batch=1)")
    print("="*40)
    
    return params, flops, macs, latency