import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.utils import visualize_compact,visualize_compact_with_heat
class CTMDNXCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_synch=None, 
        unfolds=6,    
        delta_t=0.1,  
        tau_init=1.0, 
        global_feedback=True,
        cell_clip=-1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.unfolds = unfolds
        self.delta_t = delta_t
        print(f"delta_t: {self.delta_t}")
        self.cell_clip = cell_clip
        self.global_feedback = global_feedback

        # --- 1. 基础 ODE 参数 ---
        # 优化：将 Input 和 Hidden 的投影分开，避免在 ODE 循环中重复计算 Input 的投影
        self.fc_x = nn.Linear(input_size, hidden_size, bias=False)
        self.fc_h = nn.Linear(hidden_size, hidden_size) 
        
        # 时间常数参数化 (Sigmoid 限制范围或 Softplus)
        self.tau_param = nn.Parameter(torch.tensor(tau_init))

        # --- 2. 记忆动力学参数 ---
        if n_synch is None:
            self.n_synch = hidden_size 
        else:
            self.n_synch = n_synch

        # 随机配对索引
        self.register_buffer('idx_left', torch.randint(0, hidden_size, (self.n_synch,)))
        self.register_buffer('idx_right', torch.randint(0, hidden_size, (self.n_synch,)))

        # [频率维度] Decay rate 'r'
        self.r_param = nn.Parameter(torch.randn(self.n_synch)) 
        
        # [重要性权重维度] 
        self.w_memory = nn.Linear(self.n_synch, hidden_size, bias=False)
        
        # LayerNorm 可以增加 ODE 求解的稳定性
        self.ln = nn.LayerNorm(hidden_size)
        self.adaptive_threshold = 1e-2

    def compute_memory_update(self, activated_state, prev_alpha, prev_beta, r_sigmoid):
        """
        计算一步记忆更新。
        为了速度，这里不进行复杂的开方运算，采用标准的 EMA 形式或者累积形式。
        """
        # 1. 激活匹配 (Hebb-like Coincidence)
        left = activated_state[:, self.idx_left]
        right = activated_state[:, self.idx_right]
        pairwise_product = left * right # [B, n_synch]
        # 2. 记忆状态更新
        # r_sigmoid: [1, n_synch] (decay factor, 0=forget all, 1=remember all)
        # 公式: Alpha_t = r * Alpha_{t-1} + (1-r) * Input  (标准 EMA)
        # 或者: Alpha_t = r * Alpha_{t-1} + Input (累积式，你原本的逻辑)
        # 采用你的累积式逻辑，这更符合“能量积累”的物理直觉
        new_alpha = r_sigmoid * prev_alpha + pairwise_product
        
        # Beta 用于归一化累积带来的数值膨胀
        new_beta = r_sigmoid * prev_beta + 1.0
        
        # 3. Readout (同步值)
        # 加上 1e-6 防止除零
        sync_val = new_alpha / (torch.sqrt(new_beta) + 1e-6)
        
        return sync_val, new_alpha, new_beta

    def forward(self, x, h, mem_alpha, mem_beta):
        tau = F.softplus(self.tau_param) + 0.01 # 保证 tau > 0
        r_sigmoid = torch.sigmoid(self.r_param).unsqueeze(0) # [1, n_synch]
        
        current_h = h
        current_alpha = mem_alpha
        current_beta = mem_beta

        # 预计算输入的投影 (Speed Up!)
        # 这样在 ODE 循环里只需要算 recurrent 部分
        x_drive = self.fc_x(x) 
        # prev_sync_val = None
        # Euler Integration Loop
        
        for steps in range(self.unfolds):
            # 1. 记忆更新 (Memory Dynamics)
            # 使用 tanh(h) 作为激活状态，限制幅度在 [-1, 1] 之间，防止记忆爆炸
            activated_state = torch.tanh(current_h)
            
            sync_val, next_alpha, next_beta = self.compute_memory_update(
                activated_state, current_alpha, current_beta, r_sigmoid
            )
            # 更新循环内的记忆状态
            current_alpha = next_alpha
            current_beta = next_beta
            
            # 2. 计算总驱动力
            # Base drive: W_x * x + W_h * h
            if self.global_feedback:
                # 原始逻辑：input 和 hidden 拼接
                # 这里我们利用预计算的 x_drive
                base_drive = x_drive + self.fc_h(current_h)
            else:
                base_drive = x_drive

            # Memory drive
            memory_drive = self.w_memory(sync_val)
            
            # 3. ODE Update: dh/dt = -h/tau + f(total_drive)
            # f_total = Tanh(Linear_Drive + Memory_Drive)
            f_total = torch.tanh(base_drive + memory_drive)
            
            dh = (-current_h ) / tau + f_total
            # current_h = current_h + self.delta_t * dh
            
            h_update = self.delta_t * dh
            if self.adaptive_threshold > 0 and steps >= 3:
                # 计算 h 的更新幅度 (L1 范数平均)
                # 这直接代表了系统状态改变了多少
                delta_h_norm = torch.abs(h_update).mean()
                if delta_h_norm < self.adaptive_threshold:
                    # 在 break 前，记得把最后这一小步加上去
                    current_h = current_h + h_update
                    if self.cell_clip > 0:
                        current_h = torch.clamp(current_h, -self.cell_clip, self.cell_clip)
                    break
                current_h = current_h + h_update
            else:
                current_h = current_h + h_update
            current_h = current_h + h_update
            # [Batch, unfold, Hidden]
            
        # print(f"Stable at step {steps}, delta: {delta_h_norm.item():.6f}")
        return current_h, current_alpha, current_beta, steps

class CTMDNXADAPTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, unfold=6, n_synch=None):
        super(CTMDNXADAPTClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_synch = n_synch if n_synch is not None else hidden_size
        
        # 使用之前优化过的 Cell
        self.cell = CTMDNXCell(
            input_size, 
            hidden_size, 
            n_synch=self.n_synch, 
            unfolds=unfold, 
            global_feedback=True
        )
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [Batch, SeqLen, Feats]
        B, T, F = x.size()
        
        h = torch.zeros(B, self.hidden_size, device=x.device)
        
        # 显式初始化记忆状态为 0
        mem_alpha = torch.zeros(B, self.n_synch, device=x.device)
        mem_beta = torch.zeros(B, self.n_synch, device=x.device)
        
        outputs = []
        mean_steps = 0
        
        
        for t in range(T):
            inp = x[:, t, :]
            h, mem_alpha, mem_beta, steps = self.cell(inp, h, mem_alpha, mem_beta)
            outputs.append(h)
            mean_steps += (steps+1)
        mean_steps = mean_steps / T
        
        # Stacks: [Batch, SeqLen, Hidden]
        output = torch.stack(outputs, dim=1)
        
        #  (对所有时间步进行预测):
        # output shape: [B, T, Hidden] -> logits shape: [B, T, Output_Size]
        logits = self.readout(output) 

        
        return logits, mean_steps
    
    
# --- 测试代码 ---
if __name__ == "__main__":
    # 配置
    BATCH_SIZE = 32
    SEQ_LEN = 50
    INPUT_SIZE = 10
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 5
    
    # 实例化模型
    model = CTMDNXADAPTClassifier(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, unfold=4)
    
    # 模拟输入数据
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
    
    # 前向传播
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("CT-MDN 模型构建成功。")
    
    # 检查参数量，验证是否高效
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")