import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if global_feedback:
            self.fc_input = nn.Linear(input_size + hidden_size, hidden_size)
        else:
            self.fc_input = nn.Linear(input_size, hidden_size)
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
        self.A_param = nn.Parameter(torch.randn(self.hidden_size))
        # [重要性权重维度] 
        self.w_memory = nn.Linear(self.n_synch, hidden_size, bias=False)
        
        # LayerNorm 可以增加 ODE 求解的稳定性
        self.adaptive_threshold = 1e-2

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r):
        """
        基于提供的代码片段修改的同步计算函数。
        activated_state: [Batch, Hidden]
        decay_alpha: [Batch, n_synch] - 记忆的分子部分 (历史积累)
        decay_beta:  [Batch, n_synch] - 记忆的分母部分 (归一化项)
        r: [n_synch] - 衰减率
        Synchronisation dynamics (per pair k, per batch b):
        x_t^(b,k) = h_t^(b,i_k) * h_t^(b,j_k)                 # instantaneous activation matching
        λ_k       = sigmoid(r_k)                              # decay rate in (0, 1)
        α_t^(b,k) = λ_k * α_{t-1}^(b,k) + x_t^(b,k)           # numerator memory (EWMA)
        β_t^(b,k) = λ_k * β_{t-1}^(b,k) + 1                   # denominator memory (window size)
        s_t^(b,k) = α_t^(b,k) / ( sqrt(β_t^(b,k)) + ε )       # synchronisation readout
        """
        # 1. 提取配对神经元的激活值
        left = activated_state[:, self.idx_left]   # [B, n_synch]
        right = activated_state[:, self.idx_right] # [B, n_synch]
        # 2. 激活匹配 (Activation Matching): 计算当前时刻的点积
        pairwise_product = left * right # [B, n_synch]
        # 3. 记忆动力学更新 (Memory Update)
        # 使用 sigmoid(r) 确保衰减率在 0-1 之间
        r_sigmoid = torch.sigmoid(r).unsqueeze(0) # [1, n_synch]
        if decay_alpha is None or decay_beta is None:
            new_alpha = pairwise_product
            new_beta = torch.ones_like(pairwise_product)
        else:
            # 公式: S_t = r * S_{t-1} + new_input
            new_alpha = r_sigmoid * decay_alpha + pairwise_product
            new_beta = r_sigmoid * decay_beta + 1.0
        # 4. 计算同步值 (Memory Readout)
        # 加上 epsilon 防止除零
        synchronisation = new_alpha / (torch.sqrt(new_beta) + 1e-6)
        return synchronisation, new_alpha, new_beta

    def forward(self, x, h, mem_alpha, mem_beta):
        tau = F.softplus(self.tau_param) # 确保时间常数为正
        A_para = self.A_param.unsqueeze(0) # [1, Hidden]
        current_h = h
        current_alpha = mem_alpha
        current_beta = mem_beta
        # prev_sync_val = None
        # Euler Integration Loop
        
        for steps in range(self.unfolds):
            # 1. 计算基础输入驱动
            if self.global_feedback:
                inp_combined = torch.cat([x, current_h], dim=-1)
            else:
                inp_combined = x
            # 基础循环动力学
            base_drive = self.fc_input(inp_combined) # [B, Hidden]
            # 2. 计算激活状态 (用于记忆匹配)
            # 注意：论文中提到 activate state 通常是 hidden state 经过非线性变换
            activated_state = current_h
            # activated_state = torch.tanh(current_h) 
            
            # 3. 计算记忆 (Synchronization) 并更新记忆状态
            # "Memory Dynamics"
            
            sync_val, next_alpha, next_beta = self.compute_synchronisation(activated_state, current_alpha, current_beta, self.r_param)
            # 更新循环内的记忆状态
            current_alpha = next_alpha
            current_beta = next_beta
            # 4. 将记忆注入动力学 (Importance Weights)
            # memory_drive 代表记忆对当前神经元动力学的影响
            memory_drive = self.w_memory(sync_val) # [B, Hidden]
            # 5. 完整的连续时间更新方程
            
            
            # 3. ODE Update: dh/dt = -h/tau + f(total_drive)
            # f_total = Tanh(Linear_Drive + Memory_Drive)
            f_total = torch.pow(torch.sin(base_drive + memory_drive ), 2)
            
            dh = -current_h* (1/tau+f_total) + A_para * f_total
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

class NMODECTMDNXADAPTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, unfold=6, n_synch=None):
        super(NMODECTMDNXADAPTClassifier, self).__init__()
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