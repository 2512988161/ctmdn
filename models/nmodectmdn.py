import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CTMDNCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_synch=None, # 同步对的数量，默认为 hidden_size // 2
        unfolds=6,    # ODE 求解步数
        delta_t=0.1,  # 时间步长
        tau_init=1.0, # 时间常数
        global_feedback=True,
        cell_clip=-1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.unfolds = unfolds
        self.delta_t = delta_t
        self.cell_clip = cell_clip
        self.global_feedback = global_feedback
        # --- 1. 基础 ODE 参数 ---
        if global_feedback:
            self.fc_input = nn.Linear(input_size + hidden_size, hidden_size)
        else:
            self.fc_input = nn.Linear(input_size, hidden_size)
        self.tau_param = nn.Parameter(torch.tensor(tau_init))
        # --- 2. 记忆动力学参数 (Memory Dynamics) ---
        # 默认同步对数量，通常是 hidden_size 的一半或者相等，取决于配对策略
        if n_synch is None:
            self.n_synch = hidden_size 
        else:
            self.n_synch = n_synch
        # 随机生成配对索引 (一旦初始化即固定，模拟并行的稀疏连接)
        # 这里模拟 'random-pairing' 策略
        self.register_buffer('idx_left', torch.randint(0, hidden_size, (self.n_synch,)))
        self.register_buffer('idx_right', torch.randint(0, hidden_size, (self.n_synch,)))
        # [频率维度] Decay rate 'r': 控制记忆的遗忘速度
        # 使用 sigmoid 保证在 (0, 1) 之间
        self.r_param = nn.Parameter(torch.randn(self.n_synch)) 
        self.A_param = nn.Parameter(torch.randn(self.hidden_size))
        # [重要性权重维度] Importance Weights: 将同步记忆投影回隐藏层
        # 将同步信号 (size: n_synch) 映射回动力学更新 (size: hidden_size)
        self.w_memory = nn.Linear(self.n_synch, hidden_size, bias=False)
        # 初始化权重
        nn.init.xavier_uniform_(self.w_memory.weight)
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
    def forward(self, x, h, mem_alpha=None, mem_beta=None):
        """
        x: [B, input_size]
        h: [B, hidden_size]
        mem_alpha, mem_beta: [B, n_synch] - 记忆状态
        """
        tau = F.softplus(self.tau_param) # 确保时间常数为正
        A_para = self.A_param.unsqueeze(0) # [1, Hidden]
        current_h = h
        current_alpha = mem_alpha
        current_beta = mem_beta
        # ODE Solver Loop (Euler Method with Micro-steps)
        
        for _ in range(self.unfolds):
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
            
            
            # f_total = Tanh(Linear(Inp, h) + Linear(Memory))
            # f_total = torch.tanh(base_drive + memory_drive)
            # f_total = Sin^2 (Linear(Inp, h) + Linear(Memory))
            f_total = torch.pow(torch.sin(base_drive + memory_drive ), 2)
            # dh/dt = -h/tau + f(input) + g(memory)
            # dh = -current_h / tau + f_total
            
            # A：偏置参数（Bias/Steady state），代表平衡位置。 其中 A 是平衡位置，τ 是时间常数。
            #  dh/dt = -h/tau + A dot_pr f_total
            dh = -current_h/tau+ A_para * f_total
            
            # dh/dt = -(1/tau+f)*current_h + A * f_total
            # dh = -current_h* (1/tau+f_total) + A_para * f_total
            
            # ht≈ h(0)⋅e^−t/τ + τ⋅[ f(input) + g(memory)]⋅(1−e^−t/τ )
            # current_h = current_h * torch.exp(-self.delta_t / tau) \
            #     + self.delta_t * tau * f_total * (1 - torch.exp(-self.delta_t / tau))
            
            current_h = current_h + self.delta_t * dh
            
            # 裁剪以保持数值稳定
            if self.cell_clip > 0:
                current_h = torch.clamp(current_h, -self.cell_clip, self.cell_clip)
        return current_h, current_alpha, current_beta

class nmodeCTMDNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, unfold=6, n_synch=None):
        super(nmodeCTMDNClassifier, self).__init__()
        self.hidden_size = hidden_size
        # 初始化核心单元 CT-MDN Cell
        self.cell = CTMDNCell(
            input_size, 
            hidden_size, 
            n_synch=n_synch, 
            unfolds=unfold, 
            global_feedback=True
        )
        self.readout = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # x: [Batch, SeqLen, Feats]
        B, T, F = x.size()
        # 初始化状态
        h = torch.zeros(B, self.hidden_size, device=x.device)
        # 初始化记忆状态 (Alpha, Beta)
        # 初始时没有历史，设为 None，由 Cell 内部处理为初始值
        mem_alpha = None
        mem_beta = None
        
        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            # 步进：更新 hidden state 和 memory state
            h, mem_alpha, mem_beta = self.cell(inp, h, mem_alpha, mem_beta)
            outputs.append(h)
        # Stack: [Batch, SeqLen, Hidden]
        output = torch.stack(outputs, dim=1)
        # Pooling (这里取最后一个时间步，也可以做 Mean Pooling)
        # logits = self.readout(output[:, -1, :]) 
        # 如果是序列标注或需要每个时间步输出：
        logits = self.readout(output) 
        return logits
