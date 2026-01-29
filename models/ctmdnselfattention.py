import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# 1. 位置编码 (保持不变)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# 2. [新增] 自定义多头注意力 (替代 nn.MultiheadAttention)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, batch_first=True,ticks=6):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.batch_first = batch_first
        # 定义 Q, K, V 的投影层
        # 这里拆分成三个独立的层，方便你在 forward 里单独修改某一个
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # 输出投影层
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim) # 缩放因子 1/sqrt(dk)
        
        # 时间常数参数化 (Sigmoid 限制范围或 Softplus)
        tau_init=1.0
        print(d_model)
        n_synch = d_model
        self.ticks = ticks
        self.delta_t = 1.0 / self.ticks
        self.tau_param = nn.Parameter(torch.tensor(tau_init))
        self.n_synch = n_synch
        self.register_buffer('idx_left', torch.randint(0, d_model, (self.n_synch,)))
        self.register_buffer('idx_right', torch.randint(0, d_model, (self.n_synch,)))
        self.r_param = nn.Parameter(torch.randn(self.n_synch)) 
        self.w_memory = nn.Parameter(torch.randn(num_heads, 1, 1))
        # self.w_memory = nn.Linear(self.n_synch, d_model, bias=False)
        
        
        
    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r):
        """
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
        # 我们认为，attn_scores 就相当于 激活匹配值，从中选取 n_synch 个 
        # pairwise_product = activated_state[:,:, self.idx_left,self.idx_right]   # [B,  HeadDim,n_synch,n_synch]
        # 3. 记忆动力学更新 (Memory Update)
        # 使用 sigmoid(r) 确保衰减率在 0-1 之间
        r_sigmoid = torch.sigmoid(r).unsqueeze(0) # [1, n_synch]
        if decay_alpha is None or decay_beta is None:
            new_alpha = activated_state
            new_beta = torch.ones_like(activated_state)
        else:
            # 公式: S_t = r * S_{t-1} + new_input
            new_alpha = r_sigmoid * decay_alpha + activated_state
            new_beta = r_sigmoid * decay_beta + 1.0
        # 4. 计算同步值 (Memory Readout)
        # 加上 epsilon 防止除零
        synchronisation = new_alpha / (torch.sqrt(new_beta) + 1e-6)
        return synchronisation, new_alpha, new_beta
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # query, key, value 通常在自注意力中是同一个张量 x
        # x shape: [Batch, SeqLen, d_model] (因为 batch_first=True)
        B, T, _ = query.size() # Batch, Time, Dim
        # 1. 线性投影 [Batch, SeqLen, d_model]
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        # 2. 拆分 Head [Batch, SeqLen, NumHeads, HeadDim]
        # 并转置为 [Batch, NumHeads, SeqLen, HeadDim] 以便矩阵乘法
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # 3. 计算注意力分数 (Scaled Dot-Product Attention)
        # Scores shape: [Batch, NumHeads, SeqLen, SeqLen]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_scores_init = attn_scores.clone() 
        decay_alpha = None
        decay_beta = None
        
        tau = F.softplus(self.tau_param) 
        for _ in range(self.ticks):
            synchronisation, decay_alpha, decay_alpha = self.compute_synchronisation(
                activated_state=attn_scores,
                decay_alpha=decay_alpha,
                decay_beta=decay_beta,
                r=self.r_param
            )
            # decay_alpha = new_alpha
            # decay_beta = new_beta
            # print(attn_scores.shape)
            memory_drive  = synchronisation * self.w_memory
            # memory_drive = self.w_memory(synchronisation)
            f_total = torch.tanh(attn_scores + memory_drive)
            dh = -attn_scores / tau + f_total
            attn_scores = attn_scores + dh*self.delta_t
            attn_scores = torch.clamp(attn_scores, -1.0, 1.0)
        attn_scores = attn_scores + attn_scores_init
        # dA/dt = -A/tau + f(UA+WMemory+b)
        # A = A0 + delta_t * dA/dt
        
        # --- 如果要做 CTMDN (Continuous Time)，通常就在这里魔改 ---
        # 例如: attn_scores = attn_scores - time_decay_matrix
        
        # 4. 处理 Mask (如果有)
        if attn_mask is not None:
            # attn_mask 通常是 [SeqLen, SeqLen]
            attn_scores = attn_scores.masked_fill(attn_mask == 1, float('-inf'))
        if key_padding_mask is not None:
            # key_padding_mask 通常是 [Batch, SeqLen]
            # 需要扩展维度以匹配 scores
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # [Batch, 1, 1, SeqLen]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        # 5. Softmax & Dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        # 6. 加权求和
        # [Batch, NumHeads, SeqLen, SeqLen] @ [Batch, NumHeads, SeqLen, HeadDim]
        # -> [Batch, NumHeads, SeqLen, HeadDim]
        output = torch.matmul(attn_probs, V)
        # 7. 拼接 Heads 并还原形状
        # [Batch, NumHeads, SeqLen, HeadDim] -> [Batch, SeqLen, NumHeads, HeadDim]
        output = output.transpose(1, 2).contiguous()
        # -> [Batch, SeqLen, d_model]
        output = output.view(B, T, self.d_model)
        # 8. 输出投影
        output = self.out_proj(output)
        # 返回 output 和 attention权重 (方便可视化)
        return output, attn_probs

# 2. [自定义] Transformer 编码层 
# 你可以在这里修改内部逻辑！
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True,ticks=6):
        super(TransformerEncoderLayer, self).__init__()
        # --- 子模块 1: 多头自注意力 (Multi-Head Self-Attention) ---
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,ticks=ticks)
        # --- 子模块 2: 前馈神经网络 (Feed Forward Network) ---
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # --- 正则化与 Dropout ---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # 激活函数 (通常是 ReLU 或 GELU)
        self.activation = nn.ReLU()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: [Batch, SeqLen, d_model] (如果 batch_first=True)
        # --- 第一部分: Attention + Residual + Norm ---
        # 1. 计算自注意力
        
        # 注意: nn.MultiheadAttention 返回 (attn_output, attn_weights)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, 
                              key_padding_mask=src_key_padding_mask)[0]
        
        
        # 2. Add & Norm (Post-LN 结构: 先 Dropout -> 残差相加 -> Norm)
        # 如果你想改成 Pre-LN，请调整这里的顺序
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # --- 第二部分: FeedForward + Residual + Norm ---
        # 3. 前馈网络: Linear -> Activation -> Dropout -> Linear
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 4. Add & Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# 3. [自定义] Transformer 编码器 (替代 nn.TransformerEncoder)
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        # 使用 ModuleList 堆叠多层
        # 这里使用了深拷贝(clones)的思想，将传入的 layer 复制 num_layers 份
        self.layers = nn.ModuleList([self._get_cloned_layer(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
    def _get_cloned_layer(self, layer):
        # 简单地创建新实例或深拷贝
        import copy
        return copy.deepcopy(layer)
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        # 依次通过每一层
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output
    
# 4. 主模型
class CTMDNSelfAttentionClassifier(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, output_size, unfold=6,ticks=6):
        super(CTMDNSelfAttentionClassifier, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        if hidden_size % 8 == 0: nhead = 8
        elif hidden_size % 4 == 0: nhead = 4
        elif hidden_size % 2 == 0: nhead = 2
        else: nhead = 1
        
        # --- 修改点: 使用自定义的 Layer 和 Encoder ---
        # 1. 实例化单层
        custom_encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            ticks=ticks
        )
        
        # 2. 实例化整个 Encoder (堆叠 unfold 层)
        self.transformer = TransformerEncoder(custom_encoder_layer, num_layers=unfold)
        
        self.readout = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: [Batch, SeqLen, Feats]
        x = self.input_proj(x) 
        x = self.pos_encoder(x)
        # 调用自定义 Transformer
        x = self.transformer(x) # [Batch, SeqLen, d_model]
        # --- 修正注意点 ---
        last_output = x # 取序列最后一个点: [Batch, d_model]
        logits = self.readout(last_output) # [Batch, Classes]
        return logits