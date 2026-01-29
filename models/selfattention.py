import torch
import torch.nn as nn
import math

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
    
# 2. [自定义] Transformer 编码层 (替代 nn.TransformerEncoderLayer)
# 你可以在这里修改内部逻辑！
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True):
        super(TransformerEncoderLayer, self).__init__()
        # --- 子模块 1: 多头自注意力 (Multi-Head Self-Attention) ---on
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
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
class SelfAttentionClassifier(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, output_size, unfold=6):
        super(SelfAttentionClassifier, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        if hidden_size % 8 == 0: nhead = 8
        elif hidden_size % 4 == 0: nhead = 4
        elif hidden_size % 2 == 0: nhead = 2
        else: nhead = 1
            
        # --- 修改点: 使用自定义的 Layer 和 Encoder ---
        # 1. 实例化单层
        # custom_encoder_layer = TransformerEncoderLayer(
        #     d_model=hidden_size, 
        #     nhead=nhead, 
        #     dim_feedforward=hidden_size * 4,
        #     dropout=0.1,
        #     batch_first=True 
        # )
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True 
        )
        # 2. 实例化整个 Encoder (堆叠 unfold 层)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=unfold)
        # self.transformer = TransformerEncoder(custom_encoder_layer, num_layers=unfold)
        
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