from models.ctrnns import CTRNN, NODE, CTGRU
from models.ltc import LTCCell
from models.cfcctmdn import CFCCTMDNClassifier
from models.ctmdn1 import CTMDNClassifier
from models.ctmdnx import CTMDNXClassifier
from models.selfattention import SelfAttentionClassifier
from models.ctmdnselfattention import CTMDNSelfAttentionClassifier
from models.cfcgatectmdn import CFCgateCTMDNClassifier
from models.nmodectmdn import nmodeCTMDNClassifier
from models.nmodeltcctmdn import nmodeLTCCTMDNClassifier
from models.nmodeltcmixctmdn import nmodeLTCMIXCTMDNClassifier
from models.ctmdnxadapt import CTMDNXADAPTClassifier
from models.nmodectmdnxadapt import NMODECTMDNXADAPTClassifier



import torch.nn as nn
import torch
# ==========================================
# 3. 模型封装 Wrapper (nn.Module)
# ==========================================

class RNNClassifier(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, output_size, unfold=6,sparsity_level=0.0):
        super(RNNClassifier, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.sparsity_level = sparsity_level
        self.cell = None
        
        if model_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type.startswith("ltc"):
            self.cell = LTCCell(input_size, hidden_size)
            # LTC settings based on script
            if model_type.endswith("_rk"):
                self.cell._solver = "rk4" # Placeholder, assuming LTCCell logic handles this or defaults
            # Note: The provided LTCCell code uses explicit or semi-implicit solver internally
        elif model_type == "node":
            self.cell = NODE(input_size, hidden_size, cell_clip=-1)
        elif model_type == "ctgru":
            self.cell = CTGRU(input_size, hidden_size,  M=unfold,cell_clip=-1)
        elif model_type == "ctrnn":
            self.cell = CTRNN(input_size, hidden_size,unfolds=unfold, cell_clip=-1, global_feedback=True)

        else:
            raise ValueError(f"Unknown model type '{model_type}'")

        # Linear head
        self.readout = nn.Linear(hidden_size, output_size)
        
        # Sparsity masks (if needed)
        self.masks = {}
        if self.sparsity_level > 0:
            self._init_sparsity_masks()

    def _init_sparsity_masks(self):
        # 简单的稀疏化实现：为每个权重矩阵生成一个 binary mask
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                # Skip LTC internal weights if strictly following original logic, 
                # but generally we mask RNN weights.
                mask = torch.bernoulli(torch.ones_like(param) * (1 - self.sparsity_level))
                self.register_buffer(f"mask_{name.replace('.', '_')}", mask)

    def apply_sparsity(self):
        if self.sparsity_level <= 0:
            return
        with torch.no_grad():
            for name, param in self.named_parameters():
                mask_name = f"mask_{name.replace('.', '_')}"
                if hasattr(self, mask_name):
                    mask = getattr(self, mask_name)
                    param.mul_(mask)

    def forward(self, x):
        # x: [Batch, SeqLen, Feats]
        B, T, F = x.size()
        if self.model_type == "lstm":
            # LSTM returns (output, (h_n, c_n))
            # output: [Batch, SeqLen, Hidden]
            output, _ = self.rnn(x)
        else:
            # Manual unrolling for custom cells
            outputs = []
            
            # Initial state
            # Note: CTGRU expects [Batch, Hidden * M] internal state, others [Batch, Hidden]
            # Based on provided CTGRU code, h_hat is viewed as [B, Hidden, M].
            # We initialize zero tensors.
            
            if self.model_type == "ctgru":
                # CTGRU state size is hidden * M (default M=8)
                M = self.cell.M
                h = torch.zeros(B, self.hidden_size * M, device=x.device)
            else:
                h = torch.zeros(B, self.hidden_size, device=x.device)
                # 对 h 应用kaiminghe 初始化
                # nn.init.kaiming_normal_(h)
            
            # Timestamp for LTC (regularly sampled)
            ts = torch.tensor(1.0, device=x.device) 

            for t in range(T):
                inp = x[:, t, :]
                
                if self.model_type.startswith("ltc"):
                    # LTCCell forward(input, hx, ts) -> next_state
                    h = self.cell(inp, h, ts)
                    curr_out = h
                elif self.model_type == "ctgru":
                    # CTGRU forward(x, state) -> h_next, state_next
                    h_out, h_next = self.cell(inp, h)
                    curr_out = h_out
                    h = h_next
                else: 
                    # CTRNN/NODE forward(x, h) -> h, h
                    h, _ = self.cell(inp, h)
                    curr_out = h
                
                outputs.append(curr_out)
            
            # Stack: [Batch, SeqLen, Hidden]
            output = torch.stack(outputs, dim=1)
        # Projection to classes
        # output: [Batch, SeqLen, Hidden] -> [Batch, SeqLen, Classes]
        logits = self.readout(output)
        return logits
    
    
def build_model(args, input_size, output_size, device):
    if args.model=="ctmdn":
        model = CTMDNClassifier(input_size, args.size, output_size, unfold=args.unfolds,n_synch=args.size).to(device)
    elif args.model=="ctmdnx":
        model = CTMDNXClassifier(input_size, args.size, output_size, unfold=args.unfolds,n_synch=args.size).to(device)
    elif args.model=="ctmdnxadapt":
        model = CTMDNXADAPTClassifier(input_size, args.size, output_size, unfold=args.unfolds,n_synch=args.size).to(device)
    elif args.model=="nmodectmdnxadapt":
        model = NMODECTMDNXADAPTClassifier(input_size, args.size, output_size, unfold=args.unfolds,n_synch=args.size).to(device)
        
    elif args.model == "attention":
        model = SelfAttentionClassifier(args.model, input_size, args.size, output_size, unfold=args.unfolds).to(device)
    elif args.model == "ctmdnattention":
        model = CTMDNSelfAttentionClassifier(args.model, input_size, args.size, output_size, unfold=args.unfolds,ticks=args.ticks).to(device)
    elif args.model == "cfcctmdn":
        model = CFCCTMDNClassifier(input_size, args.size, output_size, unfold=args.unfolds,n_synch=args.size).to(device)
    elif args.model == "nmodectmdn":
        model = nmodeCTMDNClassifier(input_size, args.size, output_size, unfold=args.unfolds,n_synch=args.size).to(device)
    elif args.model == "nmodeltcctmdn":
        model = nmodeLTCCTMDNClassifier(input_size, args.size, output_size, unfold=args.unfolds,n_synch=args.size).to(device)
    elif args.model == "nmodeltcmixctmdn":
        model = nmodeLTCMIXCTMDNClassifier(input_size, args.size, output_size, unfold=args.unfolds,n_synch=args.size).to(device)
    elif args.model == "cfcgatectmdn":
        model = CFCgateCTMDNClassifier(input_size, args.size, output_size, unfold=args.unfolds,n_synch=args.size).to(device)
    else:
        model = RNNClassifier(args.model, input_size, args.size, output_size,args.unfolds, args.sparsity).to(device)
    return model