import torch
import torch.nn as nn
import torch.nn.functional as F


class CTRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        unfolds=6,
        delta_t=0.1,
        global_feedback=False,
        # fix_tau=False,
        fix_tau=True,
        tau_init=1.0,
        cell_clip=-1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.unfolds = unfolds
        self.delta_t = delta_t
        self.global_feedback = global_feedback
        self.fix_tau = fix_tau
        self.cell_clip = cell_clip

        if global_feedback:
            self.fc = nn.Linear(input_size + hidden_size, hidden_size)
        else:
            self.fc = nn.Linear(input_size, hidden_size)

        if fix_tau:
            self.register_buffer("tau", torch.tensor(tau_init))
        else:
            self.tau_param = nn.Parameter(torch.tensor(tau_init))
        
        self.w_sync = nn.Linear(hidden_size, 1)
        
    def forward(self, x, h):
        """
        x: [B, input_size]
        h: [B, hidden_size]
        """
        B,hidden_size = h.shape
        if self.fix_tau:
            tau = self.tau
        else:
            tau = F.softplus(self.tau_param)
        # 记录每个时间步的隐藏状态
        # h_states = []
        for _ in range(self.unfolds):
            if self.global_feedback:
                inp = torch.cat([x, h], dim=-1)
            else:
                inp = x
            # h_states.append(h)
            f = torch.tanh(self.fc(inp))
            dh = -h / tau + f
            h = h + self.delta_t * dh
            if self.cell_clip > 0:
                h = torch.clamp(h, -self.cell_clip, self.cell_clip)
        # 记录最后一个时间步的隐藏状态
        # h_states.append(h)
        # [ B,unfolds+1, hidden_size]
        # h_states = torch.stack(h_states, dim=1)
        
        # conv_steps = estimate_convergence_step(h_states, eps=1e-3)
        # print("Convergence unfold step:", conv_steps[0])
        
        # # [ B,unfolds, hidden_size](h_states错位相减，第一维就不要了)
        # h_states_prime = h_states[:, 1:, :] - h_states[:, :-1, :]
        # # h_states_prime*h_states_prime^T
        # sync = torch.einsum("bth,btj->bhj", h_states, h_states)
        # # norm
        # sync = sync / (hidden_size )
        
        # sync = self.w_sync(sync).squeeze(-1)
        
        # # => [ unfolds+1,  hidden_size] => [   hidden_size , unfolds+1]再绘制曲线
        # save_tensor_trajectory(h_states[0,:, :].T, "output/ctrnn_trajectory.png")    
        # for i in range(hidden_size):
        #     save_tensor_trajectory(h_states[0,:, i], f"output/ctrnn_trajectory_{i}.png")    
        # # save_tensor_trajectory(h_states[0,:, 0], "output/ctrnn_trajectory.png")    
        # breakpoint()
        # return h , h
        return h , h
    
import torch
import matplotlib.pyplot as plt
import os
def estimate_convergence_step(
    h_states,          # [B, T, H]
    eps=1e-3,
    min_stable_steps=3
):
    """
    返回每个 batch 的收敛 unfold step（如果没收敛，返回 T-1）
    """
    # 差分
    dh = h_states[:, 1:, :] - h_states[:, :-1, :]  # [B, T-1, H]

    # 每个时间步的整体变化幅度
    dh_norm = torch.norm(dh, dim=-1)  # [B, T-1]

    conv_steps = []
    for b in range(dh_norm.shape[0]):
        for t in range(dh_norm.shape[1] - min_stable_steps):
            window = dh_norm[b, t:t + min_stable_steps]
            if torch.all(window < eps):
                conv_steps.append(t)
                break
        else:
            conv_steps.append(dh_norm.shape[1] - 1)

    return conv_steps
def save_tensor_trajectory(
    tensor: torch.Tensor,
    save_path: str,
    title: str = "Tensor Trajectory",
    xlabel: str = "Time step",
    ylabel: str = "Value",
    dpi: int = 300,
):
    assert tensor.dim() in (1, 2), \
        f"Expected 1D or 2D tensor, got shape {tensor.shape}"

    data = tensor.detach().cpu().numpy()

    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 4))

    if tensor.dim() == 1:
        # -------- 1D --------
        plt.plot(data, linewidth=2, label="trajectory")

    else:
        # -------- 2D --------
        num_traj = data.shape[0]
        for i in range(num_traj):
            plt.plot(
                data[i],
                linewidth=1.8,
                label=f"traj_{i}"
            )

        plt.legend(
            fontsize=8,
            ncol=min(4, num_traj),
            frameon=False
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()

class NODE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        unfolds=6,
        delta_t=0.1,
        cell_clip=-1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.unfolds = unfolds
        self.delta_t = delta_t
        self.cell_clip = cell_clip

        self.fc = nn.Linear(input_size + hidden_size, hidden_size)
        self.h_trajectory = []
    def f_prime(self, x, h):
        inp = torch.cat([x, h], dim=-1)
        return torch.tanh(self.fc(inp))

    def forward(self, x, h):
        for _ in range(self.unfolds):
            k1 = self.delta_t * self.f_prime(x, h)
            k2 = self.delta_t * self.f_prime(x, h + 0.5 * k1)
            k3 = self.delta_t * self.f_prime(x, h + 0.5 * k2)
            k4 = self.delta_t * self.f_prime(x, h + k3)
            h = h + (k1 + 2*k2 + 2*k3 + k4) / 6.0
            if self.cell_clip > 0:
                h = torch.clamp(h, -self.cell_clip, self.cell_clip)
            if self.training:
                h.retain_grad()
            self.h_trajectory.append(h)
        return h, h
    def get_gradient_flow(self):
        """
        提取存储在 h_trajectory 中的梯度信息
        返回: 
            steps: list [0, 1, ..., unfolds-1]
            grads: list [norm_0, norm_1, ...] (每一步梯度的平均L2范数)
        """
        grads = []
        steps = []
        
        for i, h_step in enumerate(self.h_trajectory):
            if h_step.grad is not None:
                # 计算该步梯度的 L2 范数或平均绝对值
                # 这里使用平均 L2 范数，能较好反映梯度强度
                grad_norm = h_step.grad.norm(2).item() / (h_step.numel() ** 0.5)
                grads.append(grad_norm)
                steps.append(i + 1) # 第 i+1 步
                
        return steps, grads
    
    
import math


class CTGRU(nn.Module):
    """
    Continuous-Time GRU
    https://arxiv.org/abs/1710.04110
    """

    def __init__(self, input_size, hidden_size, M=8, cell_clip=-1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.M = M
        self.cell_clip = cell_clip

        # log tau table
        ln_tau = []
        tau = 1.0
        for _ in range(M):
            ln_tau.append(math.log(tau))
            tau *= 10 ** 0.5
        self.register_buffer("ln_tau_table", torch.tensor(ln_tau))

        # gates
        self.fc_tau_r = nn.Linear(input_size + hidden_size, hidden_size * M)
        self.fc_tau_s = nn.Linear(input_size + hidden_size, hidden_size * M)
        self.fc_detect = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, state):
        """
        x:     [B, input_size]
        state: [B, hidden_size * M]
        """
        B = x.size(0)

        h_hat = state.view(B, self.hidden_size, self.M)
        h = h_hat.sum(dim=2)

        fused = torch.cat([x, h], dim=-1)

        # ----- reset gate -----
        ln_tau_r = self.fc_tau_r(fused).view(B, self.hidden_size, self.M)
        sf_r = -(ln_tau_r - self.ln_tau_table) ** 2
        rki = torch.softmax(sf_r, dim=2)

        q_input = (rki * h_hat).sum(dim=2)
        qk = torch.tanh(self.fc_detect(torch.cat([x, q_input], dim=-1)))
        qk = qk.unsqueeze(-1)

        # ----- update gate -----
        ln_tau_s = self.fc_tau_s(fused).view(B, self.hidden_size, self.M)
        sf_s = -(ln_tau_s - self.ln_tau_table) ** 2
        ski = torch.softmax(sf_s, dim=2)

        decay = torch.exp(-1.0 / self.ln_tau_table)
        h_hat_next = ((1 - ski) * h_hat + ski * qk) * decay

        if self.cell_clip > 0:
            h_hat_next = torch.clamp(h_hat_next, -self.cell_clip, self.cell_clip)

        h_next = h_hat_next.sum(dim=2)
        state_next = h_hat_next.view(B, self.hidden_size * self.M)

        return h_next, state_next
    
