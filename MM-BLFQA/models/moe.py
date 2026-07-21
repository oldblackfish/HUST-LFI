import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Expert Network
# ============================================================

class Projection(nn.Module):
    """
    Single expert

    Input : [B, C]
    Output: [B, out_dim]
    """

    def __init__(self, in_channels, hidden_channels, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Sparse Mixture of Experts (Fast Version)
# ============================================================

class MoE(nn.Module):
    """
    Generic Top-K Mixture of Experts

    Args:
        channels     : input feature dim
        num_experts  : number of experts
        topk         : active experts per sample
        out_dim      : output dimension
    """

    def __init__(
        self,
        channels,
        num_experts=4,
        topk=2,
        out_dim=1
    ):
        super().__init__()

        assert topk <= num_experts

        self.num_experts = num_experts
        self.topk = topk
        self.out_dim = out_dim

        # ---------------- Gating Network ----------------
        # outputs logits (NOT softmax here)
        self.gate = nn.Linear(channels, num_experts)

        # ---------------- Experts ----------------
        self.experts = nn.ModuleList([
            Projection(channels, channels, out_dim)
            for _ in range(num_experts)
        ])

    # ----------------------------------------------------
    def forward(self, f_dis):
        """
        f_dis : [B, C]

        return : [B, out_dim]
        """

        B = f_dis.size(0)

        # ==================================================
        # 1. Gating probabilities
        # ==================================================
        logits = self.gate(f_dis)             # [B,E]
        gate_prob = F.softmax(logits, dim=1)  # [B,E]

        # ==================================================
        # 2. Top-K routing
        # ==================================================
        topk_val, topk_idx = torch.topk(
            gate_prob,
            self.topk,
            dim=1
        )  # [B,K]

        # normalize weights (important!)
        topk_val = topk_val / topk_val.sum(dim=1, keepdim=True)

        # ==================================================
        # 3. Compute ALL expert outputs (parallel)
        # ==================================================
        # result: [B,E,out_dim]
        expert_outputs = torch.stack(
            [expert(f_dis) for expert in self.experts],
            dim=1
        )

        # ==================================================
        # 4. Select routed experts
        # ==================================================
        gather_index = topk_idx.unsqueeze(-1).expand(
            -1, -1, self.out_dim
        )

        selected_outputs = torch.gather(
            expert_outputs,
            dim=1,
            index=gather_index
        )  # [B,K,out_dim]

        # ==================================================
        # 5. Weighted fusion
        # ==================================================
        pred = torch.sum(
            selected_outputs * topk_val.unsqueeze(-1),
            dim=1
        )  # [B,out_dim]

        return pred