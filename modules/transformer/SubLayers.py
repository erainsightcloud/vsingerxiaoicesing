from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output

class MixtureOfExperts(nn.Module):
    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1, num_experts=10, num_cases=2):
        super(MixtureOfExperts, self).__init__()
        self.d_in = d_in
        self.layer_norm = nn.LayerNorm(d_in)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, d_hid),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hid, d_in)
            ) for _ in range(num_experts)
        ])
        self.shared = nn.Sequential(
            nn.Conv1d(d_in, d_hid, kernel_size[0], padding="same"),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_hid, d_in, kernel_size[1], padding="same")
        )
        self.num_experts = num_experts
        self.num_cases = num_cases
        self.route_critic = nn.Sequential(
            nn.Conv1d(d_in, d_in, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_in, num_experts, 3, padding=1)
        )
    # x: [*, Len, Feat]
    def forward(self, x):
        residual = x
        route_critic = self.route_critic(x.transpose(1, 2)).transpose(1, 2).reshape(-1, self.num_experts) # [*, Experts]
        x = x.reshape(-1, self.d_in) # [*, Feat]
        route_value, route_indice = torch.topk(route_critic, self.num_cases, dim=-1) # [*, Cases]
        route_value = F.softmax(route_value, dim=-1) # [*, Cases]
        route_pred = torch.full_like(route_critic, 0.0)
        route_pred = torch.scatter(route_pred, 1, route_indice, route_value) # [*, Experts]
        y = torch.full_like(x, 0.0)
        if torch.onnx.is_in_onnx_export():
            for expert_id in range(self.num_experts):
                y = y + self.experts[expert_id](x) * route_pred[:, expert_id, None]
        else:
            for expert_id in range(self.num_experts):
                mask = torch.any(route_indice == expert_id, dim=-1) # [*]
                index = torch.nonzero(mask) # [*, 1]
                if index.shape[0] < 0:
                    continue
                expert_input = torch.gather(x, 0, index.repeat(1, self.d_in)) # [*, Feat]
                expert_weight = torch.gather(route_pred[:, None, expert_id], 0, index) # [*, Feat]
                expert_output = self.experts[expert_id](expert_input) # [*, Feat]
                y = torch.scatter_add(y, 0, index.repeat(1, self.d_in), expert_output * expert_weight) # [*, Feat]
        y = y.reshape(-1, residual.shape[1], self.d_in) # [*, Len, Feat]
        return self.layer_norm(residual + y + self.shared(x.transpose(1, 2)).transpose(1, 2))