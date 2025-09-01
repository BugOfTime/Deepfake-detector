import torch
import torch.nn as nn
import torch.nn.functional as F


def lengths_to_mask(lengths: torch.Tensor, max_len: int = None, device=None):

    device = device or lengths.device
    max_len = int(max_len or lengths.max().item())
    rng = torch.arange(max_len, device=device).unsqueeze(0)  # [1, L]
    mask = rng >= lengths.unsqueeze(1)                       # [B, L]
    return mask


class masked_MHAFFN(nn.Module):

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1, dim_feedforward: int = None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        dim_ffn = dim_feedforward or (4 * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        y, _ = self.mha(self.ln1(x), self.ln1(x), self.ln1(x),
                        key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(y)

        z = self.ffn(self.ln2(x))
        x = x + self.dropout(z)
        return x


class additive_temporal_attention(nn.Module):
    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden, bias=True)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, key_padding_mask: torch.Tensor = None):
        e = self.v(torch.tanh(self.proj(self.dropout(H)))).squeeze(-1)  # [B, L]
        if key_padding_mask is not None:
            e = e.masked_fill(key_padding_mask, float('-inf'))
        alpha = torch.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)
        return context, alpha


class spatial_attention2D(nn.Module):
    def __init__(self, in_channels: int, temperature: float = 1.5, alpha: float = 0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        attn = self.conv(x)  # [B,1,H,W]
        attn = (attn.view(B, 1, H * W) / self.temperature).softmax(dim=-1).view(B, 1, H, W)
        weighted = (attn * x).sum(dim=(2, 3))  # [B, C]
        gap = x.mean(dim=(2, 3))
        pooled = gap + self.alpha * weighted
        return pooled, attn
