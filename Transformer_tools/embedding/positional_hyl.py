import torch
from torch import nn

class PostionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        super(PostionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        #  原错误：
        # batch_size, seq_len = x.size()    # ← 改掉

        batch_size, seq_len, _ = x.size()   # ✓ ← 修改1：x 是 3 维，所以要解包 3 个维度

        # 原错误：
        # return self.encoding[:seq_len, :]  # ← 改掉

        # ✓ ← 修改2：加上 batch 维（unsqueeze(0)），才能与 x 相加
        return self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        # shape = [batch, seq_len, d_model]
