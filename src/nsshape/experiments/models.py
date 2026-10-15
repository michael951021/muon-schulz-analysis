"""Small CIFAR CNN and a conventional causal GPT, without speedrun tricks."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MatrixConv(nn.Module):
    """Store convolution kernels as matrices so Muon semantics are explicit."""
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, inputs*9))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.inputs, self.stride = inputs, stride

    def forward(self, x):
        return F.conv2d(x, self.weight.view(-1, self.inputs, 3, 3),
                        stride=self.stride, padding=1)


class CifarCNN(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.layers = nn.Sequential(
            MatrixConv(3, width), nn.GroupNorm(8, width), nn.GELU(),
            MatrixConv(width, width, 2), nn.GroupNorm(8, width), nn.GELU(),
            MatrixConv(width, width*2, 2), nn.GroupNorm(8, width*2), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width*2, 10))

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width*3, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.mlp = nn.Sequential(nn.Linear(width, width*4, bias=False), nn.GELU(),
                                 nn.Linear(width*4, width, bias=False))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [a.view(b, t, self.heads, c//self.heads).transpose(1, 2) for a in (q, k, v)]
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attention.transpose(1, 2).contiguous().view(b, t, c))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, vocab=50257, width=768, layers=12, heads=12, context=256):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*[Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        position = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token(tokens) + self.position(position)
        return self.head(self.norm(self.blocks(x)))


def parameter_groups(model):
    """Only hidden matrices go to Muon; embeddings/head and scalar params use AdamW."""
    muon, adam = [], []
    for name, parameter in model.named_parameters():
        excluded = isinstance(model, GPT) and name in {'token.weight', 'position.weight', 'head.weight'}
        (muon if parameter.ndim == 2 and not excluded else adam).append(parameter)
    return muon, adam
