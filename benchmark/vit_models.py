# models.py
import torch
import crypten
import crypten.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.projection = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = crypten.nn.functional.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(x.shape)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        x = crypten.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, 3, dim)

        self.cls_token = nn.Parameter(crypten.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(crypten.randn(1, 1 + image_size // patch_size, dim))

        self.blocks = nn.ModuleList([Block(dim, heads, mlp_dim) for _ in range(depth)])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_token = self.cls_token.expand(b, -1, -1)
        x = crypten.cat((cls_token, x), dim=1)
        x += self.pos_embedding[:, :n + 1]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls = x[:, 0]
        out = self.head(cls)
        return out