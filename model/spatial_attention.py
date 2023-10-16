from .embedding import TokenEmbedding, PositionalEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, output_attention):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads

        self.output_attention = output_attention

        self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)

        self.out_projection = nn.Linear(self.d_values * n_heads, d_model)
        self.value_projection = nn.Linear(d_model, self.d_values * n_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, L, H, -1)
        values = self.value_projection(x).view(B, L, H, -1)

        E = queries.shape[-1]

        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        V = self.out_projection(V.reshape(B, L, -1))
        if self.output_attention:
            return V, torch.softmax(scale * scores, dim=-1).cpu().detach()
        else:
            return V, None


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, ffn_dim, activation, dropout):
        super().__init__()
        self.d_model = d_model
        self.ffn_dim = ffn_dim

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=ffn_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ffn_dim, out_channels=d_model, kernel_size=1)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        x = self.dropout(self.conv2(x).transpose(-1, 1))
        return x


class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 num_nodes,
                 n_heads,
                 ffn_dim,
                 st_layers,
                 st_dropout_rate,
                 output_attention):
        super().__init__()

        self.d_model = d_model
        self.num_nodes = num_nodes
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.st_layers = st_layers
        self.st_dropout_rate = st_dropout_rate
        self.output_attention = output_attention

        self.attention = AttentionLayer(d_model, n_heads, st_dropout_rate, output_attention)
        self.attention_dropout = nn.Dropout(st_dropout_rate)

        self.activation = F.gelu

        self.ffn = FeedForwardModule(d_model=d_model,
                                     ffn_dim=ffn_dim,
                                     activation=self.activation,
                                     dropout=st_dropout_rate)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        new_x, attn = self.attention(x)

        x = x + self.attention_dropout(new_x)

        x = x + self.ffn(x)

        x = self.norm(x)

        return x, attn


class SpatialAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 d_model,
                 num_nodes,
                 n_heads,
                 ffn_dim,
                 st_layers,
                 st_dropout_rate,
                 output_attention):
        super().__init__()

        self.in_channels = in_channels
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.st_layers = st_layers
        self.st_dropout_rate = st_dropout_rate
        self.output_attention = output_attention

        self.val_embedding = TokenEmbedding(in_channels, d_model)
        self.pos_embedding = PositionalEmbedding(d_model)

        self.encoders = nn.ModuleList(
            [
                Encoder(d_model=d_model,
                        num_nodes=num_nodes,
                        n_heads=n_heads,
                        ffn_dim=ffn_dim,
                        st_layers=st_layers,
                        st_dropout_rate=st_dropout_rate,
                        output_attention=output_attention)
                for _ in range(st_layers)
            ]
        )

    def forward(self, x, cur_extras):
        attn_list = []
        src = self.val_embedding(x) + self.pos_embedding(x) + cur_extras[:, None, :]
        for encoder in self.encoders:
            src, attn = encoder(src)
            attn_list.append(attn)

        if self.output_attention:
            attn_list = torch.stack(attn_list).transpose(0, 1)
        return src, attn_list
