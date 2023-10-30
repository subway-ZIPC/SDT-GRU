from model.spatial_attention import SpatialAttention

import torch
import torch.nn as nn
from torch.nn import Parameter, init
import torch.nn.functional as F
import math
import random
import yaml


def zoneout(prev_h, next_h, rate, training=True):
    if training:
        next_h = (1 - rate) * F.dropout(next_h - prev_h, rate) + prev_h
    else:
        next_h = rate * prev_h + (1 - rate) * next_h
    return next_h


class TemporalEmbedding(nn.Module):
    def __init__(self, num_embeddings, d_model, max_len=4):
        super(TemporalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings

        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        self.register_buffer('pe', pe)

        self.embedding_modules = nn.ModuleList([nn.Embedding(item, d_model) for item in num_embeddings])
        self.linear = nn.Linear((len(num_embeddings) + 1) * d_model, d_model)

    def forward(self, extras):
        assert len(extras) == 2 * len(self.num_embeddings)
        inputs_extras = extras[::2]
        targets_extras = extras[1::2]

        B, P = inputs_extras[0].shape
        _, Q = targets_extras[0].shape

        inputs_pe = self.pe[:P, :].expand(B, P, self.d_model)
        targets_pe = self.pe[:Q, :].expand(B, Q, self.d_model)

        inputs_extras_embedding = torch.cat([self.embedding_modules[i](inputs_extras[i])
                                             for i in range(len(self.num_embeddings))] + [inputs_pe], dim=-1)
        targets_extras_embedding = torch.cat([self.embedding_modules[i](targets_extras[i])
                                              for i in range(len(self.num_embeddings))] + [targets_pe], dim=-1)

        inputs_extras_embedding = self.linear(inputs_extras_embedding)
        targets_extras_embedding = self.linear(targets_extras_embedding)

        inputs_extras_embedding = inputs_extras_embedding.transpose(0, 1)
        targets_extras_embedding = targets_extras_embedding.transpose(0, 1)

        return inputs_extras_embedding, targets_extras_embedding


class SpatialTemporalCell(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_nodes,
                 n_heads,
                 ffn_dim,
                 st_layers,
                 st_dropout_rate,
                 output_attention):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.st_layers = st_layers
        self.st_dropout_rate = st_dropout_rate

        self.output_attention = output_attention

        self.spatial_attn_i = SpatialAttention(in_channels=in_channels,
                                               d_model=out_channels,
                                               num_nodes=num_nodes,
                                               n_heads=n_heads,
                                               ffn_dim=ffn_dim,
                                               st_layers=st_layers,
                                               st_dropout_rate=st_dropout_rate,
                                               output_attention=output_attention)
        self.spatial_attn_h = SpatialAttention(in_channels=out_channels,
                                               d_model=out_channels,
                                               num_nodes=num_nodes,
                                               n_heads=n_heads,
                                               ffn_dim=ffn_dim,
                                               st_layers=st_layers,
                                               st_dropout_rate=st_dropout_rate,
                                               output_attention=output_attention)

        self.gru_projection_i = nn.Linear(self.out_channels, self.out_channels * 3)
        self.gru_projection_h = nn.Linear(self.out_channels, self.out_channels * 3)

        self.bias_i_g = Parameter(torch.Tensor(self.out_channels))
        self.bias_r_g = Parameter(torch.Tensor(self.out_channels))
        self.bias_n_g = Parameter(torch.Tensor(self.out_channels))

        self.ln = nn.LayerNorm([self.out_channels])

        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.bias_i_g)
        init.ones_(self.bias_r_g)
        init.ones_(self.bias_n_g)

    def forward(self, x, cur_extras, prev_hidden=None):
        if prev_hidden is None:
            prev_hidden = torch.zeros(x.size(0),
                                      x.size(1),
                                      self.out_channels,
                                      dtype=x.dtype,
                                      device=x.device)
        input, input_attn = self.spatial_attn_i(x, cur_extras)
        hidden, hidden_attn = self.spatial_attn_h(prev_hidden, cur_extras)

        input_r, input_i, input_n = self.gru_projection_i(input).chunk(3, -1)
        hidden_r, hidden_i, hidden_n = self.gru_projection_h(hidden).chunk(3, -1)
        r = torch.sigmoid(input_r + hidden_r + self.bias_r_g)
        i = torch.sigmoid(input_i + hidden_i + self.bias_i_g)
        n = torch.tanh(input_n + r * hidden_n + self.bias_n_g)
        next_hidden = (1 - i) * n + i * hidden

        next_hidden = self.ln(next_hidden)
        output = next_hidden
        if self.output_attention:
            return output, next_hidden, torch.stack([input_attn, hidden_attn], dim=1)
        else:
            return output, next_hidden, None


class SDT_GRUs(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        print(model_cfg)

        self.num_rnn_layers = model_cfg['num_rnn_layers']

        self.num_input_dim = model_cfg['num_input_dim']
        self.num_rnn_units = model_cfg['num_rnn_units']
        self.num_nodes = model_cfg['num_nodes']
        self.num_extra_embedding = model_cfg['num_extra_embedding']

        self.n_heads = model_cfg['n_heads']
        self.ffn_dim = model_cfg['ffn_dim']
        self.st_layers = model_cfg['st_layers']
        self.st_dropout_rate = model_cfg['st_dropout_rate']

        self.output_attention = model_cfg['output_attention']
        self.use_curriculum_learning = model_cfg['use_curriculum_learning']
        self.cl_decay_steps = model_cfg['cl_decay_steps']
        self.use_input = model_cfg['use_input']

        self.temporal_embedding = TemporalEmbedding(self.num_extra_embedding,
                                                    self.num_rnn_units)

        self.STCell_param_extras = {
            'num_nodes': self.num_nodes,
            'n_heads': self.n_heads,
            'ffn_dim': self.ffn_dim,
            'st_layers': self.st_layers,
            'st_dropout_rate': self.st_dropout_rate,
            'output_attention': self.output_attention,
        }

        self.encoder_cells = nn.ModuleList(
            [
                SpatialTemporalCell(
                    self.num_input_dim,
                    self.num_rnn_units,
                    **self.STCell_param_extras
                )
            ] + [
                SpatialTemporalCell(
                    self.num_rnn_units,
                    self.num_rnn_units,
                    **self.STCell_param_extras
                )
                for _ in range(self.num_rnn_layers - 1)
            ]
        )

        self.decoder_cells = nn.ModuleList(
            [
                SpatialTemporalCell(
                    self.num_input_dim,
                    self.num_rnn_units,
                    **self.STCell_param_extras
                )
            ] + [
                SpatialTemporalCell(
                    self.num_rnn_units,
                    self.num_rnn_units,
                    **self.STCell_param_extras
                )
                for _ in range(self.num_rnn_layers - 1)
            ]
        )

        self.activation = F.relu
        self.output_layer = nn.Linear(self.num_rnn_units, self.num_input_dim)

        self.global_step = 0

    @staticmethod
    def inverse_sigmoid_scheduler_sampling(step, k):
        return k / (k + math.exp(step / k))

    def forward(self, x, y, extras):
        x = x.transpose(0, 1)  # x : seq_len, batch_size, num_nodes, input_dim
        y = y.transpose(0, 1)

        x_extras_embed, y_extras_embed = self.temporal_embedding(extras)

        memory = {'hidden': [None] * len(self.encoder_cells),
                  'hidden_res': None}

        # encoder
        encoder_outputs = []
        encoder_attn = []
        for t, (cur_x, cur_extras) in enumerate(zip(x, x_extras_embed)):
            memory['hidden_res'] = None

            attn_list = []
            for i, rnn_cell in enumerate(self.encoder_cells):
                output, next_hidden, attn = rnn_cell(
                    x=cur_x,
                    cur_extras=cur_extras,
                    prev_hidden=memory['hidden'][i]
                )
                if memory['hidden_res'] is not None:
                    output = output + memory['hidden_res']
                cur_x = self.activation(output)

                attn_list.append(attn)
                memory['hidden'][i] = next_hidden
                memory['hidden_res'] = output

            if self.output_attention:
                attn_list = torch.stack(attn_list, dim=1)
            encoder_outputs.append(output)
            encoder_attn.append(attn_list)
        if self.output_attention:
            encoder_attn = torch.stack(encoder_attn, dim=1)

        # decoder
        decoder_outputs = []
        decoder_attn = []
        zero_input = torch.zeros(memory['hidden'][0].size(0),
                                 memory['hidden'][0].size(1),
                                 self.num_input_dim,
                                 dtype=x.dtype,
                                 device=x.device)
        decoder_input = zero_input
        for t, cur_extras in enumerate(y_extras_embed):
            memory['hidden_res'] = None

            attn_list = []
            for i, rnn_cell in enumerate(self.decoder_cells):
                output, next_hidden, attn = rnn_cell(
                    x=decoder_input,
                    cur_extras=cur_extras,
                    prev_hidden=memory['hidden'][i]
                )
                if memory['hidden_res'] is not None:
                    output = output + memory['hidden_res']
                decoder_input = self.activation(output)

                attn_list.append(attn)
                memory['hidden'][i] = next_hidden
                memory['hidden_res'] = output

            final_out = self.output_layer(output)

            if self.output_attention:
                attn_list = torch.stack(attn_list, dim=1)
            decoder_outputs.append(final_out)
            decoder_attn.append(attn_list)

            if self.training and self.use_curriculum_learning:
                c = random.uniform(0, 1)
                if self.global_step < 1e5:
                    T = self.inverse_sigmoid_scheduler_sampling(
                        self.global_step,
                        self.cl_decay_steps)
                else:
                    T = 0
                use_truth_sequence = True if c < T else False
            else:
                use_truth_sequence = False

            if use_truth_sequence:
                decoder_input = y[t]
            else:
                decoder_input = final_out.detach()
            if not self.use_input:
                decoder_input = zero_input.detach()

        if self.output_attention:
            decoder_attn = torch.stack(decoder_attn, dim=1)
        if self.training:
            self.global_step += 1
        model_output = torch.stack(decoder_outputs)
        model_output = model_output.transpose(0, 1)
        # model_output : batch_size, seq_len, num_nodes, output_dim
        if self.output_attention:
            return model_output, torch.stack([encoder_attn, decoder_attn], 1)
        else:
            return model_output, None
