# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import copy
import os
from typing import Optional, List
from util.misc import inverse_sigmoid

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
global layer_count
layer_count = 0

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 ):

        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)        
        memory, K_weights = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # query_embed = gen_sineembed_for_position(refpoint_embed)
        num_queries = refpoint_embed.shape[0]
        if self.num_patterns == 0:
            tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed.device)
        else:
            tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1) # n_q*n_pat, bs, d_model
            refpoint_embed = refpoint_embed.repeat(self.num_patterns, 1, 1) # n_q*n_pat, bs, d_model
            # import ipdb; ipdb.set_trace()
        hs, references, Q_weights, C_weights = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
        return hs, references, Q_weights, K_weights, C_weights


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        inter_K_weights = list()
        for layer_id, layer in enumerate(self.layers):
            # rescale the content and pos sim
            pos_scales = self.query_scale(output)
            output, K_weights = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos*pos_scales)
            inter_K_weights.append(K_weights)

        if self.norm is not None:
            output = self.norm(output)

        return output, torch.stack(inter_K_weights)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, 
                    d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                    modulate_hw_attn=False,
                    bbox_embed_diff_each_layer=False,
                    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        
        self.bbox_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer


        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        
        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        inter_Q_weights = []
        inter_C_weights = []

        # import ipdb; ipdb.set_trace()        

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]     # [num_queries, batch_size, 2]
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)  
            query_pos = self.ref_point_head(query_sine_embed) 

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation

            # modulated HW attentions
            if self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid() # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)


            output, Q_weights, C_weights = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))

            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                inter_Q_weights.append(Q_weights)
                inter_C_weights.append(C_weights)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                    torch.stack(inter_Q_weights),
                    torch.stack(inter_C_weights),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2), 
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, K_weights = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                         key_padding_mask=src_key_padding_mask)

        # print(torch.argsort(-K_weights[0].detach().cpu(), dim=-1)[:10, :10].numpy())
        # print(torch.max(K_weights[0].detach().cpu(), dim=-1)[0][:10].numpy())

        # mma = K_weights[0].detach().cpu().numpy()
        # fig, ax = plt.subplots()
        # heatmap = ax.pcolor(mma, cmap=plt.cm.Reds)
        # # put the major ticks at the middle of each cell
        # ax.set_xticks(np.arange(mma.shape[1]) + 0.5, minor=False)
        # ax.set_yticks(np.arange(mma.shape[0]) + 0.5, minor=False)
        #
        # ax.set_xlim(0, int(mma.shape[1]))
        # ax.set_ylim(0, int(mma.shape[0]))
        #
        # ax.invert_yaxis()
        # ax.xaxis.tick_top()

        # ax.set_xticklabels(source_labels, minor=False)
        # ax.set_yticklabels(target_labels, minor=False)
        # plt.xticks(rotation=45)

        # global layer_count
        # plt.savefig("{}.png".format(layer_count + 1))
        # layer_count += 1
        # # fig.canvas.draw()
        # # vis_array = np.array(fig.canvas.renderer._renderer)
        # plt.close(fig)

        # global layer_count
        # layer_count += 1
        # K_weights = K_weights[0].detach().cpu()
        # # df = pd.DataFrame(K_weights)
        # # df.to_csv("K_{:02d}.csv".format(layer_count + 1), index=False)
        #
        # # map = K_weights.view(25, 34, 25, 34).mean(dim=0).sum(dim=1)
        # # map = K_weights.view(25, 34, 25, 34).mean(dim=1).sum(dim=2)
        # # map = map / torch.sum(map, dim=-1, keepdim=True)
        # map = K_weights
        # map = map.numpy()
        # H, W = map.shape
        # H_labels = ["{}".format(x) for x in range(1, H + 1, 1)]
        # W_labels = ["{}".format(x) for x in range(1, W + 1, 1)]
        # # map -= np.min(map)
        # # map /= np.max(map)
        # df = pd.DataFrame(map, H_labels, W_labels)
        # ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True)
        # # ax.set(xlabel="", ylabel="")
        # # tl = ax.get_xticklabels()
        # # ax.set_xticklabels(tl, rotation=90)
        # # tly = ax.get_yticklabels()
        # # ax.set_yticklabels(tly, rotation=0)
        # plt.savefig("K_{:02d}.png".format(layer_count))
        # plt.close()

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, K_weights


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
                     
        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2, Q_weights = self.self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            # ========== End of Self-Attention =============

            # print(torch.argsort(-Q_weights[0].detach().cpu(), dim=-1)[:10, :10].numpy())
            # print(torch.max(Q_weights[0].detach().cpu(), dim=-1)[0][:10].numpy())

            # print(torch.argsort(-Q_weights[0].detach().cpu(), dim=-1)[:10, :10].numpy())
            # Q_weights = Q_weights.detach().cpu()
            # print(torch.argsort(-Q_weights[0, 0].detach().cpu(), dim=-1)[:10].numpy())
            # print(Q_weights[0, 0][torch.argsort(-Q_weights[0, 0], dim=-1)[:10]].numpy())

            # print(F.cross_entropy(Q_weights, Q_weights).sum(-1).mean().detach().cpu().numpy())

            # q = q_content
            # k = k_content
            # _, C_weights = self.self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            # q = q_pos
            # k = k_pos
            # _, P_weights = self.self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            #
            # N, Q, _ = Q_weights.shape
            # Q_C = torch.bmm(F.normalize(Q_weights.flatten(1)).unsqueeze(-2),
            #                 F.normalize(C_weights.flatten(1)).unsqueeze(-1)).mean()
            # Q_P = torch.bmm(F.normalize(Q_weights.flatten(1)).unsqueeze(-2),
            #                 F.normalize(P_weights.flatten(1)).unsqueeze(-1)).mean()
            #
            # print(Q_C.detach().cpu().numpy(), Q_P.detach().cpu().numpy())

            # global layer_count
            # layer_count += 1
            # Q_weights = Q_weights[0].detach().cpu()
            #
            # # map = Q_weights.view(30, 30, 30, 30).mean(dim=1).sum(dim=2)
            # # map = Q_weights.view(3, 10, 10, 3, 10, 10).mean(dim=(0, 1)).sum(dim=(1, 2))
            # # map = Q_weights.view(3, 100, 3, 100).mean(dim=0).sum(dim=1)
            # # map = map / torch.sum(map, dim=-1, keepdim=True)
            # map = Q_weights[:40, :40]
            # # # map = K_weights.view(25, 34, 25, 34).mean(dim=1).sum(dim=2)
            # # map = map / torch.sum(map, dim=-1, keepdim=True)
            # # map = Q_weights
            #
            # map = map.numpy()
            #
            # H, W = map.shape
            # H_labels = ["{}".format(x) for x in range(1, H + 1, 1)]
            # W_labels = ["{}".format(x) for x in range(1, W + 1, 1)]
            # # map -= np.min(map)
            # # map /= np.max(map)
            # # df = pd.DataFrame(map, H_labels, W_labels)
            # df = pd.DataFrame(map, H_labels, W_labels)
            # ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True)
            # # ax.set(xlabel="", ylabel="")
            # # tl = ax.get_xticklabels()
            # # ax.set_xticklabels(tl, rotation=90)
            # # tly = ax.get_yticklabels()
            # # ax.set_yticklabels(tly, rotation=0)
            # plt.savefig("Q_{:02d}.png".format(layer_count))
            # plt.close()

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2, C_weights = self.cross_attn(query=q,
                                          key=k,
                                          value=v, attn_mask=memory_mask,
                                          key_padding_mask=memory_key_padding_mask)
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, Q_weights, C_weights



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=4,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
