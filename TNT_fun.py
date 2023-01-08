from torch import Tensor, nn
from torch.nn import functional as F

from layers import Conv1d, Res1d, Linear, LinearRes, Null
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import torch



class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        n_map = 128
        norm = "GN"
        ng = 1

        self.fc_1 = Linear(n_map, 16, norm=norm, ng=ng, act=True)
        self.fc_2 = Linear(16, 8, norm=norm, ng=ng, act=True)
        self.fc_3 = nn.Linear(8, 1, bias=False)

        # self.dropout = nn.Dropout(p=config["p_dropout"])

    def forward(self, feat):
        feat = self.fc_1(feat)
        feat = self.fc_2(feat)
        # feat = self.dropout(feat)

        feat = self.fc_3(feat)
        out = torch.sigmoid_(feat).view(-1)
        return out


class Conv1dAggreBlock(nn.Module):
    """
        Aaggregation block using max-pooling
    """

    def __init__(self, n_feat: int, dropout: float = 0.0) -> None:
        super(Conv1dAggreBlock, self).__init__()
        norm = "GN"
        ng = 1
        self.n_feat = n_feat

        self.conv_1 = Conv1d(n_feat, n_feat, kernel_size=1, norm=norm, ng=ng)
        self.conv_2 = Conv1d(n_feat*2, n_feat, kernel_size=1, norm=norm, ng=ng)

        self.aggre_func = F.adaptive_avg_pool1d

        self.conv_3 = Conv1d(n_feat, n_feat, kernel_size=1, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, feats):
        '''
            feats: (batch, c, N)
        '''
        res = feats
        feats = self.conv_1(feats)
        feats_mp, _ = feats.max(dim=1)  # global max-pooling

        feats_mp = feats_mp.unsqueeze(1).repeat((1, self.n_feat, 1))
        feats = torch.cat([feats, feats_mp], dim=1)
        feats = self.conv_2(feats)
        feats = self.dropout(feats)

        feats = self.conv_3(feats)
        feats += res
        feats = self.relu(feats)

        return feats

class GoalDecoder(nn.Module):
    def __init__(self, config, n_feat=32, n_pts=200):
        super(GoalDecoder, self).__init__()
        norm = "GN"
        ng = 1

        self.aggre_1 = Conv1dAggreBlock(n_feat=n_feat, dropout=0.1)
        self.conv_1 = Conv1d(n_feat, 8, kernel_size=1, norm=norm, ng=ng)

        self.aggre_2 = Conv1dAggreBlock(n_feat=8, dropout=0.1)
        self.conv_2 = Conv1d(8, 4, kernel_size=1, norm=norm, ng=ng)

        self.conv_3 = Conv1d(4, 1, kernel_size=1, norm=norm, ng=ng, act=False)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, feat, coord):
        '''
            feat:   (batch, N, n_feat)
            coord:  (batch, N, 2)
        '''
        feat = feat.transpose(1, 2)

        feat = self.aggre_1(feat)
        feat = self.conv_1(feat)
        feat = self.dropout(feat)

        feat = self.aggre_2(feat)
        feat = self.conv_2(feat)
        feat = self.dropout(feat)

        feat = self.conv_3(feat)

        weights = F.softmax(feat, dim=-1).transpose(1, 2)  # weights, (batch, N, 1)
        goal = torch.sum(coord * weights, dim=1)

        return goal.unsqueeze(1), weights  # (batch, 1, 2)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_k, attn_dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.d_k

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_x, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_x, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_x, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_x, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_x, bias=False)

        self.attention = ScaledDotProductAttention(d_k=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_x, eps=1e-6)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch, len_x = x.size(0), x.size(1)

        residual = x

        # Pass through the pre-attention projection: b x len_x x (n*d_v)
        # Separate different heads: b x len_x x n x d_v
        q = self.w_qs(x).view(batch, len_x, n_head, d_k)
        k = self.w_ks(x).view(batch, len_x, n_head, d_k)
        v = self.w_vs(x).view(batch, len_x, n_head, d_v)

        # Transpose for attention dot product: b x n x len_x x d_v
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        out, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x len_x x n x d_v
        # Combine the last two dimensions to concatenate all the heads together: b x len_x x (n*d_v)
        out = out.transpose(1, 2).contiguous().view(batch, len_x, -1)
        out = self.dropout(self.fc(out))
        out += residual

        out = self.layer_norm(out)

        return out, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        return x


class MultiHeadAttnEncoderLayer(nn.Module):
    def __init__(self, d_x, d_k, d_v, n_head, d_inner, dropout=0.1):
        super(MultiHeadAttnEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            n_head, d_x, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_x, d_inner, dropout=dropout)

    def forward(self, enc_input, self_attn_mask=None):
        enc_output, enc_self_attn = self.self_attn(
            enc_input, mask=self_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_self_attn


class GoalGenerator(nn.Module):
    def __init__(self, config, n_blk=2):
        super(GoalGenerator, self).__init__()
        n_mode = 6
        n_feat = 128
        norm = "GN"
        ng = 1
        self.n_blk = n_blk

        self.conv_1 = Conv1d(131, n_feat, kernel_size=1, norm=norm, ng=ng)

        self.aggre = nn.ModuleList([
            MultiHeadAttnEncoderLayer(d_x=n_feat, d_k=n_feat, d_v=n_feat, n_head=2,
                                      d_inner=n_feat, dropout=0.1)
            for _ in range(self.n_blk)])

        self.multihead_decoder = nn.ModuleList([
            GoalDecoder(config=config, n_feat=n_feat, n_pts=200) for _ in range(n_mode)
        ])

    def forward(self, score, coord, goda_feat):
        feat = torch.cat([coord, score.unsqueeze(2), goda_feat], dim=2).transpose(1, 2)  # (batch, 35, N)
        feat = self.conv_1(feat)
        feat = feat.transpose(1, 2)  # (batch, N, n_feat)

        for enc_layer in self.aggre:
            feat, _ = enc_layer(feat, self_attn_mask=None)  # (batch, N, n_feat)

        goals = []
        weights = []
        for decoder in self.multihead_decoder:
            goals_mode, weights_mode = decoder(feat, coord)
            goals.append(goals_mode)
            weights.append(weights_mode)
        goals = torch.cat(goals, dim=1)  # (batch, n_mode, 2)

        return goals

def prob_traj_output(x):
    # e.g., [batch, seq, 1] or # [batch, n_dec, seq, 1]
    muX = x[..., 0:1]  # [..., 1]
    muY = x[..., 1:2]  # [..., 1]
    sigX = x[..., 2:3]  # [..., 1]
    sigY = x[..., 3:4]  # [..., 1]
    rho = x[..., 4:5]  # [..., 1]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)

    out = torch.cat([muX, muY, sigX, sigY, rho], dim=-1)  # [..., 5]
    return out

    
class TrajCompletor(nn.Module):
    def __init__(self, config, prob_output=True):
        super(TrajCompletor, self).__init__()
        self.prob_output = prob_output
        norm = "GN"
        ng = 1

        self.fc_1 = LinearRes(130, 128, norm=norm, ng=ng)
        # self.fc_2 = LinearRes(128, 128, norm=norm, ng=ng)
        self.dropout = nn.Dropout(p=0.1)

        if self.prob_output:
            self.fc_d = nn.Linear(128, 30*5, bias=False)
        else:
            self.fc_d = nn.Linear(128, 30*2, bias=False)

    def forward(self, traj_enc, goal):
        '''
            traj_enc:   (batch, 128)
            goal:       (batch, n_mode, 2)
        '''
        n_batch = goal.shape[0]
        n_mode = goal.shape[1]
        x = torch.cat([traj_enc.unsqueeze(1).repeat((1, n_mode, 1)), goal], dim=2)

        x = x.reshape(-1, 130)
        x = self.fc_1(x)
        # x = self.fc_2(x)
        x = self.dropout(x)

        if self.prob_output:
            traj_pred = self.fc_d(x).reshape(n_batch, n_mode, 30, 5)
            traj_pred = prob_traj_output(traj_pred)
        else:
            traj_pred = self.fc_d(x).reshape(n_batch, n_mode, 30, 2)

        return traj_pred