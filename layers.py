import argparse
import os
import pickle
import random
import sys
import tempfile
import time

import gc
import matplotlib.cm
import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from torch.nn import init


class Attention(nn.Module):                                     # 自注意力模块，外部注意力可以在这里加入
    def __init__(self, temperature, attn_dropout=0.1):          # temperature 是一个温度系数，当t很大时，我们得到的loss比较大，避免了局部最优解的问题，当t较小时，我们得到的实验结果更加准确
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)                 # 防止过拟合

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))# transpose矩阵转置a
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # softmax+dropout
        attn = attn/abs(attn.min())                             # abs返回参数的绝对值
        attn = self.dropout(F.softmax(F.normalize(attn, dim=-1), dim=-1))
        #attn = self.dropout(F.softmax(attn, dim=-1))
        # 概率分布xV
        output = torch.matmul(attn, v)

        return output, attn, v

class ExternalAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)  # nn.Parameter(d_model, S)
        self.mv = nn.Linear(S, d_model, bias=False)  # nn.Parameter(S, d_model)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries) #bs,n,S                 batch_size, pixel, cahnnel  # attn = queries @ self.mk
        attn = self.softmax(attn) #bs,n,S
        attn = attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out = self.mv(attn) #bs,n,d_model  # out = attn @ slef.mv

        return out


class FeedForwardLayer(nn.Module): # 前馈传播层

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # 两个fc层（全连接层），对最后的512维度进行变换
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):               # 向前传播
        residual = x

        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x    
    
class VariLengthInputLayer(nn.Module):                                      # 确定长度的输入层
    def __init__(self, input_data_dims, d_k, d_v, n_head, dropout):         # 输入数据的维度、键、值、有几层注意力机制、防过拟合
        super(VariLengthInputLayer, self).__init__()
        self.n_head = n_head
        self.dims = input_data_dims
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = []
        self.w_ks = []
        self.w_vs = []
        for i, dim in enumerate(self.dims):
            self.w_q = nn.Linear(dim, n_head * d_k, bias = False)
            self.w_k = nn.Linear(dim, n_head * d_k, bias = False)
            self.w_v = nn.Linear(dim, n_head * d_v, bias = False)
            self.w_qs.append(self.w_q)
            self.w_ks.append(self.w_k)
            self.w_vs.append(self.w_v)
            self.add_module('linear_q_%d_%d' % (dim, i), self.w_q)
            self.add_module('linear_k_%d_%d' % (dim, i), self.w_k)
            self.add_module('linear_v_%d_%d' % (dim, i), self.w_v)
        
        self.attention = Attention(temperature=d_k**0.5, attn_dropout=dropout)          # 注意力机制添加的地方
        self.attention2 = ExternalAttention(d_model=128, S=128)                      #
        self.fc = nn.Linear(n_head * d_v, n_head * d_v)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-10)
    
    def forward(self, input_data, mask=None):                                           # 数据向前传播
        """
        输入的向量是各个模态concatenate起来的
        """
        temp_dim = 0
        bs = input_data.size(0)
        modal_num = len(self.dims)
        q = torch.zeros(bs, modal_num, self.n_head * self.d_k).cuda()
        k = torch.zeros(bs, modal_num, self.n_head * self.d_k).cuda()
        v = torch.zeros(bs, modal_num, self.n_head * self.d_v).cuda()
        for i in range(modal_num):
            w_q = self.w_qs[i]
            w_k = self.w_ks[i]
            w_v = self.w_vs[i]
            
            
            data = input_data[:, temp_dim : temp_dim + self.dims[i]]
            temp_dim += self.dims[i]
            q[:,i,:] = w_q(data)
            k[:,i,:] = w_k(data)
            v[:,i,:] = w_v(data)
            
        q = q.view(bs, modal_num, self.n_head, self.d_k)
        k = k.view(bs, modal_num, self.n_head, self.d_k)
        v = v.view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, residual = self.attention(q, k, v)#注意因为没有同输入相比维度发生变化，因此以v作为残差  # 这里也可以修改一下
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        residual = residual.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        q1 = self.attention2(q)
        return q, attn, q1
    
class EncodeLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, dropout):
        super(EncodeLayer, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)
        
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = Attention(temperature=d_k ** 0.5)
        self.attention2 = ExternalAttention(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-10)
        
    def forward(self, q, k, v, modal_num, mask = None):
        bs = q.size(0)
        residual = q
        q = self.w_q(q).view(bs, modal_num, self.n_head, self.d_k)
        k = self.w_k(k).view(bs, modal_num, self.n_head, self.d_k)
        v = self.w_v(v).view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        q, attn, _ = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        q1 = self.attention2(q)
        q = (q+q1)
        return q, attn, q1

class OutputLayer(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes, modal_num, dropout = 0.5):
        super(OutputLayer, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(d_hidden + modal_num**2, n_classes)
    def forward(self, x, attn_embedding):
        x = self.mlp_head(x)
        combined_x = torch.cat((x, attn_embedding), dim=-1)
        output = self.classifier(combined_x)
        return F.log_softmax(output, dim=1), combined_x
    

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        #if bias:
        #    self.bias = nn.Parameter(torch.FloatTensor(out_features))
        #else:
        #    self.register_parameter('bias', None)
            
    #def reset_parameters(self):
    #    stdv = 1. / math.sqrt(self.W.size(1))
    #    self.W.data.uniform_(-stdv, stdv)
    #    if self.bias is not None:
    #        self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input, adj):
        support = torch.mm(input, self.W)
        output = torch.mm(adj, support)
        #if self.bias is not None:
        #    return output + self.bias
        #else:
        return output

class FusionGate(nn.Module):
    def __init__(self, channel, reduction=1):
        super(FusionGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x), y.sum(-2)

class GraphAttConv(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttConv, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.attention_w = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.attention_w.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        hidden = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        attention_input = self._prepare_attentional_mechanism_input(hidden)
        e = self.leakyrelu(torch.matmul(attention_input, self.attention_w).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e * adj, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, hidden)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
    def _prepare_attentional_mechanism_input(self, Wh):
            N = Wh.size(0) # number of nodes
            Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
            Wh_repeated_alternating = Wh.repeat(N, 1)
            all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

            return all_combinations_matrix.view(N, N, 2 * self.out_features)
