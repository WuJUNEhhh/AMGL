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

from torch.nn.utils import init
from tqdm import tqdm

from layers import *


class VLTransformer(nn.Module):                                     # 多模态transformer：将多模态数据导入transformer
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.d_out = self.d_v * self.n_head * self.modal_num
        
        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        
        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            #encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)
            
            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout = self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout)
        
    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn, _attn2 = self.InputLayer(x)   # _attn2是增加的，防止数据处理问题
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())
        
        for i in range(self.n_layer):
            x, _attn, _attn2 = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)         # _attn2是增加的，防止数据出错
            attn = _attn.mean(dim=1)
            #x = x.transpose(1, 0)#nn.multi_head_attn
            #x, attn = self.Encoder[i](x, x, x)#nn.multi_head_attn
            #x = x.transpose(1, 0)#nn.MULTI_HEAD_ATTN
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())
           
        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden, attn_map
    
    
class VLTransformer_Gate(nn.Module):                                # VLtransformer的变形
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer_Gate, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.d_out = self.d_v * self.n_head * self.modal_num
        
        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        
        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.add_module('encode_%d' % i, encoder)  # add_module将子模块添加到当前模块。可以使用给定名称作为属性访问模块。 Args：名称（字符串）：子模块的名称。可以使用给定名称模块（Module）从此模块访问子模块：要添加到模块的子模块。
            self.Encoder.append(encoder)
            
            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        
        self.FGLayer = FusionGate(self.modal_num)    
        self.Outputlayer = OutputLayer(self.d_v * self.n_head, self.d_v * self.n_head, self.n_class)
        
    def forward(self, x):
        bs = x.size(0)
        x, attn = self.InputLayer(x)
        for i in range(self.n_layer):
            x, attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            x = self.FeedForward[i](x)
        x, norm = self.FGLayer(x)    
        x = x.sum(-2)/norm
        output, hidden = self.Outputlayer(x)
        return output, hidden

class GraphLearn(nn.Module):                                    # 图学习的算法，用来提取特征
    def __init__(self, input_dim, th, mode = 'Sigmoid-like'):
        super(GraphLearn, self).__init__()
        self.mode = mode
        self.w = nn.Linear(input_dim, 1)
        self.t = nn.Parameter(torch.ones(1))
        self.p = nn.Linear(input_dim, input_dim)
        self.threshold = nn.Parameter(torch.zeros(1))
        self.th = th
    def forward(self, x):
        initial_x = x.clone()
        num, feat_dim = x.size(0), x.size(1)
        
        if self.mode == "Sigmoid-like":
            x = x.repeat_interleave(num, dim = 0)
            x = x.view(num, num, feat_dim)
            diff = abs(x - initial_x)
            diff = diff.pow(2).sum(dim=2).pow(1/2)
            diff = (diff + self.threshold) * self.t
            output = 1 - torch.sigmoid(diff)
            
        elif self.mode == "adaptive-learning":
            x = x.repeat_interleave(num, dim = 0)
            x = x.view(num, num, feat_dim)
            diff = abs(x - initial_x)
            diff = F.relu(self.w(diff)).view(num, num)
            output = F.softmax(diff, dim = 1)
        
        elif self.mode == 'weighted-cosine':
            th = self.th
            x = self.p(x)
            x_norm = F.normalize(x,dim=-1)
            #x_norm_repeat = x_norm.repeat_interleave(num, dim = 0).view(num, num, feat_dim).detach()
            #cos_sim = torch.mul(x_norm.unsqueeze(0), x_norm_repeat)
            #score = cos_sim.sum(dim = -1)
            score = torch.matmul(x_norm, x_norm.T)
            mask = (score > th).detach().float()
            markoff_value = 0
            output = score * mask + markoff_value * (1 - mask)
        return output

class GCN(nn.Module):                                               # 图卷积模型，用来预测和调整结果的
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x3 = self.gc2(x2, adj)
        return F.log_softmax(x3, dim=1), x2


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, self.n_node*self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(GGNN, self).__init__()

        assert (opt.state_dim >= opt.annotation_dim,  \
                'state_dim must be no less than annotation_dim')

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps
        self.gc1 = GraphConv(opt.nfeat, opt.nhid)
        self.gc2 = GraphConv(opt.nhid, opt.nclass)
        self.dropout = opt.dropout


        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model     传播模型
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model          输出模型
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1),
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A)

        join_state = torch.cat((prop_state, annotation), 2)
        output = self.out(join_state)
        output = output.sum(2)
        return output

def choose_device(cuda=True):
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


class KMEANS:
    def __init__(self, n_clusters=10, max_iter=None, verbose=True, device=torch.device("cpu")):

        # self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(torch.device("cpu"))
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        init_row = torch.randint(1, x.shape[0], (self.n_clusters,)).to(self.device)
        # print(init_row.shape)    # shape 10
        init_points = x[init_row]
        # print(init_points.shape) # shape (10, 2048)
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        return self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        # print(labels.shape)  # shape (250000)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        # print(dists.shape)   # shape (0, 10)
        for i, sample in tqdm(enumerate(x)):
            # print(self.centers.shape) # shape(10, 2048)
            # print(sample.shape)       # shape 2048
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            # print(dist.shape)         # shape 10
            labels[i] = torch.argmin(dist)
            # print(labels.shape)       # shape 250000
            # print(labels[:10])
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
            # print(dists.shape)        # shape (1,10)
            # print('*')
        self.labels = labels           # shape 250000
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists              # 250000, 10
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device) # shape (0, 250000)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))  # 10, 2048
        self.centers = centers  # shape (10, 2048)

    def representative_sample(self):
        # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
        # print(self.dists.shape)
        self.representative_samples = torch.argmin(self.dists, 1)
        # print(self.representative_samples.shape)  # shape 250000
        # print('*')
        return self.representative_samples


class GAT(nn.Module):                                                 # 图注意力网络
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        self.dropout = dropout
        super(GAT, self).__init__()

        self.attentions = [GraphAttConv(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttConv(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1), x
