import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import matplotlib
import pandas as pd
from collections import Counter

import numpy as np
import scipy.io as scio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--datname', type=str, default='ABIDE')

args = parser.parse_args()
data_name = args.datname

attn_map_list = np.load('./attn/attn_map_{}.npy'.format(data_name), allow_pickle=True)
label = np.load('./graph/{}_weighted-cosine_graph_.npz'.format(data_name))['label']
em = np.load('./graph/{}_weighted-cosine_graph_.npz'.format(data_name))['fused']



layer = 2 if data_name =="TADPOLE" else 1
labels = label
attn  = attn_map_list[layer]
attn_ = attn.reshape([attn.shape[0],-1])
_attn = attn.sum(axis=1)
print('attn shape ', attn_map_list.shape, ' layer ', layer, 'attn_ shape ', attn_.shape)

index_0 = []
index_1 = []
index_2 = []
for i in range(len(labels)):
    if labels[i] == 0:
        index_0.append(i)
    elif labels[i] == 1:
        index_1.append(i)
    else:
        index_2.append(i)


layer = 2 if data_name == 'TADPOLE' else 1
if data_name == 'ABIDE':
    attn_0 = np.mean(attn_[index_0],axis = 0)
    attn_1 = np.mean(attn_[index_1],axis = 0)

    fig = plt.figure(figsize=(20,5))
    palette = np.array(sns.color_palette("hls", 5))
    index = np.array(list(range(16)))


    fig,ax = plt.subplots(1, 2, figsize=(10,3))
    #plt.subplots_adjust(wspace=0.3) 
    # for i in range(1):
    #     #plt.subplot(2, 5, i+1)
    #     ax[0].set_ylim(0, 0.6)
    #     ax[0].bar(index[:4], [0.26,0.23,0.21,0.23], color = palette[0], align='center', label='1st Modality')
    #     ax[0].bar(index[4:8], [0.24,0.23,0.19,0.17], color = palette[1], align='center', label='2nd Modality')
    #     ax[0].bar(index[8:12], [0.14,0.22,0.19,0.21], color = palette[2], align='center', label='3rd Modality')
    #     ax[0].bar(index[12:]+0.08, [0.35,0.32,0.41,0.39], color = palette[3], align='center', label='4th Modality')
        #ax[0].set_xlabel('({})'.format(chr(97+i)), {'fontsize': 'large'})


        #plt.subplot(2, 5, i+6)
        # ax[1].set_ylim(0, 0.6)
        # ax[1].bar(index[:4], [0.28,0.23,0.25,0.34], color = palette[0], align='center', label='1st Modality')
        # ax[1].bar(index[4:8], [0.25,0.28,0.30,0.24], color = palette[1], align='center', label='2nd Modality')
        # ax[1].bar(index[8:12], [0.28,0.26,0.19,0.21], color = palette[2], align='center', label='3rd Modality')
        # ax[1].bar(index[12:]+0.08, [0.19,0.23,0.26,0.21], color = palette[3], align='center', label='4th Modality')
        #ax[1].set_xlabel('({})'.format(chr(97+i+5)), {'fontsize': 'large'})

    ax[0].set_xlabel('(a) ASD', fontsize=16)
    ax[1].set_xlabel('(b) NC', fontsize=16)

   # ax[0].set_ylabel('Attention score', fontsize=16)
    #ax[1].set_ylabel('Attention score', fontsize=16)

    for a in ax:
        a.set_xticks([])
    plt.tight_layout()
    for a in ax:
        a.tick_params(axis='both', labelsize=12)  # 或 16，与你的 ylabel 字体一致

    ax[0].legend(bbox_to_anchor=(0.6, 1.35), loc=2, borderaxespad=0, ncol=2, frameon=False, fontsize = 'x-large')
    fig.savefig('attn_avg/{}_{}_layer.pdf'.format(data_name,layer), dpi=600, format='pdf', bbox_inches = 'tight')

elif data_name == 'TADPOLE':
    attn_0 = np.mean(attn_[index_0],axis = 0)
    attn_1 = np.mean(attn_[index_1],axis = 0)
    attn_2 = np.mean(attn_[index_2],axis = 0)

    #fig = plt.figure(figsize=(100,100))
    palette = np.array(sns.color_palette("hls", 6))
    index = np.array(list(range(36)))   


    fig,ax = plt.subplots(1, 3, figsize=(20,4))
    # for i in range(1):
    #     #plt.subplot(2, 5, i+1)
    #     ax[1].set_ylim(0, 0.35)
    #     ax[1].bar(index[:4],  [0.12, 0.18, 0.15, 0.16], color = palette[0], align='center', label='1st Modality')
    #     ax[1].bar(index[4:8],  [0.08, 0.12, 0.10, 0.09], color = palette[1], align='center', label='2nd Modality')
    #     ax[1].bar(index[8:12],   [0.32, 0.33, 0.33, 0.34], color = palette[2], align='center', label='3rd Modality')
    #     ax[1].bar(index[12:16],   [0.22, 0.24, 0.25, 0.27], color = palette[3], align='center', label='4th Modality')
    #     ax[1].bar(index[16:20],   [0.11, 0.10, 0.12, 0.13], color = palette[4], align='center', label='5th Modality')
    #     ax[1].bar(index[20:24],  [0.05, 0.03, 0.05, 0.01], color = palette[5], align='center', label='6th Modality')
        #ax[0].set_xlabel('({})'.format(chr(97+i)), {'fontsize': 'large'})


        #plt.subplot(2, 5, i+6)
        # ax[0].set_ylim(0, 0.35)
        # ax[0].bar(index[:4],  [0.18, 0.12, 0.18, 0.17], color = palette[0], align='center', label='1st Modality')
        # ax[0].bar(index[4:8],  [0.19, 0.15, 0.14, 0.18], color = palette[1], align='center', label='2nd Modality')
        # ax[0].bar(index[8:12],  [0.18, 0.20, 0.16, 0.15], color = palette[2], align='center', label='3rd Modality')
        # ax[0].bar(index[12:16],  [0.14, 0.17, 0.19, 0.16], color = palette[3], align='center', label='4th Modality')
        # ax[0].bar(index[16:20],  [0.16, 0.18, 0.19, 0.15], color = palette[4], align='center', label='5th Modality')
        # ax[0].bar(index[20:24], [0.15, 0.18, 0.14, 0.19], color = palette[5], align='center', label='6th Modality')
        # #ax[1].set_xlabel('({})'.format(chr(97+i+5)), {'fontsize': 'large'})

        # ax[2].set_ylim(0, 0.35)
        # ax[2].bar(index[:4],   [0.18, 0.22, 0.19, 0.21], color = palette[0], align='center', label='1st Modality')
        # ax[2].bar(index[4:8],  [0.24, 0.26, 0.25, 0.23], color = palette[1], align='center', label='2nd Modality')
        # ax[2].bar(index[8:12],   [0.32, 0.28, 0.31, 0.29], color = palette[2], align='center', label='3rd Modality')
        # ax[2].bar(index[12:16],   [0.16, 0.14, 0.15, 0.17], color = palette[3], align='center', label='4th Modality')
        # ax[2].bar(index[16:20],   [0.07, 0.08, 0.08, 0.09], color = palette[4], align='center', label='5th Modality')
        # ax[2].bar(index[20:24], [0.03, 0.02, 0.02, 0.01], color = palette[5], align='center', label='6th Modality')

    ax[0].set_xlabel('(a) Normal', fontsize=16)
    ax[1].set_xlabel('(b) MCI', fontsize=16)
    ax[2].set_xlabel('(b) AD', fontsize=16)

   # ax[0].set_ylabel('Attention score', fontsize=16)
    #ax[1].set_ylabel('Attention score', fontsize=16)
   # ax[2].set_ylabel('Attention score', fontsize=16)
    for a in ax:
        a.set_xticks([])
    plt.tight_layout()
    for a in ax:
        a.tick_params(axis='both', labelsize=12)  # 或 16，与你的 ylabel 字体一致
    ax[0].legend(bbox_to_anchor=(1.06, 1.35), loc=2, borderaxespad=0, ncol=3, frameon=False, fontsize = 'x-large')
fig.savefig('attn_avg/{}_{}_layer_no_self.pdf'.format(data_name,layer), dpi=600, format='pdf', bbox_inches = 'tight')