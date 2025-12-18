#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np
import pandas as pd
# from emvert_functional_code import emvert_ml
from scipy.sparse import csgraph
import csv
import shutil

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

import os
import csv
import numpy as np
import scipy.io as sio

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from nilearn import connectome
from collections import Counter
from nilearn import datasets
import matplotlib.pyplot as plt


# In[2]:


pipeline = 'cpac'

# Input data variables
# root_folder = '/data/zfzhu/zs/MMGL/miccai_ABIDE/data'
root_folder = '/data/ABIDE/'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal/')
# phenotype = os.path.join(root_folder, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
phenotype = os.path.join('E:/桌面/ieee3/ADNI/MMGL/MMGL/data/ABIDE/ABIDE_pcp/cpac/filt_noglobal/Phenotypic_V1_0b_preprocessed1.csv')
#phenotype = os.path.join(root_folder, 'TADPOLE/processed_standard_data.csv')TADPOLE数据集
def fetch_filenames(subject_IDs, file_type): # 填写项目ID以及文件类型

    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types
    returns:
        filenames    : list of filetypes (same length as subject_list)
    """

    import glob

    # specify file mappings for the possible file types
    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_ho': '_rois_ho.1D'}

    # The list to be filled（填写文件名）
    filenames = []

    # Fill list with requested file paths（用请求的文件路径去填写列表）
    for i in range(len(subject_IDs)):
        
        os.chdir(os.path.join(data_folder, subject_IDs[i]))  # os.path.join(data_folder, subject_IDs[i]))
        try:
            filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
        except IndexError:
            # Return N/A if subject ID is not found
            filenames.append('N/A')

    return filenames


# Get timeseries arrays for list of subjects（获得主题的时间序列数组）
def get_timeseries(subject_list, atlas_name):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200
    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

    timeseries = []
    for i in range(len(subject_list)):
        subject_folder = os.path.join(data_folder, subject_list[i])
        ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
        fl = os.path.join(subject_folder, ro_file[0])
        print("Reading timeseries file %s" %fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries


# Compute connectivity matrices（计算连接矩阵）
def subject_connectivity(timeseries, subject, atlas_name, kind, save=True, save_path=data_folder):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subject      : the subject ID
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder
    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    print("Estimating %s matrix for subject %s" % (kind, subject))

    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    if save:
        subject_file = os.path.join(save_path, subject,
                                    subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity


# Get the list of subject IDs（获得主题ID）
def get_ids(num_subjects=None):
    """
    return:
        subject_IDs    : list of all subject IDs
    """

    subject_IDs = np.genfromtxt(os.path.join(data_folder, '/subject_IDs.txt'), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


# Get phenotype values for a list of subjects（获得实验者的得分）
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]

    return scores_dict


# Dimensionality reduction step for the feature vector using a ridge classifier（用脊分类器来进行降维）
def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection
    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, step=100, verbose=1)
    print(np.shape(train_ind))
    print(train_ind)
    print("at ferature selection")

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data


# Make sure each site is represented in the training set when selecting a subset of the training set
def site_percentage(train_ind, perc, subject_list):
    """
        train_ind    : indices of the training samples
        perc         : percentage of training set used
        subject_list : list of subject IDs
    return:
        labeled_indices      : indices of the subset of training samples
    """

    train_list = subject_list[train_ind]
    sites = get_subject_score(train_list, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()
    site = np.array([unique.index(sites[train_list[x]]) for x in range(len(train_list))])

    labeled_indices = []

    for i in np.unique(site):
        id_in_site = np.argwhere(site == i).flatten()

        num_nodes = len(id_in_site)
        labeled_num = int(round(perc * num_nodes))
        labeled_indices.extend(train_ind[id_in_site[:labeled_num]])

    return labeled_indices

def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection
    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, step=100, verbose=1)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data

# Load precomputed fMRI connectivity networks（加载预计算的 fMRI 连接网络）
def get_networks(subject_list, kind, atlas_name="aal", variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks
    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder, subject,
                          subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)
    # all_networks=np.array(all_networks)

    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)

    return matrix

def create_affinity_graph_from_scores(scores, subject_list):
    """
        scores       : list of phenotypic information to be used to construct the affinity graph
        subject_list : list of subject IDs

    return:
        graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
    """

    num_nodes = len(subject_list)
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = get_subject_score(subject_list, l)

        # quantitative phenotypic scores
        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph


# In[3]:


files = ['rois_ho']
num_subjects = 871
filemapping = {'func_preproc': 'func_preproc.nii.gz',
               'rois_ho': 'rois_ho.1D'}

if not os.path.exists(data_folder): os.makedirs(data_folder)
# shutil.copyfile('./subject_IDs.txt', os.path.join(data_folder, 'subject_IDs.txt')) #要替换掉subject_IDs用自己的文件名
shutil.copyfile('./ABIDE_pcp/cpac/filt_noglobal/subject_IDs.txt', os.path.join(data_folder,
                                                                             '/subject_IDs.txt'))


# 

# In[4]:


# Download database files
abide = datasets.fetch_abide_pcp(data_dir=root_folder, n_subjects=num_subjects, pipeline=pipeline,
                                 band_pass_filtering=True, global_signal_regression=False, derivatives=files)


#subject_IDs = Reader.get_ids(num_subjects)
#subject_IDs = subject_IDs.tolist()


# In[5]:


subject_IDs = get_ids()
subject_IDs = subject_IDs.tolist()
labels = get_subject_score(subject_IDs, score='DX_GROUP')

for s, fname in zip(subject_IDs, fetch_filenames(subject_IDs, files[0])):
    subject_folder = os.path.join(data_folder, s)
    if not os.path.exists(subject_folder):
        os.mkdir(subject_folder)

    # Get the base filename for each subject
    base = fname.split(files[0])[0]

    # Move each subject file to the subject folder
    for fl in files:
        if not os.path.exists(os.path.join(subject_folder, base + filemapping[fl])):

            shutil.move(base + filemapping[fl], subject_folder)

time_series = get_timeseries(subject_IDs, 'ho')

# Compute and save connectivity matrices
for i in range(len(subject_IDs)):
        subject_connectivity(time_series[i], subject_IDs[i], 'ho', 'correlation')


# In[6]:


SEX   = get_subject_score(subject_IDs, score='SEX')
AGE   = get_subject_score(subject_IDs, score='AGE_AT_SCAN')
FIQ   = get_subject_score(subject_IDs, score='FIQ')
LABEL = get_subject_score(subject_IDs, score='DX_GROUP')
EYE_S = get_subject_score(subject_IDs, score='EYE_STATUS_AT_SCAN')
BMI   = get_subject_score(subject_IDs, score='BMI')


# In[7]:


ADS = []
Normal = []
for i in LABEL:
    if LABEL[i] == '1':
        ADS.append(i)
    elif LABEL[i] == '2':
        Normal.append(i)


# In[8]:




# Get acquisition site
sites = get_subject_score(subject_IDs, score='SITE_ID')
unique = np.unique(list(sites.values())).tolist()

num_classes = 2
num_nodes = len(subject_IDs)

# Initialise variables for class labels and acquisition sites
y_data = np.zeros([num_nodes, num_classes])
y = np.zeros([num_nodes, 1])
site = np.zeros([num_nodes, 1], dtype=np.int)

# Get class labels and acquisition site for all subjects
for i in range(num_nodes):
    y_data[i, int(labels[subject_IDs[i]])-1] = 1
    y[i] = int(labels[subject_IDs[i]])
    site[i] = unique.index(sites[subject_IDs[i]])

# Compute feature vectors (vectorised connectivity networks)
features = get_networks(subject_IDs, kind='correlation', atlas_name='ho')


# # feature selection

# In[9]:


label = y
label = label.reshape(-1)


# In[10]:


from sklearn.model_selection import KFold,StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)


# In[11]:


for train_index, test_index in skf.split(features, label):
    print(train_index)
    print(test_index)


# In[12]:


feat_256_1 = feature_selection(features, label, train_index, 256)


# In[13]:


feat_256_1_pd = pd.DataFrame(feat_256_1)
feat_256_1_pd['label'] = label
feat_256_1_pd.to_csv("E:/桌面/ieee3/ADNI/MMGL/MMGL/data/ABIDE/processed_standard_data_256_1.csv", index = False)


# In[ ]:





# In[14]:


feat_dict = {}
for i in range(1):
    feat_dict[i] = list(range(i*256,(i+1)*256))

np.save('E:/桌面/ieee3/ADNI/MMGL/MMGL/data/ABIDE/modal_feat_dict.npy',feat_dict)






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




