import numpy as np
import gc
import os
import re
from collections import defaultdict

import joblib
import networkx as nx
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .data_loader import split_test, get_ui_dict, get_uu_dict


def load_ds_recfm_rate(
        args,
        dataset='Epinions',
        batch_size=1024,
        test_size=0.2,
        split='fo',
        seed=27,
        item_maxlen=20,
        nbrs_maxlen=20,
        num_worker=0):

    if dataset == 'Ciao':
        rating_mat = loadmat(f'../datasets/Ciao/rating_with_timestamp.mat')
        rating = rating_mat['rating']
        uu_elist = loadmat(f'../datasets/{args.dataset}/trust.mat')['trust']

    elif dataset == 'Epinions':
        rating_mat = loadmat(f'../datasets/Epinions/rating_with_timestamp.mat')
        rating = rating_mat['rating_with_timestamp']
        uu_elist = loadmat(f'../datasets/{args.dataset}/trust.mat')['trust']

    df = pd.DataFrame(data=rating, columns=['user', 'item', 'cate', 'rate', 'help', 'ts'])
    df.drop(columns=['help'], inplace=True)
    df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
    df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
    df.drop(columns=['ts'], inplace=True)

    user_num = df['user'].max() + 1
    item_num = df['item'].max() + 1
    cate_num = df['cate'].max() + 1

    train_set, test_set = split_test(df, split, test_size=test_size, seed=seed)
    val_set, test_set = split_test(test_set, split, test_size=0.5, seed=seed)
    train_u_ir = get_ui_dict(train_set, args, key='u', values='ir')
    nbrs = get_uu_dict(uu_elist, user_num, args)

    cate_cnt = [0] * cate_num
    for c in train_set['cate'].values:
        cate_cnt[c] += 1
    cate_cnt = np.array(cate_cnt, dtype=np.float32)
    cate_prior = cate_cnt / np.sum(cate_cnt)

    train_data = [train_set.values, train_u_ir, nbrs]
    train_dataset = DecFMDataset_rate(train_data, item_maxlen, nbrs_maxlen)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_worker,
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True)

    val_data = [val_set.values, train_u_ir, nbrs]
    val_dataset = DecFMDataset_rate(val_data, item_maxlen, nbrs_maxlen)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=2,
                            shuffle=False,
                            persistent_workers=True)

    test_data = [test_set.values, train_u_ir, nbrs]
    test_dataset = DecFMDataset_rate(test_data, item_maxlen, nbrs_maxlen)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=2,
                             shuffle=False,
                             persistent_workers=True)

    return train_loader, val_loader, test_loader, user_num, item_num, cate_num, cate_prior


class DecFMDataset_rate(Dataset):
    def __init__(self, data, item_maxlen, nbrs_maxlen):
        super(DecFMDataset_rate, self).__init__()
        uicr, u_ir, nbrs = data
        self.uicr_list = uicr
        self.u_ir_dict = u_ir
        self.nbrs_dict = nbrs
        self.item_maxlen = item_maxlen
        self.nbrs_maxlen = nbrs_maxlen

    def __len__(self):
        return len(self.uicr_list)

    def __getitem__(self, idx):
        user, item, cate, rate = self.uicr_list[idx]
        u_ir = self.sample_from_arr(self.u_ir_dict[user], self.item_maxlen, item)
        nbr = self.sample_from_nbr(self.nbrs_dict[user], self.nbrs_maxlen)
        return user, u_ir, nbr, item, cate, rate

    def sample_from_arr(self, sample_arr, num_sample, input_item):
        if len(sample_arr) == 0:
            return np.zeros((num_sample, 2), dtype=int)

        indices = np.arange(len(sample_arr))
        exclude_index = np.where(sample_arr == input_item)[0]
        indices = np.delete(indices, exclude_index)

        if len(indices) == 0:
            return np.zeros((num_sample, 2), dtype=int)

        if len(indices) > num_sample:
            idx = np.random.choice(indices, num_sample, replace=False)
            return sample_arr[idx]
        else:
            padding = np.zeros((num_sample - len(indices), 2), dtype=int)
            return np.vstack([sample_arr[indices], padding])

    def sample_from_nbr(self, sample_arr, num_sample):
        sample_arr = np.array(sample_arr, dtype=int)
        if len(sample_arr) == 0: return np.zeros((num_sample,), dtype=int)
        indices = np.arange(len(sample_arr)).astype(int)
        if len(indices) > num_sample:
            idx = np.random.choice(indices, num_sample, replace=False)
            pos_nbr = sample_arr[idx]
        else:
            padding = np.zeros((num_sample - len(indices),), dtype=int)
            pos_nbr = np.hstack([sample_arr[indices].flatten(), padding])

        return pos_nbr
