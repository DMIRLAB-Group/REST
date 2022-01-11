import torch
import json
import pandas as pd
import scipy.sparse as sp
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
import logging
import os
import re
import sys
import pickle as pkl
from collections import defaultdict
from datetime import datetime
from multiprocessing import Queue, Process
from tqdm import tqdm
import gc
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from time import time, mktime, strptime
import torch.multiprocessing
import joblib

torch.multiprocessing.set_sharing_strategy('file_system')
np.set_printoptions(threshold=1000000)


def preprocess_uir(df, prepro='origin', binary=False, pos_threshold=None, level='ui'):
    # set rating >= threshold as positive samples
    if pos_threshold is not None:
        df = df.query(f'rate >= {pos_threshold}').reset_index(drop=True)

    # reset rating to interaction, here just treat all rating as 1
    if binary:
        df['rate'] = 1.0

    # which type of pre-dataset will use
    if prepro == 'origin':
        pass

    elif prepro.endswith('filter'):
        pattern = re.compile(r'\d+')
        filter_num = int(pattern.findall(prepro)[0])

        tmp1 = df.groupby(['user'], as_index=False)['item'].count()
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
        if level == 'ui':
            df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'u':
            df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'i':
            df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()
        else:
            raise ValueError(f'Invalid level value: {level}')

        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()

    elif prepro.endswith('core'):
        pattern = re.compile(r'\d+')
        core_num = int(pattern.findall(prepro)[0])

        def filter_user(df):
            tmp = df.groupby(['user'], as_index=False)['item'].count()
            tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
            df = df.merge(tmp, on=['user'])
            df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)

            return df

        def filter_item(df):
            tmp = df.groupby(['item'], as_index=False)['user'].count()
            tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp, on=['item'])
            df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)

            return df

        if level == 'ui':
            while 1:
                df = filter_user(df)
                df = filter_item(df)
                chk_u = df.groupby('user')['item'].count()
                chk_i = df.groupby('item')['user'].count()
                if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
                    break
        elif level == 'u':
            df = filter_user(df)
        elif level == 'i':
            df = filter_item(df)
        else:
            raise ValueError(f'Invalid level value: {level}')

        gc.collect()

    else:
        raise ValueError('Invalid dataset preprocess type, origin/Ncore/Nfilter (N is int number) expected')

    return df


def load_and_save_yelp():

    # keys = ['user_id', 'name', 'review_count', 'yelping_since', 'useful',
    # 'funny', 'cool', 'elite', 'friends', 'fans', 'average_stars',
    # 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
    # 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool',
    # 'compliment_funny', 'compliment_writer', 'compliment_photos']

    user_path = '../datasets/Yelp/yelp_academic_dataset_user.json'
    u2u_list = list()
    with open(user_path, 'rb') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            uid = line['user_id']
            friends = line['friends'].split(', ')
            for fid in friends:
                u2u_list.append([uid, fid])

    rating_path = '../datasets/Yelp/yelp_academic_dataset_review.json'
    u2i_list = list()
    with open(rating_path, 'rb') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            uid = line['user_id']
            iid = line['business_id']
            rate = line['stars']
            ts  = int(mktime(strptime(line['date'], "%Y-%m-%d %H:%M:%S")))
            u2i_list.append([uid, iid, ts, rate])

    print('u2u =', len(u2u_list))
    print('u2i =', len(u2i_list))

    uid_map = defaultdict(int)  # str id --> int id
    iid_map = defaultdict(int)
    uid_map[0] = 0
    iid_map[0] = 0
    user_num = 1
    item_num = 1
    for i, (uid, iid, ts, rate) in tqdm(enumerate(u2i_list)):
        if uid_map[uid] == 0:
            uid_map[uid] = user_num
            user_num += 1

        if iid_map[iid] == 0:
            iid_map[iid] = item_num
            item_num += 1

        u2i_list[i] = [uid_map[uid], iid_map[iid], ts, rate]

    u2i = np.array(u2i_list, dtype=np.int)
    u2i = u2i[np.argsort(u2i[:, 0])]  # sort by user id

    new_u2u_list = list()
    for u1, u2 in u2u_list:
        new_u1, new_u2 = uid_map[u1], uid_map[u2]
        if new_u1 and new_u2:
            new_u2u_list.append([new_u1, new_u2])

    u2u = np.array(new_u2u_list, dtype=np.int)
    u2u = u2u[np.argsort(u2u[:, 0])]  # sort by u1 id

    print('min uid =', np.min(u2i[:, 0]))
    print('max uid =', np.max(u2i[:, 0]))
    print('num uid =', len(np.unique(u2i[:, 0])))

    print('min iid =', np.min(u2i[:, 1]))
    print('max iid =', np.max(u2i[:, 1]))
    print('num iid =', len(np.unique(u2i[:, 1])))

    print('min ts =', np.min(u2i[:, 2]))
    print('max ts =', np.max(u2i[:, 2]))

    print('min rate =', np.min(u2i[:, 3]))
    print('max rate =', np.max(u2i[:, 3]))
    print('num rate =', len(np.unique(u2i[:, 3])))

    print('min u1 id =', np.min(u2u[:, 0]))
    print('max u1 id =', np.max(u2u[:, 0]))
    print('num u1 id =', len(np.unique(u2u[:, 0])))

    print('min u2 id =', np.min(u2u[:, 1]))
    print('max u2 id =', np.max(u2u[:, 1]))
    print('num u2 id =', len(np.unique(u2u[:, 1])))

    print(u2i[:50])
    print(u2u[:50])

    np.savez(file='../datasets/Yelp/u2ui.npz',
             u2i=u2i,
             u2u=u2u)
    np.savez(file='../datasets/Yelp/uid_map.npz',
             uid_map=uid_map)

    print('saved at', '../datasets/Yelp/u2ui.npz')


def filter_and_reid():
    u2ui = np.load(f'../datasets/Yelp/u2ui.npz')
    u2u, u2i = u2ui['u2u'], u2ui['u2i']
    df = pd.DataFrame(data=u2i, columns=['user', 'item', 'ts', 'rate'])
    df.drop_duplicates(subset=['user', 'item', 'ts', 'rate'], keep='first', inplace=True)

    print('Raw u2i', df.shape)
    print('min user =', df['user'].min())
    print('max user =', df['user'].max())
    print('num user =', len(np.unique(df.values[:, 0])))
    print('min item =', df['item'].min())
    print('max item =', df['item'].max())
    print('num item =', len(np.unique(df.values[:, 1])))

    df = preprocess_uir(df, prepro='3filter', level='u')
    df.drop(['ts'], axis=1, inplace=True)

    print('Processed u2i', df.shape)
    print('min user =', df['user'].min())
    print('max user =', df['user'].max())
    print('num user =', len(np.unique(df.values[:, 0])))
    print('min item =', df['item'].min())
    print('max item =', df['item'].max())
    print('num item =', len(np.unique(df.values[:, 1])))

    df = df.sort_values(['user', 'item'], kind='mergesort').reset_index(drop=True)
    uir = df.values

    user_idmap = defaultdict(int)  # src id -> new id
    user_idmap[0] = 0
    user_num = 1

    for i, (user, item, rate) in tqdm(enumerate(uir)):
        if user_idmap[user] == 0:
            user_idmap[user] = user_num
            user_num += 1

        uir[i, 0] = user_idmap[user]

    print('Raw u2u edges:', len(u2u))
    new_uu_elist = []
    for u1, u2 in tqdm(u2u):
        new_u1 = user_idmap[u1]
        new_u2 = user_idmap[u2]
        if new_u1 and new_u2:
            new_uu_elist.append([new_u1, new_u2])

    print('Processed u2u edges:', len(new_uu_elist))
    u2u = np.array(new_uu_elist).astype(np.int32)
    uir = uir.astype(np.int32)

    save_path = '../datasets/Yelp/reid_u2uir.npz'
    np.savez(file=save_path, u2u=u2u, u2i=uir)

    print('saved at', save_path)


def delete_isolated_user():

    u2ui = np.load('../datasets/Yelp/reid_u2uir.npz')
    uu_elist = u2ui['u2u']
    uir = u2ui['u2i']

    print('Building u2u graph...')
    user_num = np.max(uir[:, 0]) + 1
    g = nx.Graph()
    g.add_nodes_from(list(range(user_num)))
    g.add_edges_from(uu_elist)
    g.remove_node(0)

    isolated_user_set = set(nx.isolates(g))
    print('Isolated user =', len(isolated_user_set))

    new_uir = []
    for user, item, rate in tqdm(uir):
        if user not in isolated_user_set:
            new_uir.append([user, item, rate])

    new_uir = np.array(new_uir, dtype=np.int32)

    print('No isolated user u2i =', new_uir.shape)

    user_idmap = defaultdict(int)  # src id -> new id
    user_idmap[0] = 0
    user_num = 1
    for i, (user, item, rate) in tqdm(enumerate(new_uir)):
        if user_idmap[user] == 0:
            user_idmap[user] = user_num
            user_num += 1

        new_uir[i, 0] = user_idmap[user]

    new_uu_elist = []
    for u1, u2 in tqdm(uu_elist):
        new_u1 = user_idmap[u1]
        new_u2 = user_idmap[u2]
        if new_u1 and new_u2:
            new_uu_elist.append([new_u1, new_u2])

    new_uu_elist = np.array(new_uu_elist, dtype=np.int32)

    df = pd.DataFrame(data=new_uir, columns=['user', 'item', 'rate'])
    df['item'] = pd.Categorical(df['item']).codes + 1

    print(df.head(20))

    user_num = df['user'].max() + 1
    item_num = df['item'].max() + 1

    print('min user =', df['user'].min())
    print('max user =', df['user'].max())
    num_user = len(np.unique(df.values[:, 0]))
    print('num user =', num_user)

    print('min item =', df['item'].min())
    print('max item =', df['item'].max())
    num_item = len(np.unique(df.values[:, 1]))
    print('num item =', num_item)

    print(f'Loaded Yelp dataset with {user_num} users, {item_num} items, '
          f'{len(df.values)} u2i, {len(new_uu_elist)} u2u. ')

    new_uir = df.values.astype(np.int32)
    save_path = '../datasets/Yelp/noiso_reid_u2uir.npz'
    np.savez(file=save_path, u2u=new_uu_elist, u2i=new_uir)


def preprocess_yelp_rate():
    load_and_save_yelp()
    filter_and_reid()  # 3filter删除item和user -> 重新赋值uid -> 从u2u删除没有item的user
    delete_isolated_user()  # 删除孤立用户 -> 重新赋值uid -> 更新u2u和u2b的uid
    print('Pre-process yelp done!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Yelp')
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--seq_maxlen', type=int, default=20)
    parser.add_argument('--nbr_maxlen', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=27)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    seq_maxlen = args.seq_maxlen
    nbr_maxlen = args.nbr_maxlen

    preprocess_yelp_rate()






