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


def save_pkl(file, obj, compress=0):
    # compress=('gzip', 3)
    joblib.dump(value=obj, filename=file, compress=compress)


def load_pkl(file):
    return joblib.load(file)


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


def split_test(df, test_method='fo', test_size=.2, seed=27):
    """
    method of splitting data into training data and test data
    Parameters
    ----------
    df : pd.DataFrame raw data waiting for test set splitting
    test_method : str, way to split test set
                    'fo': split by ratio
                    'tfo': split by ratio with timestamp
                    'tloo': leave one out with timestamp
                    'loo': leave one out
                    'ufo': split by ratio in user level
                    'utfo': time-aware split by ratio in user level
    test_size : float, size of test set

    Returns
    -------
    train_set : pd.DataFrame training dataset
    test_set : pd.DataFrame test dataset

    """

    train_set, test_set = pd.DataFrame(), pd.DataFrame()
    if test_method == 'ufo':
        driver_ids = df['user']
        _, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        for train_idx, test_idx in gss.split(df, groups=driver_indices):
            train_set, test_set = df.loc[train_idx, :].copy(), df.loc[test_idx, :].copy()

    elif test_method == 'utfo':
        df = df.sort_values(['user', 'timestamp']).reset_index(drop=True)

        def time_split(grp):
            start_idx = grp.index[0]
            split_len = int(np.ceil(len(grp) * (1 - test_size)))
            split_idx = start_idx + split_len
            end_idx = grp.index[-1]

            return list(range(split_idx, end_idx + 1))

        test_index = df.groupby('user').apply(time_split).explode().values
        test_set = df.loc[test_index, :]
        train_set = df[~df.index.isin(test_index)]

    elif test_method == 'tfo':
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        split_idx = int(np.ceil(len(df) * (1 - test_size)))
        train_set, test_set = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy()

    elif test_method == 'fo':
        train_set, test_set = train_test_split(df, test_size=test_size, random_state=seed)

    elif test_method == 'tloo':
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        df['rank_latest'] = df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        train_set, test_set = df[df['rank_latest'] > 1].copy(), df[df['rank_latest'] == 1].copy()
        del train_set['rank_latest'], test_set['rank_latest']

    elif test_method == 'loo':
        test_index = df.groupby(['user']).apply(lambda grp: np.random.choice(grp.index))
        test_set = df.loc[test_index, :].copy()
        train_set = df[~df.index.isin(test_index)].copy()
    else:
        raise ValueError('Invalid data_split value, expect: loo, fo, tloo, tfo')

    train_set, test_set = train_set.reset_index(drop=True), test_set.reset_index(drop=True)
    train_set = train_set.sort_values(['user', 'item']).reset_index(drop=True)
    test_set = test_set.sort_values(['user', 'item']).reset_index(drop=True)

    return train_set, test_set


def get_uu_dict(uu_elist, user_num, args):
    save_path = f'../datasets/{args.dataset}/nbrs_dict.pkl'
    if os.path.exists(save_path):
        return load_pkl(save_path)
    else:
        print(f'Building neighbors dictionary for {args.dataset}...')
        g = nx.Graph()
        g.add_nodes_from(list(range(user_num)))
        g.add_edges_from(uu_elist)
        g.add_edges_from([[u, u] for u in g.nodes])  # add self-loop to avoid NaN attention scores
        nbrs = nx.to_dict_of_lists(g)
        save_pkl(save_path, nbrs)
        return nbrs


def get_ui_dict(train_df, args, key='u', values='i'):
    save_path = f'../datasets/{args.dataset}/{key}_{values}_seed{args.seed}.pkl'
    if os.path.exists(save_path):
        return load_pkl(save_path)
    else:
        str_mapping = {'u': 'user', 'i': 'item', 'r': 'rate', 'c': 'cate', 't': 'ts'}
        print(f'Building {key}-{values} dictionary for {args.dataset}...')
        key, val_list = str_mapping[key], [str_mapping[v] for v in values]
        ui_dict = defaultdict(list)

        def apply_fn(grp):
            key_id = grp[key].values[0]
            ui_dict[key_id] = grp[val_list].values

        train_df.groupby(key).apply(apply_fn)
        save_pkl(save_path, ui_dict)
        return ui_dict


def get_unexp_dict_rate(u_ir_dict, nbrs_dict, user_num, args):
    save_path = f'../datasets/{args.dataset}/unexpo_dict_seed{args.seed}_thresh{args.unexp_pos_threshold}.pkl'
    if os.path.exists(save_path):
        return load_pkl(save_path)
    else:
        print('Generating unexposure item dict...')
        unexp_dict = defaultdict(list)
        for user in tqdm(range(1, user_num)):

            user_rated_items = u_ir_dict[user]
            if len(user_rated_items) == 0: continue
            user_rated_items = set(user_rated_items[:, 0].tolist())

            candidates = list()
            for nbr in nbrs_dict[user]:
                nbr_rated_items = u_ir_dict[nbr]
                for item, rate in nbr_rated_items:
                    if int(item) not in user_rated_items:
                        candidates.append([item, rate])

            if len(candidates) == 0: continue

            sample_arr = np.array(candidates, dtype=np.int64)
            sample_item_ids = sample_arr[:, 0]
            sample_item_rates = sample_arr[:, 1]

            bcnt = np.bincount(sample_item_ids)
            if np.max(bcnt) < args.unexp_pos_threshold:
                continue
            else:
                common_cands = np.where(bcnt == np.max(bcnt))[0]  # common item ids (multiple)
                unexp_list = list()
                # print('user =', user)
                # print('sampl_arr =', sample_arr.shape, sample_arr[:])
                # print('common cands =', common_cands)

                for common_item in common_cands:
                    item_idx = np.where(sample_item_ids == common_item)[0]
                    item_rates = sample_item_rates[item_idx]
                    most_rate = int(np.argmax(np.bincount(item_rates)))
                    unexp_list.append([common_item, most_rate])
                    # print('common item =', common_item)
                    # print('common item rates =', item_rates)
                    # print('most rate =', most_rate)
                    # print('----------------------------')

                unexp_dict[user] = unexp_list

        save_pkl(save_path, unexp_dict)
        return unexp_dict


def get_unexp_dict_rank(u_i_dict, nbrs_dict, user_num, args):
    save_path = f'../datasets/{args.dataset}/unexpo_dict_seed{args.seed}_thresh{args.unexp_pos_threshold}.pkl'
    if os.path.exists(save_path):
        return load_pkl(save_path)
    else:
        print('Generating unexposure item dict...')
        unexp_dict = defaultdict(list)
        for user in tqdm(range(1, user_num)):

            user_rated_items = u_i_dict[user]
            if len(user_rated_items) == 0: continue
            user_rated_items = set(user_rated_items.flatten().tolist())

            candidates = list()
            for nbr in nbrs_dict[user]:
                nbr_rated_items = u_i_dict[nbr]
                if len(nbr_rated_items):
                    for item in nbr_rated_items.flatten():
                        if int(item) not in user_rated_items:
                            candidates.append(item)

            if len(candidates) == 0: continue

            sample_item_ids = np.array(candidates, dtype=np.int64)
            bcnt = np.bincount(sample_item_ids)
            if np.max(bcnt) < args.unexp_pos_threshold:
                continue
            else:
                common_cands = np.where(bcnt == np.max(bcnt))[0]  # common item ids (multiple)
                unexp_dict[user] = common_cands.tolist()

        save_pkl(save_path, unexp_dict)
        return unexp_dict


def get_ui_graph(dataset, user_train, user_num, item_num):
    saved_path = f'datasets/{dataset}/norm_adj.npz'
    if os.path.exists(saved_path):
        norm_adj = sp.load_npz(saved_path)
        print('Loaded normalized joint rating matrix.')
    else:
        print('Generating sparse rating matrix...')
        train_users = []
        train_items = []

        for user in range(1, user_num):
            items = user_train[user]
            if len(items):
                train_users.extend(len(items) * [user])
                train_items.extend(items[:, 0].tolist())

        print('Step1: list to csr_matrix...')

        rating_maxtrix = sp.csr_matrix(
            (np.ones(len(train_users), dtype=np.int8), (train_users, train_items)),
            shape=(user_num, item_num)
        ).tolil()

        print('Step2: csr to dok...')

        adj_mat = sp.dok_matrix(
            (user_num + item_num, user_num + item_num),
            dtype=np.int8).tolil()

        print('adj_mat =', adj_mat.dtype, adj_mat.shape)
        print('rating_maxtrix =', rating_maxtrix.dtype, rating_maxtrix.shape)

        print('Step3: slicing...')

        adj_mat[:user_num, user_num:] = rating_maxtrix
        adj_mat[user_num:, :user_num] = rating_maxtrix.T
        adj_mat = adj_mat.todok().astype(np.float16)

        print('Step4: Normalizing...')

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocoo()
        sp.save_npz(saved_path, norm_adj)
        print('norm_adj saved at', saved_path)

    print('Npz to SparseTensor...')
    row = torch.Tensor(norm_adj.row).long()
    col = torch.Tensor(norm_adj.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(norm_adj.data)
    ui_graph = torch.sparse.FloatTensor(index, data, torch.Size(norm_adj.shape)).coalesce()
    return ui_graph


def load_ds_causalrec_rank(args):
    dataset = 'Wechat'
    batch_size = args.batch_size
    test_size = 0.2
    split = 'fo'
    seed = args.seed
    item_maxlen = args.seq_maxlen
    nbrs_maxlen = args.nbr_maxlen
    num_workers = args.num_workers

    save_path = f'datasets/{dataset}/processed_{dataset}_causal.npz'

    if os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True)
        train_ui = data['train_set']
        val_ui = data['val_set']
        test_ui = data['test_set']
        train_u_i = data['train_u_i'][()]
        nbrs = data['nbrs'][()]
        user_num = data['user_num']
        item_num = data['item_num']
    else:
        data = np.load(f'datasets/{dataset}/noiso_reid_wechat_u2ub.npz', allow_pickle=True)
        df = pd.DataFrame(data=data['u2b'], columns=['user', 'item', 'ts'])
        df.drop(columns=['ts'], inplace=True)
        df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
        df = df.sort_values(['user', 'item'], kind='mergesort').reset_index(drop=True)

        user_num = df['user'].max() + 1
        item_num = df['item'].max() + 1

        train_set, test_set = split_test(df, split, test_size=test_size, seed=seed)
        val_set, test_set = split_test(test_set, split, test_size=0.5, seed=seed)
        train_u_i = get_ui_dict(train_set, args, key='u', values='i')
        nbrs = get_uu_dict(data['u2u'], user_num, args)

        train_ui = train_set.values
        val_ui = val_set.values
        test_ui = test_set.values

        np.savez(
            save_path,
            train_set=train_ui,
            val_set=val_ui,
            test_set=test_ui,
            train_u_i=train_u_i,
            nbrs=nbrs,
            user_num=user_num,
            item_num=item_num)

    unexp_dict = get_unexp_dict_rank(train_u_i, nbrs, user_num, args)

    train_data = [train_ui, train_u_i, nbrs, unexp_dict]
    train_dataset = CausalRecDataset_rank(train_data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              persistent_workers=True)

    val_data = [val_ui, train_u_i, nbrs, unexp_dict]
    val_dataset = CausalRecDataset_rank(val_data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                            persistent_workers=True)

    test_data = [test_ui, train_u_i, nbrs, unexp_dict]
    test_dataset = CausalRecDataset_rank(test_data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader, user_num, item_num, train_u_i


class CausalRecDataset_rank(Dataset):
    def __init__(self, data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train):
        super(CausalRecDataset_rank, self).__init__()
        ui, u_i, nbrs, unexp_dict = data
        self.ui_list = ui
        self.u_i_dict = u_i
        self.nbrs_dict = nbrs
        self.unexp_dict = unexp_dict

        self.item_maxlen = item_maxlen
        self.nbrs_maxlen = nbrs_maxlen
        self.user_num = user_num
        self.item_num = item_num

        self.is_train = is_train

    def __len__(self):
        return len(self.ui_list)

    def __getitem__(self, idx):
        user, item = self.ui_list[idx]

        u_i = self.sample_from_arr(self.u_i_dict[user], self.item_maxlen, item)
        pos_nbr, neg_nbr = self.sample_from_nbr(self.nbrs_dict[user], self.nbrs_maxlen)
        if self.is_train:
            unexp_item = self.sample_unexp(user)
            neg_item, unexp_neg_item = self.sample_neg_items(user)
            return user, u_i, pos_nbr, neg_nbr, item, unexp_item, neg_item, unexp_neg_item
        else:
            eval_items = self.sample_eval_items(user, item)
            return user, u_i, pos_nbr, eval_items

    def sample_from_arr(self, sample_arr, num_sample, input_item):
        if len(sample_arr) == 0:
            return np.zeros((num_sample,), dtype=int)

        indices = np.arange(len(sample_arr))
        exclude_index = np.where(sample_arr == input_item)[0]
        indices = np.delete(indices, exclude_index)

        if len(indices) == 0:
            return np.zeros((num_sample,), dtype=int)

        if len(indices) > num_sample:
            idx = np.random.choice(indices, num_sample, replace=False)
            return sample_arr[idx].flatten()
        else:
            padding = np.zeros((num_sample - len(indices),), dtype=int)
            return np.hstack([sample_arr[indices].flatten(), padding])

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

        neg_nbr = np.random.randint(low=1, high=self.user_num, size=len(pos_nbr))
        return pos_nbr, neg_nbr

    def sample_unexp(self, user):
        unexp_cand = self.unexp_dict[user]
        if len(unexp_cand) == 0:
            return 0
        else:
            indices = np.arange(len(unexp_cand))
            idx = np.random.choice(indices)
            return unexp_cand[idx]

    def sample_neg_items(self, user):
        rated_items = self.u_i_dict[user]
        if len(rated_items): rated_items = rated_items.flatten()
        neg_item = np.random.randint(low=1, high=self.item_num)
        while neg_item in rated_items:
            neg_item = np.random.randint(low=1, high=self.item_num)
        unexp_neg_item = np.random.randint(low=1, high=self.item_num)
        while neg_item in rated_items:
            unexp_neg_item = np.random.randint(low=1, high=self.item_num)
        return neg_item, unexp_neg_item

    def sample_eval_items(self, user, eval_item):
        sample_size = 100
        # sample_size = 2
        eval_items = [eval_item]
        while len(eval_items) < sample_size:
            neg = np.random.randint(low=1, high=self.item_num)
            while neg in self.u_i_dict[user]:
                neg = np.random.randint(low=1, high=self.item_num)
            eval_items.append(neg)

        eval_items = np.array(eval_items, dtype=int)
        return eval_items


def load_ds_causalrec_rate(args):
    dataset = args.dataset
    batch_size = args.batch_size
    test_size = 0.2
    split = 'fo'
    seed = args.seed
    item_maxlen = args.seq_maxlen
    nbrs_maxlen = args.nbr_maxlen
    num_workers = args.num_workers

    if dataset == 'Ciao':
        rating_mat = loadmat(f'../datasets/Ciao/rating_with_timestamp.mat')
        rating = rating_mat['rating']
        uu_elist = loadmat(f'../datasets/{args.dataset}/trust.mat')['trust']

        df = pd.DataFrame(data=rating, columns=['user', 'item', 'cate', 'rate', 'help', 'ts'])
        df.drop(columns=['cate', 'help'], inplace=True)
        df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
        df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
        df.drop(columns=['ts'], inplace=True)

        user_num = df['user'].max() + 1
        item_num = df['item'].max() + 1

        train_set, test_set = split_test(df, split, test_size=test_size, seed=seed)
        val_set, test_set = split_test(test_set, split, test_size=0.5, seed=seed)
        train_u_ir = get_ui_dict(train_set, args, key='u', values='ir')
        nbrs = get_uu_dict(uu_elist, user_num, args)
        unexp_dict = get_unexp_dict_rate(train_u_ir, nbrs, user_num, args)

        train_uir = train_set.values
        val_uir = val_set.values
        test_uir = test_set.values

    elif dataset == 'Epinions':
        rating_mat = loadmat(f'../datasets/Epinions/rating_with_timestamp.mat')
        rating = rating_mat['rating_with_timestamp']
        uu_elist = loadmat(f'../datasets/{args.dataset}/trust.mat')['trust']

        df = pd.DataFrame(data=rating, columns=['user', 'item', 'cate', 'rate', 'help', 'ts'])
        df.drop(columns=['cate', 'help'], inplace=True)
        df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
        df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
        df.drop(columns=['ts'], inplace=True)

        user_num = df['user'].max() + 1
        item_num = df['item'].max() + 1

        train_set, test_set = split_test(df, split, test_size=test_size, seed=seed)
        val_set, test_set = split_test(test_set, split, test_size=0.5, seed=seed)
        train_u_ir = get_ui_dict(train_set, args, key='u', values='ir')
        nbrs = get_uu_dict(uu_elist, user_num, args)
        unexp_dict = get_unexp_dict_rate(train_u_ir, nbrs, user_num, args)

        train_uir = train_set.values
        val_uir = val_set.values
        test_uir = test_set.values

    elif dataset == 'Yelp':

        # Loaded Yelp dataset with 332133 users, 197231 items, 4567966 u2i, 7043148 u2u.

        save_path = f'../datasets/Yelp/processed_yelp_causalrec_thresh{args.unexp_pos_threshold}.npz'
        if os.path.exists(save_path):
            data = np.load(save_path, allow_pickle=True)
            train_uir = data['train_set']
            val_uir = data['val_set']
            test_uir = data['test_set']
            train_u_ir = data['train_u_ir'][()]
            nbrs = data['nbrs'][()]
            unexp_dict = data['unexp_dict'][()]
            user_num = data['user_num']
            item_num = data['item_num']
        else:
            print('Preprocessing Yelp dataset...')

            data = np.load('../datasets/Yelp/noiso_reid_u2uir.npz')
            rating = data['u2i']
            uu_elist = data['u2u']

            df = pd.DataFrame(data=rating, columns=['user', 'item', 'rate'])
            user_num = df['user'].max() + 1
            item_num = df['item'].max() + 1
            train_set, test_set = split_test(df, split, test_size=test_size, seed=seed)
            val_set, test_set = split_test(test_set, split, test_size=0.5, seed=seed)
            train_u_ir = get_ui_dict(train_set, args, key='u', values='ir')
            nbrs = get_uu_dict(uu_elist, user_num, args)
            unexp_dict = get_unexp_dict_rate(train_u_ir, nbrs, user_num, args)

            train_uir = train_set.values
            val_uir = val_set.values
            test_uir = test_set.values

            np.savez(
                save_path,
                train_set=train_uir,
                val_set=val_uir,
                test_set=test_uir,
                train_u_ir=train_u_ir,
                nbrs=nbrs,
                unexp_dict=unexp_dict,
                user_num=user_num,
                item_num=item_num
            )

    train_data = [train_uir, train_u_ir, nbrs, unexp_dict]
    train_dataset = CausalRecDataset_rate(train_data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              pin_memory=True)

    val_data = [val_uir, train_u_ir, nbrs, unexp_dict]
    val_dataset = CausalRecDataset_rate(val_data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    test_data = [test_uir, train_u_ir, nbrs, unexp_dict]
    test_dataset = CausalRecDataset_rate(test_data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader, user_num, item_num, train_u_ir


class CausalRecDataset_rate(Dataset):
    def __init__(self, data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train):
        super(CausalRecDataset_rate, self).__init__()
        uir, u_ir, nbrs, unexp_dict = data
        self.uir_list = uir
        self.u_ir_dict = u_ir
        self.nbrs_dict = nbrs
        self.unexp_dict = unexp_dict

        self.item_maxlen = item_maxlen
        self.nbrs_maxlen = nbrs_maxlen
        self.user_num = user_num
        self.item_num = item_num

        self.is_train = is_train

    def __len__(self):
        return len(self.uir_list)

    def __getitem__(self, idx):
        user, item, rate = self.uir_list[idx]
        u_ir = self.sample_from_arr(self.u_ir_dict[user], self.item_maxlen, item)
        pos_nbr, neg_nbr = self.sample_from_nbr(self.nbrs_dict[user], self.nbrs_maxlen)
        unexp_item, unexp_rate = self.sample_unexp(user)
        return user, u_ir, pos_nbr, neg_nbr, item, rate, unexp_item, unexp_rate

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

        neg_nbr = np.random.randint(low=1, high=self.user_num, size=len(pos_nbr))
        return pos_nbr, neg_nbr

    def sample_unexp(self, user):
        unexp_cand = self.unexp_dict[user]
        if len(unexp_cand) == 0:
            return np.array([0, 0], dtype=int)
        else:
            indices = np.arange(len(unexp_cand))
            idx = np.random.choice(indices)
            return unexp_cand[idx]


def load_ds_baseline_rank(
        args,
        dataset='Wechat',
        batch_size=1024,
        test_size=0.2,
        split='fo',
        seed=27,
        item_maxlen=20,
        nbrs_maxlen=20,
        num_worker=0,
        is_mf=False):
    save_path = f'datasets/{dataset}/processed_{dataset}_causal.npz'

    if os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True)
        train_ui = data['train_set']
        val_ui = data['val_set']
        test_ui = data['test_set']
        train_u_i = data['train_u_i'][()]
        nbrs = data['nbrs'][()]
        user_num = data['user_num']
        item_num = data['item_num']
    else:
        data = np.load(f'datasets/{dataset}/noiso_reid_u2ui.npz', allow_pickle=True)
        df = pd.DataFrame(data=data['u2i'], columns=['user', 'item', 'ts'])
        df.drop(columns=['ts'], inplace=True)
        df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
        df = df.sort_values(['user', 'item'], kind='mergesort').reset_index(drop=True)

        user_num = df['user'].max() + 1
        item_num = df['item'].max() + 1

        train_set, test_set = split_test(df, split, test_size=test_size, seed=seed)
        val_set, test_set = split_test(test_set, split, test_size=0.5, seed=seed)
        train_u_i = get_ui_dict(train_set, args, key='u', values='i')
        nbrs = get_uu_dict(data['u2u'], user_num, args)

        train_ui = train_set.values
        val_ui = val_set.values
        test_ui = test_set.values

        np.savez(
            save_path,
            train_set=train_ui,
            val_set=val_ui,
            test_set=test_ui,
            train_u_i=train_u_i,
            nbrs=nbrs,
            user_num=user_num,
            item_num=item_num
        )

    train_data = [train_ui, train_u_i, nbrs]
    train_dataset = BaselineDataset_rank(train_data, item_maxlen, nbrs_maxlen, user_num, item_num,
                                         is_train=True, is_mf=is_mf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_worker, shuffle=True,
                              persistent_workers=True)

    val_data = [val_ui, train_u_i, nbrs]
    val_dataset = BaselineDataset_rank(val_data, item_maxlen, nbrs_maxlen, user_num, item_num,
                                       is_train=False, is_mf=is_mf)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_worker, shuffle=False,
                            persistent_workers=True)

    test_data = [test_ui, train_u_i, nbrs]
    test_dataset = BaselineDataset_rank(test_data, item_maxlen, nbrs_maxlen, user_num, item_num,
                                        is_train=False, is_mf=is_mf)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_worker, shuffle=False)

    return train_loader, val_loader, test_loader, user_num, item_num, train_u_i


class BaselineDataset_rank(Dataset):
    def __init__(self, data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train, is_mf=False):
        super(BaselineDataset_rank, self).__init__()
        ui, u_i, nbrs = data
        self.ui_list = ui
        self.u_i_dict = u_i
        self.nbrs_dict = nbrs

        self.item_maxlen = item_maxlen
        self.nbrs_maxlen = nbrs_maxlen
        self.user_num = user_num
        self.item_num = item_num

        self.is_train = is_train
        self.is_mf = is_mf

    def __len__(self):
        return len(self.ui_list)

    def __getitem__(self, idx):
        user, pos_item = self.ui_list[idx]
        u_i = self.sample_from_arr(self.u_i_dict[user], self.item_maxlen, pos_item)
        nbr = self.sample_from_nbr(self.nbrs_dict[user], self.nbrs_maxlen)
        if self.is_train:
            neg_item = self.sample_neg(user)
            return user, u_i, nbr, pos_item, neg_item
        else:
            eval_items = self.sample_eval_items(user, pos_item)
            return user, u_i, nbr, eval_items

    def sample_from_arr(self, sample_arr, num_sample, input_item):
        if len(sample_arr) == 0 or self.is_mf:
            return np.zeros((num_sample,), dtype=int)

        indices = np.arange(len(sample_arr))
        exclude_index = np.where(sample_arr == input_item)[0]
        indices = np.delete(indices, exclude_index)

        if len(indices) == 0:
            return np.zeros((num_sample,), dtype=int)

        if len(indices) > num_sample:
            idx = np.random.choice(indices, num_sample, replace=False)
            return sample_arr[idx].flatten()
        else:
            padding = np.zeros((num_sample - len(indices),), dtype=int)
            return np.hstack([sample_arr[indices].flatten(), padding])

    def sample_from_nbr(self, sample_arr, num_sample):
        sample_arr = np.array(sample_arr, dtype=int)
        if len(sample_arr) == 0 or self.is_mf: return np.zeros((num_sample,), dtype=int)
        indices = np.arange(len(sample_arr)).astype(int)
        if len(indices) > num_sample:
            idx = np.random.choice(indices, num_sample, replace=False)
            pos_nbr = sample_arr[idx]
        else:
            padding = np.zeros((num_sample - len(indices),), dtype=int)
            pos_nbr = np.hstack([sample_arr[indices].flatten(), padding])

        return pos_nbr

    def sample_neg(self, user):
        neg_item = np.random.randint(low=1, high=self.item_num)
        while neg_item in self.u_i_dict[user]:
            neg_item = np.random.randint(low=1, high=self.item_num)

        return neg_item

    def sample_eval_items(self, user, eval_item):
        sample_size = 100
        eval_items = [eval_item]
        while len(eval_items) < sample_size:
            neg = np.random.randint(low=1, high=self.item_num)
            while neg in self.u_i_dict[user]:
                neg = np.random.randint(low=1, high=self.item_num)
            eval_items.append(neg)

        eval_items = np.array(eval_items, dtype=int)
        return eval_items


def load_ds_baseline_rate(
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

        df = pd.DataFrame(data=rating, columns=['user', 'item', 'cate', 'rate', 'help', 'ts'])
        df.drop(columns=['cate', 'help'], inplace=True)
        df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
        df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
        df.drop(columns=['ts'], inplace=True)

        # print('Ciao')
        # print('max ts =', df['ts'].max())
        # print('min ts =', df['ts'].min())

        # max ts = 1302591600, 2011-04-12 15:00:00
        # min ts = 959670000, 2000-05-30 15:00:00
        user_num = df['user'].max() + 1
        item_num = df['item'].max() + 1

        train_set, test_set = split_test(df, split, test_size=test_size, seed=seed)
        val_set, test_set = split_test(test_set, split, test_size=0.5, seed=seed)
        train_u_ir = get_ui_dict(train_set, args, key='u', values='ir')
        nbrs = get_uu_dict(uu_elist, user_num, args)

        train_uir = train_set.values
        val_uir = val_set.values
        test_uir = test_set.values

    elif dataset == 'Epinions':
        rating_mat = loadmat(f'../datasets/Epinions/rating_with_timestamp.mat')
        rating = rating_mat['rating_with_timestamp']
        uu_elist = loadmat(f'../datasets/{args.dataset}/trust.mat')['trust']

        df = pd.DataFrame(data=rating, columns=['user', 'item', 'cate', 'rate', 'help', 'ts'])
        df.drop(columns=['cate', 'help'], inplace=True)
        df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
        df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
        df.drop(columns=['ts'], inplace=True)

        # print('Epinions')
        # print('max ts =', df['ts'].max())
        # print('min ts =', df['ts'].min())

        # max ts = 1304924400, 2011-05-09 15:00:00
        # min ts = 931158000, 1999-07-05 15:00:00

        user_num = df['user'].max() + 1
        item_num = df['item'].max() + 1

        train_set, test_set = split_test(df, split, test_size=test_size, seed=seed)
        val_set, test_set = split_test(test_set, split, test_size=0.5, seed=seed)
        train_u_ir = get_ui_dict(train_set, args, key='u', values='ir')
        nbrs = get_uu_dict(uu_elist, user_num, args)

        train_uir = train_set.values
        val_uir = val_set.values
        test_uir = test_set.values

    elif dataset == 'Yelp':
        save_path = '../datasets/Yelp/processed_yelp_causalrec_thresh5.npz'
        if os.path.exists(save_path):
            data = np.load(save_path, allow_pickle=True)
            train_uir = data['train_set']
            val_uir = data['val_set']
            test_uir = data['test_set']
            train_u_ir = data['train_u_ir'][()]
            nbrs = data['nbrs'][()]
            user_num = data['user_num']
            item_num = data['item_num']
        else:
            data = np.load('../datasets/Yelp/noiso_reid_u2uir.npz')
            rating = data['u2i']
            uu_elist = data['u2u']

            df = pd.DataFrame(data=rating, columns=['user', 'item', 'rate'])
            user_num = df['user'].max() + 1
            item_num = df['item'].max() + 1
            train_set, test_set = split_test(df, split, test_size=test_size, seed=seed)
            val_set, test_set = split_test(test_set, split, test_size=0.5, seed=seed)
            train_u_ir = get_ui_dict(train_set, args, key='u', values='ir')
            nbrs = get_uu_dict(uu_elist, user_num, args)

            train_uir = train_set.values
            val_uir = val_set.values
            test_uir = test_set.values

            np.savez(
                save_path,
                train_set=train_uir,
                val_set=val_uir,
                test_set=test_uir,
                train_u_ir=train_u_ir,
                nbrs=nbrs,
                user_num=user_num,
                item_num=item_num
            )

    train_data = [train_uir, train_u_ir, nbrs]
    train_dataset = BaselineDataset_rate(train_data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_worker,
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True)

    val_data = [val_uir, train_u_ir, nbrs]
    val_dataset = BaselineDataset_rate(val_data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            shuffle=False,
                            persistent_workers=True)

    test_data = [test_uir, train_u_ir, nbrs]
    test_dataset = BaselineDataset_rate(test_data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=4,
                             shuffle=False,
                             persistent_workers=True)

    return train_loader, val_loader, test_loader, user_num, item_num, train_u_ir


class BaselineDataset_rate(Dataset):
    def __init__(self, data, item_maxlen, nbrs_maxlen, user_num, item_num, is_train, is_mf=False):
        super(BaselineDataset_rate, self).__init__()
        uir, u_ir, nbrs = data
        self.uir_list = uir
        self.u_ir_dict = u_ir
        self.nbrs_dict = nbrs

        self.item_maxlen = item_maxlen
        self.nbrs_maxlen = nbrs_maxlen
        self.user_num = user_num
        self.item_num = item_num

        self.is_train = is_train
        self.is_mf = is_mf

    def __len__(self):
        return len(self.uir_list)

    def __getitem__(self, idx):
        user, item, rate = self.uir_list[idx]
        u_ir = self.sample_from_arr(self.u_ir_dict[user], self.item_maxlen, item)
        nbr = self.sample_from_nbr(self.nbrs_dict[user], self.nbrs_maxlen)
        return user, u_ir, nbr, item, rate

    def sample_from_arr(self, sample_arr, num_sample, input_item):
        if len(sample_arr) == 0 or self.is_mf:
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
        if len(sample_arr) == 0 or self.is_mf: return np.zeros((num_sample,), dtype=int)
        indices = np.arange(len(sample_arr)).astype(int)
        if len(indices) > num_sample:
            idx = np.random.choice(indices, num_sample, replace=False)
            pos_nbr = sample_arr[idx]
        else:
            padding = np.zeros((num_sample - len(indices),), dtype=int)
            pos_nbr = np.hstack([sample_arr[indices].flatten(), padding])

        return pos_nbr
