import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from collections import Counter
from einops import repeat, rearrange
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.metrics.pairwise import euclidean_distances
import utils


TRAJ_ID_COL = 'trip'
X_COL = 'lng'
Y_COL = 'lat'
T_COL = 'timestamp'
DT_COL = 'delta_t'
ROAD_COL = 'road'
COL_I = {
    "spatial": [0, 1],
    "temporal": [2, 3],
    "road": 4
}
FEATURE_PAD = 0
MIN_TRIP_LEN = 5
MAX_TRIP_LEN = 120
SEARCH_META_DIR = "/data/LiuYichen/TrajClip_datasets/processed_data/search_meta"


class TrajClipDataset(Dataset):
    """Dataset support class for TrajCLIP.

    Args:
        traj_df (pd.DataFrame): contains points of all trajectories.
        traj_ids (pd.Series): records the unique IDs of all trajectory sequences.
        spatial_border (list): coordinates indicating the spatial border: [[x_min, y_min], [x_max, y_max]].
    """

    def __init__(self, traj_df):
        """
        Args:
            traj_df (pd.DataFrame): contains points of all trajectories.
        """
        super().__init__()
        # Filtering trips to keep the trajectories with length at [MIN_TRIP_LEN, MAX_TRIP_LEN]
        traj_ids = []
        for _, group in tqdm(traj_df.groupby(TRAJ_ID_COL), desc='Filtering trips', total=len(traj_df[TRAJ_ID_COL].unique()), leave=False, ncols=70):
            if (not group.isna().any().any()) and group.shape[0] >= MIN_TRIP_LEN and group.shape[0] <= MAX_TRIP_LEN:
                traj_ids.append(group.iloc[0]['trip'])
        self.traj_ids = np.array(traj_ids)
        self.traj_df = traj_df[traj_df['trip'].isin(self.traj_ids)].copy()

        self.traj_df['timestamp'] = self.traj_df['time'].apply(lambda x: x.timestamp())

        spatial_border = traj_df[[X_COL, Y_COL]]
        self.spatial_border = [spatial_border.min().tolist(), spatial_border.max().tolist()]

    def __len__(self):
        return self.traj_ids.shape[0]

    def __getitem__(self, index):
        one_traj = self.traj_df[self.traj_df[TRAJ_ID_COL] == self.traj_ids[index]].copy()
        one_traj[DT_COL] = one_traj[T_COL] - one_traj[T_COL].iloc[0]
        return one_traj


class PretrainPadder:
    """Collate function for padding pre-training data.
    """

    def __init__(self, device):
        """
        Args:
            device (str): name of the device to put tensors on.
        """
        self.device = device

    def __call__(self, raw_batch):
        """Collate function for padding the raw batch of trajectory DataFrames into Tensors.

        Args:
            raw_batch (list): each item is a `pd.DataFrame` representing one trajectory.

        Returns:
            torch.FloatTensor: the padded batch of trajectory features, with shape (B, L, F).
            torch.LongTensor: the valid lengths of trajectories in the batch, with shape (B).
        """
        traj_batch, valid_lens = [], []
        for row in raw_batch:
            traj = row[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL]].to_numpy()
            valid_len = traj.shape[0]
            traj_batch.append(traj)
            valid_lens.append(valid_len)
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)

        return traj_batch, valid_lens


class DpPadder:
    """Collate function for padding destination prediction (DP) task data.
    """

    def __init__(self, device, pred_len, pred_cols):
        """
        Args:
            device (str): name of the device to put tensors on.
            pred_len (int): the length of the tail sub-trajectory to remove from the input trajectory.
            pred_cols (list): the columns to predict.
        """
        self.device = device
        self.pred_len = pred_len
        self.pred_cols = pred_cols

    def __call__(self, raw_batch):
        """
        Returns:
            torch.FloatTensor: the padded batch of trajectory features, with shape (B, L, F).
            torch.LongTensor: the valid lengths of trajectories in the batch, with shape (B).
            torch.FloatTensor: the ground truth of the DP task, i.e., features of the last trajectory point, 
            with shape (B, F).
        """
        traj_batch, valid_lens, label_batch = [], [], []
        for row in raw_batch:
            traj = row[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL]].to_numpy()
            traj = traj[:-self.pred_len]
            valid_len = traj.shape[0]
            traj_batch.append(traj)
            valid_lens.append(valid_len)

            label = row.iloc[-1][self.pred_cols].to_numpy()
            label_batch.append(label)
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)
        label_batch = torch.from_numpy(np.stack(label_batch, 0).astype(float)).float().to(self.device)

        return traj_batch, valid_lens, label_batch


class TtePadder:
    """Collate function for padding travel time estimation (TTE) task data.
    """

    def __init__(self, device):
        """
        Args:
            device (str): name of the device to put tensors on.
        """
        self.device = device

    def __call__(self, raw_batch):
        """
        Returns:
            torch.FloatTensor: the padded batch of trajectory features, with shape (B, L, F).
            torch.LongTensor: the valid lengths of trajectories in the batch, with shape (B).
            torch.FloatTensor: the ground truth of the TTE task, i.e., travel time of trajectories in minutes, 
            with shape (B).
        """
        traj_batch, valid_lens, label_batch = [], [], []
        for row in raw_batch:
            traj = row[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL]].to_numpy()
            traj[1:, COL_I['temporal']] = -1  # Fill the temporal features in trajectory to -1.
            valid_len = traj.shape[0]
            traj_batch.append(traj)
            valid_lens.append(valid_len)
            label_batch.append(row.iloc[-1][DT_COL] / 60)
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)
        label_batch = torch.tensor(label_batch).float().to(self.device)

        return traj_batch, valid_lens, label_batch


def fetch_task_padder(padder_name, device, padder_params):
    if padder_name == 'dp':
        task_padder = DpPadder(device, **padder_params)
    elif padder_name == 'tte':
        task_padder = TtePadder(device, **padder_params)
    else:
        raise NotImplementedError(f'No Padder named {padder_name}')

    return task_padder


def pad_batch(batch):
    """
    Pad the batch to the maximum length of the batch.

    Args:
        batch (list): the batch of arrays to pad, [(L1, F), (L2, F), ...].

    Returns:
        np.array: the padded array.
    """
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch = np.full((len(batch), max_len, batch[0].shape[-1]), FEATURE_PAD, dtype=float)
    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr

    return padded_batch


class TrajectorySearchTestdata:
    def __init__(self, test_dataset:TrajClipDataset, spatial_border, num_target=1000, num_negative=5000, neg_random_choice=False):
        trajs = []
        for traj_id in tqdm(test_dataset.traj_ids, desc='Gathering trips', total=len(test_dataset.traj_ids), leave=False, ncols=70):
            one_traj = test_dataset.traj_df[test_dataset.traj_df[TRAJ_ID_COL] == traj_id].copy()
            one_traj[DT_COL] = one_traj[T_COL] - one_traj[T_COL].iloc[0]
            traj = one_traj[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL]].to_numpy()
            trajs.append(traj)
        self.trajs = np.array(trajs, dtype=object)
        
        num_target = min(len(self.trajs) - 1, num_target)
        random.seed(10)
        sampled_trip_ids = random.sample(range(len(self.trajs)), num_target)
        # generate query trips and target trips from sampled trips
        qry_trips = [t[::2] for t in self.trajs[sampled_trip_ids]]
        tgt_trips = [t[1::2] for t in self.trajs[sampled_trip_ids]]
        self.hopqrytgt = np.array(qry_trips + tgt_trips, dtype=object)
        
        all_hoptgts = [t[1::2] for t in self.trajs]
        self.all_hoptgts = np.array(all_hoptgts, dtype=object)
        
        num_negative = min(len(self.trajs) - num_target, num_negative)
        if neg_random_choice:
            neg_indices = []
            for i in trange(num_target, desc='Gathering sim idx'):
                neg_trip_ids = np.delete(np.arange(len(self.trajs)), sampled_trip_ids[i])
                neg_indice = np.random.choice(neg_trip_ids, num_negative, replace=False)
                neg_indices.append(neg_indice)
        else:
            select_index = COL_I['spatial']
            spatial_border = np.array(spatial_border)
            kseg_trips = []
            for arr in tqdm(self.trajs, desc='Gathering kseg trips', total=len(self.trajs)):
                norm_spatial_arr = (arr[..., select_index] - spatial_border[0]) / (spatial_border[1] - spatial_border[0])
                kseg_arr = self.resample_to_k_segments(norm_spatial_arr, MIN_TRIP_LEN)
                kseg_trips.append(kseg_arr)
            kseg_trips = np.stack(kseg_trips)

            qry_trips = self.hopqrytgt[:num_target]
            # Farthest neighbor search
            neg_euclidean = lambda x, y: - euclidean_distances(x.reshape(1, -1), y.reshape(1, -1))
            farthest_knn = NearestNeighbors(n_neighbors=len(kseg_trips) - 10, metric=neg_euclidean)
            farthest_knn.fit(kseg_trips)

            qry_indices = np.arange(num_target)
            neg_indices = []
            # sampled_trip_ids_array = np.array(sampled_trip_ids)
            for arr in tqdm(kseg_trips[sampled_trip_ids], desc='Gathering sim idx', 
                            total=num_target):
                # negative index: the farthest neighbors of the query
                farthest_idx = farthest_knn.kneighbors([arr], return_distance=False)
                # choose num_negtive neighbors randomly
                farthest_idx = np.random.choice(farthest_idx[0], num_negative, replace=False)
                neg_indices.append(farthest_idx)
        self.neg_indices = np.array(neg_indices)
        print("neg_indices shape: ", self.neg_indices.shape)

        self.hopqrytgt_savename = f"hopqrytgt-{num_target}"
        self.neg_indices_savename = f"hopnearernegindex-{num_target}-{num_negative}-v2" if not neg_random_choice else f"hoprandomnegindex-{num_target}-{num_negative}"

    def save_search_meta(self, meta_dir):
        utils.create_if_noexists(meta_dir)
        np.save(os.path.join(meta_dir, "all_hoptgts.npy"), self.all_hoptgts)
        np.save(os.path.join(meta_dir, f"{self.hopqrytgt_savename}.npy"), self.hopqrytgt)
        np.save(os.path.join(meta_dir, f"{self.neg_indices_savename}.npy"), self.neg_indices)
        print("Saved meta to", meta_dir)

    def get_search_meta(self):
        return self.all_hoptgts, self.hopqrytgt, self.neg_indices
    
    @staticmethod
    def parse_label(length):
        qry_idx = list(range(int(length / 2)))
        tgt_idx = list(range(int(length / 2), length))
        return qry_idx, tgt_idx

    @staticmethod
    def cal_pres_and_labels(query, target, negs):
        """
        query: (N, d)
        target: (N, d)
        negs: (N, n, d)
        """
        num_queries = query.shape[0]
        num_targets = target.shape[0]
        num_negs = negs.shape[1]
        print("query: ", query.shape)
        print("target: ", target.shape)
        print("neg: ", negs.shape)
        assert num_queries == num_targets, "Number of queries and targets should be the same."

        query_t = repeat(query, 'nq d -> nq nt d', nt=num_targets)
        query_n = repeat(query, 'nq d -> nq nn d', nn=num_negs)
        target = repeat(target, 'nt d -> nq nt d', nq=num_queries)
        # negs = repeat(negs, 'nn d -> nq nn d', nq=num_queries)

        dist_mat_qt = np.linalg.norm(query_t - target, ord=2, axis=2)
        dist_mat_qn = np.linalg.norm(query_n - negs, ord=2, axis=2)
        dist_mat = np.concatenate([dist_mat_qt[np.eye(num_queries).astype(bool)][:, None], dist_mat_qn], axis=1)

        pres = -1 * dist_mat

        labels = np.zeros(num_queries)

        return pres, labels
    
    @staticmethod
    def resample_to_k_segments(trip, kseg):
        """
        Resample a trajectory to k segments.
        :return: a numpy array of shape (kseg * 3,)
        """
        ksegs = []
        seg = len(trip) // kseg

        for i in range(kseg):
            if i == kseg - 1:
                ksegs.append(np.mean(trip[i * seg:], axis=0))
            else:
                ksegs.append(np.mean(trip[i * seg: i * seg + seg], axis=0))
        ksegs = np.array(ksegs).reshape(-1)

        return ksegs

def load_trajSearch_testdata(search_meta_dir, num_target=1000, num_negative=5000, neg_random_choice=False):
    neg_indices_metaname = f"hopnearernegindex-{num_target}-{num_negative}-v2.npy" if not neg_random_choice else \
                             f"hoprandomnegindex-{num_target}-{num_negative}.npy"
    print("neg_indices_type:", neg_indices_metaname)
    
    alltrajtgt = np.load(os.path.join(search_meta_dir, "all_hoptgts.npy"), allow_pickle=True)
    hopqrytgt = np.load(os.path.join(search_meta_dir, f"hopqrytgt-{num_target}.npy"), allow_pickle=True)
    neg_indices = np.load(os.path.join(search_meta_dir, neg_indices_metaname), allow_pickle=True)

    return alltrajtgt, hopqrytgt, neg_indices


class TrajectorySearchDataset(Dataset):
    def __init__(self, trajs):
        super().__init__()
        self.trajs = trajs
    def __len__(self):
        return self.trajs.shape[0]

    def __getitem__(self, index):
        one_traj = self.trajs[index].copy()
        return one_traj

class SearchPadder:
    """Collate function for padding data of similar trajectory search.
    """

    def __init__(self, device):
        """
        Args:
            device (str): name of the device to put tensors on.
        """
        self.device = device

    def __call__(self, raw_batch):
        """Collate function for padding the raw batch of trajectory DataFrames into Tensors.

        Args:
            raw_batch (list): each item is a `pd.DataFrame` representing one trajectory.

        Returns:
            torch.FloatTensor: the padded batch of trajectory features, with shape (B, L, F).
            torch.LongTensor: the valid lengths of trajectories in the batch, with shape (B).
        """
        traj_batch, valid_lens = [], []
        for traj in raw_batch:
            valid_len = traj.shape[0]
            traj_batch.append(traj)
            valid_lens.append(valid_len)
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)

        return traj_batch, valid_lens


if __name__ == '__main__':
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-s', '--settings', help='name of the settings file to use', type=str, default="local_test_search") # required=True
    args = parser.parse_args()

    # Load the settings file, and save a backup in the cache directory.
    with open(os.path.join('settings', f'{args.settings}.json'), 'r') as fp:
        settings = json.load(fp)
    # Iterate through the multiple settings.
    for setting_i, setting in enumerate(settings):
        print(f'===SETTING {setting_i}/{len(settings)}===')
        SAVE_NAME = setting.get('save_name', None)

        if 'test' in setting:
            train_traj_df = pd.read_hdf(setting['dataset']['train_traj_df'], key='trips')
            test_traj_df = pd.read_hdf(setting['dataset']['test_traj_df'], key='trips')
            train_dataset = TrajClipDataset(traj_df=train_traj_df)
            test_dataset = TrajClipDataset(traj_df=test_traj_df)
            simTrajSearch_testData = TrajectorySearchTestdata(test_dataset, spatial_border=train_dataset.spatial_border, **setting['test']["search_data_params"])

            eval_dataset = os.path.basename(setting['dataset']['test_traj_df']).split(".")[0]
            meta_dir = os.path.join(SEARCH_META_DIR, eval_dataset)
            simTrajSearch_testData.save_search_meta(meta_dir)