import torch
import numpy as np
from torch.utils.data import Dataset


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

        self.traj_df = traj_df

        self.traj_df['timestamp'] = self.traj_df['time'].apply(lambda x: x.timestamp())
        self.traj_ids = self.traj_df[TRAJ_ID_COL].unique()

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
