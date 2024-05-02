import torch
import numpy as np
from torch.utils.data import Dataset


TRAJ_ID_COL = 'trip'
X_COL = 'lng'
Y_COL = 'lat'
T_COL = 'timestamp'
DT_COL = 'delta_t'
ROAD_COL = 'road'
FEATURE_PAD = 0


class TrajClipDataset(Dataset):
    def __init__(self, traj_df):
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
    def __init__(self, device):
        self.device = device

    def __call__(self, raw_batch):
        traj_batch, valid_lens = [], []
        for row in raw_batch:
            traj = row[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL]].to_numpy()
            valid_len = traj.shape[0]
            traj_batch.append(traj)
            valid_lens.append(valid_len)
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)

        return traj_batch, valid_lens


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
