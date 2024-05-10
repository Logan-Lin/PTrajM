import os

import numpy as np
import torch
from torch import nn
from tqdm import trange, tqdm

from utils import create_if_noexists


def pretrain_model(model, dataloader, num_epoch, lr):
    """Pre-train the model with the given training dataloader.

    Args:
        model (nn.Module): the model to train.
        dataloader (DataLoader): batch iterator containing the training data.
        num_epoch (int): number of epoches to train.
        lr (float): learning rate for the optimizer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    bar_desc = 'Pretraining, avg loss: %.5f'
    with trange(num_epoch, desc=bar_desc % 0.0) as bar:
        for epoch_i in bar:
            loss_values = []
            for batch in tqdm(dataloader, desc='-->Traversing', leave=False):
                optimizer.zero_grad()
                loss = model.loss(*batch)
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
            bar.set_description(bar_desc % np.mean(loss_values))
