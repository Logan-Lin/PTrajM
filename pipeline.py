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


def finetune_model(model, pred_head, dataloader, num_epoch, lr, ft_encoder=True):
    """Fine-tune the model with specific task labels.

    Args:
        model (nn.Module): the model to finetune.
        pred_head (nn.Module): the prediction head for mapping the embeddings to predictions.
        dataloader (DataLoader): batch iterator containing the finetune data.
        num_epoch (int): number of epoches to finetune.
        lr (float): learning rate of the optimizer.
        ft_encoder (bool, optional): Whether to finetune the trajectory encoder. Defaults to True.
        If set to False, then only the task-specific prediction module will be finetuned.
    """
    pred_head.train()
    if ft_encoder:
        optimizer = torch.optim.Adam(list(model.parameters() + pred_head.parameters()), lr=lr)
        model.train()
    else:
        optimizer = torch.optim.Adam(pred_head.parameters(), lr=lr)
        model.eval()

    bar_desc = 'Finetuning, avg loss: %.5f'
    with trange(num_epoch, desc=bar_desc % 0.0) as bar:
        for epoch_i in bar:
            loss_values = []
            for batch in tqdm(dataloader, desc='-->Traversing', leave=False):
                *input_batch, label = batch

                optimizer.zero_grad()
                traj_h = model(*input_batch)
                if not ft_encoder:
                    traj_h = traj_h.detach()
                loss = pred_head.loss(traj_h, label)
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
            bar.set_description(bar_desc % np.mean(loss_values))