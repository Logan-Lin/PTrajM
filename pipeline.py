import os

import numpy as np
import torch
from torch import nn
from tqdm import trange, tqdm

from utils import create_if_noexists


MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', 'saved_model')


def pretrain_model(model, dataloader, num_epoch, lr,
                   save_model=False, load_model=False, save_name=None):
    if load_model:
        model.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{save_name}.pretrain'),
                                         map_location=model.device))
        return model

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

    if save_model:
        create_if_noexists(MODEL_CACHE_DIR)
        torch.save(model.state_dict(), os.path.join(MODEL_CACHE_DIR, f'{save_name}.pretrain'))
