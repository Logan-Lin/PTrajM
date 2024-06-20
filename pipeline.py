import numpy as np
import torch
from tqdm import trange, tqdm


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
            for batch in tqdm(dataloader, desc='-->Traversing', leave=False, ncols=50):
                optimizer.zero_grad()
                loss = model.loss(*batch)
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
            bar.set_description(bar_desc % np.mean(loss_values))


def finetune_model(model, pred_head, dataloader, num_epoch, lr, ft_encoder=True, denormalize=False):
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
        optimizer = torch.optim.Adam(list(model.parameters()) + list(pred_head.parameters()), lr=lr)
        model.train()
    else:
        optimizer = torch.optim.Adam(pred_head.parameters(), lr=lr)
        model.eval()

    bar_desc = 'Finetuning, avg loss: %.5f'
    with trange(num_epoch, desc=bar_desc % 0.0) as bar:
        for epoch_i in bar:
            loss_values = []
            for batch in tqdm(dataloader, desc='-->Traversing', leave=False, ncols=50):
                *input_batch, label = batch

                optimizer.zero_grad()
                traj_h = model(*input_batch)
                if not ft_encoder:
                    traj_h = traj_h.detach()
                loss = pred_head.loss(traj_h, label, denormalize)
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
            bar.set_description(bar_desc % np.mean(loss_values))


@torch.no_grad()
def test_model(model, pred_head, dataloader, denormalize=False):
    """Test the model with specific prediction tasks.

    Args:
        model (nn.Module): the trajectory embedding model to test.
        pred_head (nn.Module): the prediction head for mapping trajectory embeddings to predictions.
        dataloader (DataLoader): batch iterator containing the testing data.
    """
    model.eval()
    pred_head.eval()

    predictions, targets = [], []
    for batch in tqdm(dataloader, 'Testing', ncols=50):
        *input_batch, target = batch
        traj_h = model(*input_batch)
        pred = pred_head(traj_h)
        if denormalize: # GPS预测值反归一化
            pred = pred * (pred_head.spatial_border[1] - pred_head.spatial_border[0]).unsqueeze(0) + \
                        pred_head.spatial_border[0].unsqueeze(0)
        predictions.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
    predictions = np.concatenate(predictions, 0)
    targets = np.concatenate(targets, 0)
    return predictions, targets