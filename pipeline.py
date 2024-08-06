import numpy as np
import torch
from tqdm import trange, tqdm
from data import TrajectorySearchTestdata

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
    with trange(num_epoch, desc=bar_desc % 0.0, position=0) as bar:
        for epoch_i in bar:
            loss_values = []
            for batch in tqdm(dataloader, desc='-->Traversing', leave=False, ncols=60):
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
    with trange(num_epoch, desc=bar_desc % 0.0, position=0) as bar:
        for epoch_i in bar:
            loss_values = []
            for batch in tqdm(dataloader, desc='-->Traversing', leave=False, ncols=60):
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
    for batch in tqdm(dataloader, 'Testing', ncols=60):
        *input_batch, target = batch
        traj_h = model(*input_batch)
        pred = pred_head(traj_h)
        if denormalize:
            pred = pred * (pred_head.spatial_border[1] - pred_head.spatial_border[0]).unsqueeze(0) + \
                        pred_head.spatial_border[0].unsqueeze(0)
        predictions.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
    predictions = np.concatenate(predictions, 0)
    targets = np.concatenate(targets, 0)
    return predictions, targets


@torch.no_grad()
def test_model_on_search(model, traj_dataloader, qrytgt_dataloader, neg_indices, set_name="test"):
    """Test the model with similar trajectory search.

    Args:
        model (nn.Module): the trajectory embedding model to test.
        dataloader (DataLoader): batch iterator containing the testing data.
    """
    model.eval()

    qrytgt_embeds = []
    for batch_meta in tqdm(qrytgt_dataloader,
                            desc=f"Calculating query and target embeds on {set_name} set",
                            total=len(qrytgt_dataloader), ncols=60):
        encodes = model(*batch_meta)
        qrytgt_embeds.append(encodes.detach().cpu().numpy())
    qrytgt_embeds = np.concatenate(qrytgt_embeds, 0)
    qry_indices, tgt_indices = TrajectorySearchTestdata.parse_label(len(qrytgt_embeds))

    embeds = []
    whole_enc_time = []
    traj_process_time = []
    for batch_meta in tqdm(traj_dataloader,
                            desc=f"Calculating embeds on {set_name} set",
                            total=len(traj_dataloader), ncols=60):
        encodes, enc_time, process_time = model.forward_on_search_mode(*batch_meta)
        embeds.append(encodes.detach().cpu().numpy())
        whole_enc_time.append(enc_time)
        traj_process_time.append(process_time)
    whole_enc_time = np.array(whole_enc_time)
    traj_process_time = np.array(traj_process_time)
    print("Embedding time: {:.3f}s".format(whole_enc_time.sum()))
    print("Check traj process time: {:.3f}s".format(traj_process_time.sum()))
    embeds = np.concatenate(embeds, 0)

    predictions, targets = TrajectorySearchTestdata.cal_pres_and_labels(qrytgt_embeds[qry_indices], qrytgt_embeds[tgt_indices], embeds[neg_indices])

    return predictions, targets