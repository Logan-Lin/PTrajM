from torch import nn
from torch.nn import functional as F


class MlpPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_type):
        """
        Args:
            input_size (int): number of input feature dimension.
            hidden_size (int): number of hidden feature dimension.
            output_size (int): number of output feature dimension.
            pred_type (str): type of prediction, 'regression' or 'classification'.
        """
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.pred_type = pred_type
    
    def forward(self, traj_h):
        pred = self.net(traj_h)
        if self.pred_type == 'classification':
            pred = pred.argmax(-1)
        return pred

    def loss(self, traj_h, label):
        pred = self.net(traj_h)

        if self.pred_type == 'regression':
            loss = F.mse_loss(pred, label)
        elif self.pred_type == 'classification':
            loss = F.cross_entropy(pred, label)
        else:
            raise NotImplementedError(f'No prediction type: {self.pred_type}.')
        
        return loss

