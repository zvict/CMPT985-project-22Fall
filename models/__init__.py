import torch.nn as nn
from .vabank import VolumetricBank
from .lpips import LPNet
from .renderer import MVCRenderer


class BasicLoss(nn.Module):
    def __init__(self, losses_and_weights):
        super(BasicLoss, self).__init__()
        self.losses_and_weights = losses_and_weights

    def forward(self, pred, target):
        loss = 0
        for weight, loss_name in self.losses_and_weights.items():
            loss += float(weight) * loss_name(pred, target)
        return loss


def get_model(args):
    return VolumetricBank(args)


def get_renderer(args, num_pts):
    return MVCRenderer(args, num_pts) 


def get_loss(args):
    losses = nn.ModuleDict()
    for loss_name, weight in args.items():
        if weight > 0:
            if loss_name == "mse":
                losses[str(format(weight, '.0e'))] = nn.MSELoss()
            elif loss_name == "lpips":
                lpips = LPNet()
                lpips.eval()
                losses[str(format(weight, '.0e'))] = lpips
            else:
                raise NotImplementedError('loss [{:s}] is not supported'.format(loss_name))
    return BasicLoss(losses)