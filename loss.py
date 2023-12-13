import torch
import torch.nn as nn
import math


class LossFirstStage(nn.Module):
    def __init__(self, beta=1e-3):
        super().__init__()
        self.bce = nn.BCELoss()
        self.beta = beta

    def forward(self, pred, label, mu, std, mask_v, mask_l):
        # mask_v: B * 1
        # reference:
        # https://github.com/1Konny/VIB-pytorch/blob/master/solver.py
        class_loss = self.bce(pred * mask_v * mask_l, label * mask_v * mask_l).div(math.log(2))
        info_loss = -0.5 * ((1 + 2 * std.log() - mu.pow(2) - std.pow(2)) * mask_v).mean().div(math.log(2))
        return class_loss + self.beta * info_loss


class LossSecondStage(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss()

    def forward(self, pred, label, mask_v, mask_l, data_r, rec_r):
        assert pred.shape == label.shape == mask_l.shape
        assert len(data_r) == len(rec_r) == mask_v.shape[1]
        n_view = len(data_r)

        mse1 = mse2 = 0
        index = [i for i in range(n_view)]
        for v in range(n_view):
            index_ = index.copy()
            index_.remove(v)
            mse1 += torch.mean(torch.pow(data_r[v].detach() - rec_r[v][:, v, :], 2) * mask_v[:, v:v+1])
            mse2 += torch.mean(torch.pow(((data_r[v].detach()).unsqueeze(1).expand(-1, n_view-1, -1) - rec_r[v][:, index_, :]) * (mask_v[:, v:v+1].expand(-1, n_view-1) * mask_v[:, index_]).unsqueeze(2), 2))

        return self.bce(pred * mask_l, label * mask_l) + self.alpha * mse1 / n_view + self.gamma * mse2 / n_view
