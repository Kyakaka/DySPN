import torch
import torch.nn as nn


class SLoss(nn.Module):
    def __init__(self, w1=1.0, w2=1.0, depth_range=None):
        super(SLoss, self).__init__()
        if depth_range is None:
            depth_range = [0.1, 100]
        self.l1loss = L1Loss(depth_range)
        self.l2loss = L2Loss(depth_range)
        self.w1 = w1
        self.w2 = w2

    def forward(self, predict, gt):
        l1 = self.l1loss(predict, gt)
        l2 = self.l2loss(predict, gt)
        return self.w1 * l1 + self.w2 * l2


class L1Loss(nn.Module):
    def __init__(self, depth_range=None):
        super(L1Loss, self).__init__()
        self.depth_range = depth_range

    def forward(self, predict, gt):
        mask = torch.where((gt > self.depth_range[0]) & (gt < self.depth_range[1]))
        loss = torch.abs(predict[mask] - gt[mask])
        return loss.mean()


class L2Loss(nn.Module):
    def __init__(self, depth_range=None):
        super(L2Loss, self).__init__()
        self.depth_range = depth_range

    def forward(self, predict, gt):
        mask = torch.where((gt > self.depth_range[0]) & (gt < self.depth_range[1]))
        loss = (predict[mask] - gt[mask]) ** 2
        return loss.mean()




class decay_Loss(nn.Module):
    def __init__(self, depth_range=None):
        super(decay_Loss, self).__init__()
        self.param_list = []
        # for i in range(4):
        #     self.param_list.append(torch.linspace(i, 3 - i, 6))
        # self.param_list = torch.ones(3,3)-torch.eye(3)

        for i in range(4):
            self.param_list.append(torch.linspace(i, 3 - i, 6))

    def forward(self, dynamic):
        loss = 0
        for k in range(6):
            attention = dynamic.narrow(1, 4 * k, 4)
            # loss = self.param_list[0][i] * torch.abs(attention[:, i * 4, ...]) \
            #        + self.param_list[1][i] * torch.abs(attention[:, i * 4 + 1, ...]) \
            #        + self.param_list[2][i] * torch.abs(attention[:, i * 4 + 2, ...]) \
            # + self.param_list[3][i] * torch.abs(attention[:, i * 4 + 3, ...])
            # loss += self.param_list[k//2,0] * attention[:, 0:1, :, :]  \
            #        + self.param_list[k//2,1] * attention[:, 1:2, :, :]  \
            #        + self.param_list[k//2,2] * attention[:, 2:3, :, :]  \
            loss += self.param_list[0][k] * attention[:, 0:1, :, :] \
                    + self.param_list[1][k] * attention[:, 1:2, :, :] \
                    + self.param_list[2][k] * attention[:, 2:3, :, :] \
                    + self.param_list[3][k] * attention[:, 3:4, :, :]
        return loss.mean()

class offset_Loss(torch.nn.Module):
    def __init__(self,num_epoch):
        super(offset_Loss, self).__init__()
        self.offset_penalty = 0.001
        self.num_epoch=num_epoch
        self.annealing = torch.nn.Parameter(torch.linspace(1, 0, steps=self.num_epoch, device=torch.device('cpu')))
        self.annealing.requires_grad=False
    def forward(self, sample,epoch):
        B, _, H, W = sample['pred'].shape
        offset = sample['offset']
        if epoch<self.num_epoch:
            ref_y = torch.linspace(-H + 1, H - 1, H, device=torch.device(offset[0].device))
            ref_x = torch.linspace(-W + 1, W - 1, W, device=torch.device(offset[0].device))
            penalty = torch.linspace(0, self.offset_penalty, steps=len(offset), device=torch.device(offset[0].device))
            loss_sum = 0
            for i in range(len(offset)):
                loss_sum += torch.mean(torch.abs(penalty[i] * (offset[i][..., 0] * W - ref_x.view(1, 1, 1, W)) / 2))
                loss_sum += torch.mean(torch.abs(penalty[i] * (offset[i][..., 1] * H - ref_y.view(1, 1, H, 1)) / 2))

            return self.annealing[epoch]*loss_sum
        else:
            return torch.zeros(1,device=torch.device(offset[0].device))