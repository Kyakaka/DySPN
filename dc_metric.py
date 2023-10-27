"""
    Metric implementation
"""

import torch
from torchmetrics import Metric

class DC_Metric(Metric):
    def __init__(self,eval_range=None):
        super(DC_Metric, self).__init__()
        self.t_valid = 0.0001

        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'delta_1', 'delta_2', 'delta_3'
        ]

        for metric_name in self.metric_name:
            self.add_state(metric_name, default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.eval_range = eval_range
    def update(self, gt, pred):
        with torch.no_grad():
            pred = pred.detach()
            gt = gt.detach()

            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            # For numerical stability
            if self.eval_range is not None:
                mask = ((gt > self.eval_range[0]) & (gt < self.eval_range[1]))
            else:
                mask =  (gt > self.t_valid)

            num_valid = mask.sum()

            pred = pred[mask]
            gt = gt[mask]

            pred_inv = pred_inv[mask]
            gt_inv = gt_inv[mask]

            pred_inv[pred <= self.t_valid] = 0.0
            gt_inv[gt <= self.t_valid] = 0.0

            # RMSE / MAE
            diff = pred - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = diff_sqr.sum() / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = diff_abs.sum() / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pred_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = diff_inv_abs.sum() / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (gt + 1e-8)
            rel = rel.sum() / (num_valid + 1e-8)

            # delta
            r1 = gt / (pred + 1e-8)
            r2 = pred / (gt + 1e-8)
            ratio = torch.max(r1, r2)

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25 ** 2).type_as(ratio)
            del_3 = (ratio < 1.25 ** 3).type_as(ratio)

            del_1 = del_1.sum() / (num_valid + 1e-8)
            del_2 = del_2.sum() / (num_valid + 1e-8)
            del_3 = del_3.sum() / (num_valid + 1e-8)

            self.RMSE += rmse
            self.MAE += mae
            self.iRMSE += irmse
            self.iMAE += imae
            self.REL += rel
            self.delta_1 += del_1
            self.delta_2 += del_2
            self.delta_3 += del_3
            self.total += 1

    def compute(self):
        return self.RMSE.float() / self.total, \
               self.MAE.float() / self.total, \
               self.iRMSE.float() / self.total, \
               self.iMAE.float() / self.total, \
               self.REL.float() / self.total, \
               self.delta_1.float() / self.total, \
               self.delta_2.float() / self.total, \
               self.delta_3.float() / self.total
