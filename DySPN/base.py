from DySPN.module import Dynamic_deformable_DySample_restart, Dynamic_deformable_DySamplev6, Dynamic_7x7_naivev2, \
    cspn_3x3_naive, dspn_3x3_naive, DySPN_Module, Dynamic_deformablev2
from DySPN.enc_dec import BaseModel, BaseModelv2
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, iteration=3, num_neighbor=9, mode="dyspn", shuffle_up=False, norm_depth=[0.1, 8.0], res="res18",
                 bm="v1", norm_layer='bn', stodepth=True):
        super(Model, self).__init__()
        self.sto = stodepth
        self.mode = mode
        # self.sto = False
        assert res in ["res18", "res34"]
        self.res = res
        assert bm in ["v1", "v2"]
        self.bm = bm
        self.shuffle_up = shuffle_up
        assert mode in ["naive", "restart", "cspn", "dspn", "deform_dyspn", "dyspn"]
        self.mode = mode
        assert norm_layer in ['bn', 'in']
        self.norm_layer = norm_layer
        self.iteration = iteration
        self.num_sample = num_neighbor
        if self.bm == "v1":
            BM = BaseModel
        else:
            BM = BaseModelv2
        self.base = BM(iteration=self.iteration, num_sample=self.num_sample, mode=self.mode, sto=self.sto, res=self.res,
                       suffle_up=self.shuffle_up, norm_layer=self.norm_layer)
        if self.mode == "naive":
            self.dyspn_naive = Dynamic_7x7_naivev2()
        elif self.mode == "restart":
            self.dyspn_restart = Dynamic_deformable_DySample_restart(prop_time=iteration)
        elif self.mode == "cspn":
            self.cspn = cspn_3x3_naive(iteration=iteration)
        elif self.mode == "dspn":
            self.dspn = dspn_3x3_naive(iteration=iteration)
        elif self.mode == "deform_dyspn":
            self.deform_dyspn = Dynamic_deformablev2(iteration=iteration)
        elif self.mode == "dyspn":
            exec("self.dyspn_{}_{}=DySPN_Module(iteration=self.iteration,num=self.num_sample,mode='yx')".format(
                self.iteration,
                self.num_sample))
        else:
            self.base = BM(iteration * 9 + 1 + 1, mode=self.mode, sto=self.sto, res=self.res, suffle_up=self.shuffle_up)
            assert self.num_sample == 9
            self.dyspn = Dynamic_deformable_DySamplev6(prop_time=iteration)
        self.norm = norm_depth

    def forward(self, rgb0, dep):
        guide = self.base(rgb0, dep)

        #######################################

        if self.mode == "naive":
            pred_init = guide[:, 73:74, :, :]
            confidence = torch.sigmoid(guide[:, 72:73, :, :])
            output = self.dyspn_naive(pred_init,
                                      guide[:, 0:48, :, :],
                                      guide[:, 48:72, :, :],
                                      confidence,
                                      dep)
        elif self.mode == "restart":
            pred_init = guide[:, 31:32, :, :]
            output = self.dyspn_restart(pred_init,
                                        guide[:, 0:30, :, :],
                                        guide[:, 30:31, :, :],
                                        dep)
        elif self.mode == "cspn":
            pred_init = guide[:, 9:10, :, :]
            output = self.cspn(pred_init,
                               guide[:, 0:8, :, :],
                               guide[:, 8:9, :, :],
                               dep)
        elif self.mode == "dspn":
            pred_init = guide[:, 10:11, :, :]
            output = self.dspn(pred_init,
                               guide[:, 0:9, :, :],
                               guide[:, 9:10, :, :],
                               dep)

        elif self.mode == "deform_dyspn":
            pred_init = guide[:, 49:50, :, :]
            output = self.deform_dyspn(pred_init,
                                       guide[:, 0:24, :, :],
                                       guide[:, 24:48, :, :],
                                       guide[:, 48:49, :, :],
                                       dep)

        elif self.mode == "dyspn":
            pred_init = guide.narrow(1, self.iteration * self.num_sample + 1, 1)

            output = eval(
                "self.dyspn_{}_{}(pred_init,guide.narrow(1,0,self.iteration*self.num_sample),dep,guide.narrow(1,self.iteration*self.num_sample,1))".format(
                    self.iteration, self.num_sample))
        else:
            pred_init = guide[:, 28:29, :, :]
            output = self.dyspn(pred_init,
                                guide[:, 0:27, :, :],
                                guide[:, 27:28, :, :],
                                dep)
        return output
