import torchvision
import torch
import torch.nn as nn
from DySPN.common import conv_bn_relu, conv_shuffle_bn_relu, convt_bn_relu
from DySPN.stodepth_lineardecay import se_resnet34_StoDepth_lineardecay, se_resnet18_StoDepth_lineardecay,se_resnet68_StoDepth_lineardecay


class BaseModel(nn.Module):
    def __init__(self, iteration, num_sample, mode='naive', sto=True, res="res34", suffle_up=False, norm_layer=None):
        super(BaseModel, self).__init__()

        assert mode in ["naive", "restart", "cspn", "dspn", "deform_dyspn", "dyspn"]
        self.mode = mode
        if self.mode == "naive":
            out_channel = 48 + 4 * iteration + 1 + 1
        elif self.mode == "restart":
            out_channel = iteration * (num_sample + 1) + 1 + 1
        elif self.mode == "cspn":
            assert num_sample == 8
            out_channel = num_sample + 1 + 1
        elif self.mode == "dspn":
            assert num_sample == 9
            out_channel = num_sample + 1 + 1
        elif self.mode == "deform_dyspn":
            out_channel = 3 * 8 + 4 * iteration + 1 + 1
        elif self.mode == "dyspn":
            out_channel = iteration * num_sample + 1 + 1
        else:
            raise NotImplementedError
        self.out_channel = out_channel
        self.iteration = iteration
        self.num_sample = num_sample
        self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                      bn=False)
        if sto == True:
            if res == "res18":
                net = se_resnet18_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=True)
            else:
                net = se_resnet34_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=True)
                # net = se_resnet68_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=True)
        else:
            if res == "res18":
                net = torchvision.models.resnet18(pretrained=True)
            else:
                net = torchvision.models.resnet34(pretrained=True)
        # 1/1
        self.conv2 = net.layer1
        # 1/2
        self.conv3 = net.layer2
        # 1/4
        self.conv4 = net.layer3
        # 1/8
        self.conv5 = net.layer4

        del net

        # # 1/16
        self.conv6 = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1)

        if suffle_up == True:
            # 1/8
            self.dec5 = conv_shuffle_bn_relu(512, 256, kernel=3, stride=1, padding=1)
            # 1/4
            self.dec4 = conv_shuffle_bn_relu(256 + 512, 128, kernel=3, stride=1, padding=1)
            # 1/2
            self.dec3 = conv_shuffle_bn_relu(128 + 256, 64, kernel=3, stride=1, padding=1)
            # 1/
            self.dec2 = conv_shuffle_bn_relu(64 + 128, 64, kernel=3, stride=1, padding=1)
        else:
            # Shared Decoder
            # # 1/8
            self.dec5 = convt_bn_relu(512, 256, kernel=3, stride=2, padding=1, output_padding=1)
            # 1/4
            self.dec4 = convt_bn_relu(256 + 512, 128, kernel=3, stride=2, padding=1, output_padding=1)
            # 1/2
            self.dec3 = convt_bn_relu(128 + 256, 64, kernel=3, stride=2, padding=1, output_padding=1)
            # 1/
            self.dec2 = convt_bn_relu(64 + 128, 64, kernel=3, stride=2, padding=1, output_padding=1)

        # Guidance Branch
        # 1/1
        self.gd_dec1_ = conv_bn_relu(64 + 64, 64, kernel=3, stride=1,
                                     padding=1)

        if self.mode == "naive":
            self.gd_dec0_naive = conv_bn_relu(64 + 64, out_channel, kernel=3, stride=1,
                                              padding=1, bn=False, relu=False)
        elif self.mode == "restart":
            self.gd_dec0_restart = conv_bn_relu(64 + 64, out_channel, kernel=3, stride=1,
                                                padding=1, bn=False, relu=False)
        elif self.mode == "cspn":
            self.gd_dec0_cspn = conv_bn_relu(64 + 64, out_channel, kernel=3, stride=1,
                                             padding=1, bn=False, relu=False)
        elif self.mode == "dspn":
            self.gd_dec0_dspn = conv_bn_relu(64 + 64, out_channel, kernel=3, stride=1,
                                             padding=1, bn=False, relu=False)
        elif self.mode == "deform_dyspn":
            self.gd_dec0_deform_dyspn = conv_bn_relu(64 + 64, out_channel, kernel=3, stride=1,
                                             padding=1, bn=False, relu=False)
        elif self.mode == "dyspn":
            exec(
                "self.gd_dec0_{}_{}_{} = conv_bn_relu(64 + 64, self.out_channel, kernel=3, stride=1, padding=1, bn=False, relu=False)".format(
                    self.mode, self.iteration, self.num_sample))
        else:
            self.gd_dec0_ = conv_bn_relu(64 + 64, out_channel, kernel=3, stride=1,
                                         padding=1, bn=False, relu=False)

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, rgb, dep):
        fe1_rgb = self.conv1_rgb(rgb)

        fe1_dep = self.conv1_dep(dep)

        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)

        # Shared Decoding
        fd5 = self.dec5(fe6)
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Guidance Decoding
        gd_fd1 = self.gd_dec1_(self._concat(fd2, fe2))
        if self.mode == "naive":
            guide = self.gd_dec0_naive(self._concat(gd_fd1, fe1))
        elif self.mode == "restart":
            guide = self.gd_dec0_restart(self._concat(gd_fd1, fe1))
        elif self.mode == "cspn":
            guide = self.gd_dec0_cspn(self._concat(gd_fd1, fe1))
        elif self.mode == "dspn":
            guide = self.gd_dec0_dspn(self._concat(gd_fd1, fe1))
        elif self.mode == "deform_dyspn":
            guide = self.gd_dec0_deform_dyspn(self._concat(gd_fd1, fe1))
        elif self.mode == "dyspn":
            guide = eval(
                " self.gd_dec0_{}_{}_{}(self._concat(gd_fd1, fe1))".format(self.mode, self.iteration, self.num_sample))
        else:
            guide = self.gd_dec0_(self._concat(gd_fd1, fe1))

        return guide


class BaseModelv2(nn.Module):
    def __init__(self, iteration, num_sample, mode='naive', sto=True, res="res34", suffle_up=False, norm_layer='bn'):
        # def __init__(self):
        super(BaseModelv2, self).__init__()

        assert mode in ["naive", "restart", "cspn", "dspn", "dyspn"]
        assert norm_layer in ['bn', 'in']
        self.mode = mode
        if self.mode == "naive":
            out_channel = 48 + 4 * iteration + 1 + 1
        elif self.mode == "restart":
            out_channel = iteration * (num_sample + 1) + 1 + 1
        elif self.mode == "cspn":
            assert num_sample == 8
            out_channel = num_sample + 1 + 1
        elif self.mode == "dspn":
            assert num_sample == 9
            out_channel = num_sample + 1 + 1
        elif self.mode == "dyspn":
            out_channel = iteration * num_sample + 1 + 1
        else:
            raise NotImplementedError
        self.out_channel = out_channel
        self.iteration = iteration
        self.num_sample = num_sample
        # 1/1
        self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv1 = conv_bn_relu(64, 64, kernel=3, stride=2, padding=1, norm_layer=norm_layer)
        if sto == True:
            if res == "res18":
                net = se_resnet18_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=False, norm_layer=norm_layer)
            else:

                net = se_resnet34_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=True, norm_layer=norm_layer)
        else:
            if res == "res18":
                net = torchvision.models.resnet18(pretrained=True)
            else:
                net = torchvision.models.resnet34(pretrained=True)
        # 1/2 64
        self.conv2 = net.layer1
        # 1/4 128
        self.conv3 = net.layer2
        # 1/8 256
        self.conv4 = net.layer3
        # 1/16 512
        self.conv5 = net.layer4

        del net

        if suffle_up == True:
            # 1/16
            self.dec5_ = conv_shuffle_bn_relu(512, 256, kernel=3, stride=1, padding=1, norm_layer=norm_layer)
            # 1/8
            self.dec4_ = conv_shuffle_bn_relu(256 + 256, 128, kernel=3, stride=1, padding=1, norm_layer=norm_layer)
            # 1/4
            self.dec3_ = conv_shuffle_bn_relu(128 + 128, 64, kernel=3, stride=1, padding=1, norm_layer=norm_layer)
            # 1/2
            self.dec2_ = conv_shuffle_bn_relu(64 + 64, 64, kernel=3, stride=1, padding=1, norm_layer=norm_layer)
            # 1
            # self.dec1_ = conv_shuffle_bn_relu(64 + 64, 64, kernel=3, stride=2, padding=1)
        else:
            # 1/16
            self.dec5_ = convt_bn_relu(512, 256, kernel=3, stride=2, padding=1, output_padding=1, norm_layer=norm_layer)
            # 1/8
            self.dec4_ = convt_bn_relu(256 + 256, 128, kernel=3, stride=2, padding=1, output_padding=1,
                                       norm_layer=norm_layer)
            # 1/4
            self.dec3_ = convt_bn_relu(128 + 128, 64, kernel=3, stride=2, padding=1, output_padding=1,
                                       norm_layer=norm_layer)
            # 1/2
            self.dec2_ = convt_bn_relu(64 + 64, 64, kernel=3, stride=2, padding=1, output_padding=1,
                                       norm_layer=norm_layer)
        # Guidance Branch
        # 1/1
        self.gd_dec1_ = conv_bn_relu(64 + 64, 128, kernel=3, stride=1, padding=1, norm_layer=norm_layer)

        self.mode = mode
        if self.mode == "naive":
            self.gd_dec0_naive = conv_bn_relu(64 + 64, out_channel, kernel=3, stride=1,
                                              padding=1, bn=False, relu=False, norm_layer=norm_layer)
        elif self.mode == "restart":
            self.gd_dec0_restart = conv_bn_relu(64 + 64, out_channel, kernel=3, stride=1,
                                                padding=1, bn=False, relu=False, norm_layer=norm_layer)
        elif self.mode == "cspn":
            self.gd_dec0_cspn = conv_bn_relu(64 + 64, out_channel, kernel=3, stride=1,
                                             padding=1, bn=False, relu=False, norm_layer=norm_layer)
        elif self.mode == "dspn":
            self.gd_dec0_dspn = conv_bn_relu(64 + 64, out_channel, kernel=3, stride=1,
                                             padding=1, bn=False, relu=False, norm_layer=norm_layer)
        elif self.mode == "dyspn":
            exec(
                "self.gd_dec0_{}_{}_{} = conv_bn_relu(64 + 64, self.out_channel, kernel=3, stride=1, padding=1, bn=False, relu=False,norm_layer=norm_layer)".format(
                    self.mode, self.iteration, self.num_sample))
        else:
            self.gd_dec0_ = conv_bn_relu(64 + 64, out_channel, kernel=3, stride=1,
                                         padding=1, bn=False, relu=False, norm_layer=norm_layer)

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, rgb, dep):
        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)
        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
        # fe2 = self.conv2(fe1)
        fe2 = self.conv1(fe1)
        fe2 = self.conv2(fe2)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)

        # Shared Decoding
        fd4 = self.dec5_(fe5)
        fd3 = self.dec4_(self._concat(fd4, fe4))
        fd2 = self.dec3_(self._concat(fd3, fe3))
        fd1 = self.dec2_(self._concat(fd2, fe2))

        # Guidance Decoding
        gd_fd1 = self.gd_dec1_(self._concat(fd1, fe1))

        if self.mode == "naive":
            guide = self.gd_dec0_naive(gd_fd1)
        elif self.mode == "restart":
            guide = self.gd_dec0_restart(gd_fd1)
        elif self.mode == "cspn":
            guide = self.gd_dec0_cspn(gd_fd1)
        elif self.mode == "dspn":
            guide = self.gd_dec0_dspn(gd_fd1)
        elif self.mode == "dyspn":
            guide = eval(" self.gd_dec0_{}_{}_{}(gd_fd1)".format(self.mode, self.iteration, self.num_sample))
        else:
            guide = self.gd_dec0_(gd_fd1)
        return guide
