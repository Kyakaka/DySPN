"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    Some of useful functions are defined here.
"""


import torch
import torch.nn as nn
import torchvision


model_path = {
    'resnet18': 'pretrained/resnet18.pth',
    'resnet34': 'pretrained/resnet34.pth'
}


def get_resnet18(pretrained=True):
    net = torchvision.models.resnet18(pretrained=False)
    if pretrained:
        state_dict = torch.load(model_path['resnet18'])
        net.load_state_dict(state_dict)

    return net


def get_resnet34(pretrained=True):
    net = torchvision.models.resnet34(pretrained=False)
    if pretrained:
        state_dict = torch.load(model_path['resnet34'])
        net.load_state_dict(state_dict)

    return net


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True,norm_layer='bn'):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)
    assert norm_layer in ['bn', 'in']
    if norm_layer == 'in':
        NL = nn.InstanceNorm2d
    else:
        NL = nn.BatchNorm2d
    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(NL(ch_out))
    if relu:
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.ReLU(inplace=True))
    layers = nn.Sequential(*layers)

    return layers

def conv_bn_relu_bias(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True, bias=True,norm_layer='bn'):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)
    assert norm_layer in ['bn', 'in']
    if norm_layer == 'in':
        NL = nn.InstanceNorm2d
    else:
        NL = nn.BatchNorm2d
    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=bias))
    if bn:
        layers.append(NL(ch_out))
    if relu:
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers
def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True,norm_layer='bn'):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)
    assert norm_layer in ['bn', 'in']
    if norm_layer == 'in':
        NL = nn.InstanceNorm2d
    else:
        NL = nn.BatchNorm2d
    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(NL(ch_out))
    if relu:
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.ReLU(inplace=True))
    layers = nn.Sequential(*layers)

    return layers

def conv_up_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0,
                  bn=True, relu=True,norm_layer='bn'):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)
    assert norm_layer in ['bn', 'in']
    if norm_layer == 'in':
        NL = nn.InstanceNorm2d
    else:
        NL = nn.BatchNorm2d
    layers = []
    layers.append(nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False))
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(NL(ch_out))
    if relu:
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.ReLU(inplace=True))
    layers = nn.Sequential(*layers)

    return layers


def conv_shuffle_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0,
                  bn=True, relu=True,norm_layer='bn'):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)
    assert norm_layer in ['bn', 'in']
    if norm_layer == 'in':
        NL = nn.InstanceNorm2d
    else:
        NL = nn.BatchNorm2d
    layers = []

    layers.append(nn.Conv2d(ch_in, ch_out*4, kernel, stride, padding,
                            bias=not bn))
    layers.append(nn.PixelShuffle(2))
    if bn:
        layers.append(NL(ch_out))
    if relu:
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.ReLU(inplace=True))
    layers = nn.Sequential(*layers)

    return layers

