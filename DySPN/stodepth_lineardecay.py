import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
__all__ = ['ResNet_StoDepth_lineardecay', 'resnet18_StoDepth_lineardecay', 'resnet34_StoDepth_lineardecay', 'resnet50_StoDepth_lineardecay', 'resnet101_StoDepth_lineardecay',
           'resnet152_StoDepth_lineardecay']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_dilated(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,dilation=3,
                     padding=3, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# class SALayer(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SALayer, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         avg = torch.mean(x, dim=1, keepdim=True)
#         max, _ = torch.max(x, dim=1, keepdim=True)
#         y = self.conv(torch.cat([avg, max], dim=1))
#         y = torch.sigmoid(y)
#         return x * y.expand_as(x)

class StoDepth_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, prob, multFlag, inplanes, planes, stride=1, downsample=None):
        super(StoDepth_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.multFlag = multFlag

    def forward(self, x):
        
        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(),torch.ones(1)):

                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                
                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:
            

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            if self.multFlag:
                out = self.prob*out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out

class StoDepth_SE_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, prob, multFlag, inplanes, planes, stride=1, downsample=None, norm_layer='bn'):
        super(StoDepth_SE_BasicBlock, self).__init__()
        assert norm_layer in ['bn','in']
        if norm_layer=='in':
            NL = nn.InstanceNorm2d
        else:
            NL = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = NL(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = NL(planes)
        self.se = SELayer(planes, reduction=4)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.multFlag = multFlag

    def forward(self, x):

        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(), torch.ones(1)):

                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True
                self.se.requires_grad = True
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.se(out)
                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                self.se.requires_grad = False
                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.se(out)
            if self.downsample is not None:
                identity = self.downsample(x)

            if self.multFlag:
                out = self.prob * out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out


class StoDepth_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, prob, multFlag, inplanes, planes, stride=1, downsample=None):
        super(StoDepth_Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.multFlag = multFlag

    def forward(self, x):

        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(),torch.ones(1)):
                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True
                self.conv3.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.bn3(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                self.conv3.weight.requires_grad = False

                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            if self.multFlag:
                out = self.prob*out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out


class ResNet_StoDepth_lineardecay(nn.Module):

    def __init__(self, block, prob_0_L, multFlag, layers, num_classes=1000, zero_init_residual=False,norm_layer = 'bn'):
        super(ResNet_StoDepth_lineardecay, self).__init__()
        self.inplanes = 64
        assert norm_layer in ['bn','in']
        if norm_layer=='in':
            NL = nn.InstanceNorm2d
        else:
            NL = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = NL(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.multFlag = multFlag
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0]-prob_0_L[1]
        self.prob_step = self.prob_delta/(sum(layers)-1)

        self.layer1 = self._make_layer(block, 64, layers[0],norm_layer =norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,norm_layer =norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,norm_layer =norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,norm_layer =norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, StoDepth_lineardecayBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, StoDepth_lineardecayBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1,norm_layer ='bn'):
        downsample = None
        if norm_layer=='in':
            NL=nn.InstanceNorm2d
        else:
            NL=nn.BatchNorm2d
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                NL(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes, stride, downsample,norm_layer))
        self.prob_now = self.prob_now - self.prob_step
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes))
            self.prob_now = self.prob_now - self.prob_step

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_StoDepth_lineardecay(pretrained=False, prob_0_L=[1,0.5], multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth_lineardecay-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth_lineardecay(StoDepth_BasicBlock, prob_0_L, multFlag, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def se_resnet18_StoDepth_lineardecay(pretrained=False, prob_0_L=[1,0.5], multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth_lineardecay-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth_lineardecay(StoDepth_SE_BasicBlock, prob_0_L, multFlag, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False)
    return model

def resnet34_StoDepth_lineardecay(pretrained=False, prob_0_L=[1,0.5], multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth_lineardecay-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth_lineardecay(StoDepth_BasicBlock, prob_0_L, multFlag, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def se_resnet34_StoDepth_lineardecay(pretrained=False, prob_0_L=[1,0.5], multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth_lineardecay-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth_lineardecay(StoDepth_SE_BasicBlock, prob_0_L, multFlag, [3, 4, 6, 3], **kwargs)
    # model = ResNet_StoDepth_lineardecay(StoDepth_SE_BasicBlock, prob_0_L, multFlag, [3, 3, 9, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']),strict=False)
    return model

def se_resnet68_StoDepth_lineardecay(pretrained=False, prob_0_L=[1,0.5], multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth_lineardecay-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet_StoDepth_lineardecay(StoDepth_CBAM_BasicBlock, prob_0_L, multFlag, [2, 2, 6, 2], **kwargs)
    model = ResNet_StoDepth_lineardecay(StoDepth_SE_BasicBlock, prob_0_L, multFlag, [4, 4, 12, 4], **kwargs)
    # model = ResNet_StoDepth_lineardecay(StoDepth_SE_BasicBlock, prob_0_L, multFlag, [3, 3, 9, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']),strict=False)
    return model

def se_sp_resnet34_StoDepth_lineardecay(pretrained=False, prob_0_L=[1,0.5], multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth_lineardecay-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth_lineardecay(StoDepth_SE_decouled_BasicBlock, prob_0_L, multFlag, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50_StoDepth_lineardecay(pretrained=False, prob_0_L=[1,0.5], multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth_lineardecay-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth_lineardecay(StoDepth_Bottleneck, prob_0_L, multFlag, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_StoDepth_lineardecay(pretrained=False, prob_0_L=[1,0.5], multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth_lineardecay-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth_lineardecay(StoDepth_Bottleneck, prob_0_L, multFlag, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_StoDepth_lineardecay(pretrained=False, prob_0_L=[1,0.5], multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth_lineardecay-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth_lineardecay(StoDepth_Bottleneck, prob_0_L, multFlag, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model