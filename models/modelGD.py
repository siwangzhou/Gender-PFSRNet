import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import transforms


def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to('cuda')
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)


def mytest(N1, N2, loader):
    data_iter = iter(loader)
    model = N1
    G = N2
    G.eval()
    correct = 0
    total = 50
    sig = nn.Sigmoid()
    for i in range(50):
        x, y = next(data_iter)
        x, y = x.to('cuda:0'), y.to('cuda:0')
        with torch.no_grad():
            x1 = G(x)
            pre = model(x1)
            pre = sig(pre)
        # pre[pre > 0.5] = 1
        # pre[pre <= 0.5] = 0
        correct += pre.cpu().detach().numpy()[0][0]
    G.train()
    return float(correct) / total


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.

        return self.main(x)


def generator(path='', layers=64):
    model = Generator(layers)
    if path:
        state_dic = torch.load(path, map_location=torch.device('cuda'))
        model.load_state_dict(state_dic['state_dict'])
        # model.load_state_dict(state_dic)
    return model


class Discriminator_STGAN(nn.Module):
    def __init__(self, image_size=224, conv_dim=64, fc_dim=1024, n_layers=6):
        super(Discriminator_STGAN, self).__init__()
        layers = []
        in_channels = 3
        for i in range(n_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_dim * 2 ** i, 4, 2, 1),
                nn.InstanceNorm2d(conv_dim * 2 ** i, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = conv_dim * 2 ** i
        self.conv = nn.Sequential(*layers)
        feature_size = image_size // 2 ** n_layers
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_dim * 2 ** (n_layers - 1) * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_adv = self.fc_adv(y)
        return logit_adv


def discriminator_stgan(path='', layers=64, repeat_num=5):
    model = Discriminator_STGAN(conv_dim=layers, n_layers=repeat_num)
    if path:
        state_dic = torch.load(path, map_location=torch.device('cuda'))
        model.load_state_dict(state_dic['state_dict'])
    return model


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=2, padding=0, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src.view(out_src.size(0), out_src.size(1))


def discriminator(path='', layers=64, repeat_num=6):
    model = Discriminator(layers, repeat_num=repeat_num)
    if path:
        state_dic = torch.load(path, map_location=torch.device('cuda'))
        model.load_state_dict(state_dic['state_dict'])
        # model.load_state_dict(state_dic)
    return model



if __name__ == '__main__' :
    model = generator()
    input = torch.randn(4, 3, 224, 224)
    out = model(input)
    print(out.shape)