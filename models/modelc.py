# -*- coding: utf-8 -*-
# @Time : 2022/10/26 0026 20:32
# @Author : ZeroOne
# @Email : hnuliujia@hnu.cn
# @File : classifier.py

import torch
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torch.nn as nn


def initModel(name = '', num_classes = 2, feature_extract=False, use_pretrained=False):
    model_ft = None
    if name == 'resnet':
        """
        ResNet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_features, num_classes)
    elif name == "vgg19":
        """ 
        VGG19
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    elif name == "mobilenet":
        """ 
        VGG19
        """
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)

    elif name == "alexnet":
        """ 
        AlexNet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    elif name  == 'squeezenet':
        """ 
        SqueezeNet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
    elif name == 'densenet':
        """ 
        DenseNet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)
    elif name == 'efficientnet-b0':
        """ 
        EfficientNet-B0
        """
        model_ft = EfficientNet.from_name(name)  # 加载模型，使用b几的就改为b几
        # state_dict = torch.load('./check_points/Classifier/efficientnet-b0-355c32eb.pth')
        # model_ft.load_state_dict(state_dict)
        # 修改全连接层
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = torch.nn.Linear(num_ftrs, num_classes)
    elif name == 'googlenet':
        """
        GoogleNet
        """
        model_ft = models.googlenet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Classifier(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, c_dim=2, repeat_num=6):
        super(Classifier, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, c_dim, kernel_size=3, stride=2, padding=0, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src.view(out_src.size(0), out_src.size(1))


def classifier(path='', layers=64, repeat_num=6):
    model = Classifier(layers, repeat_num=repeat_num)
    if path:
        state_dic = torch.load(path, map_location=torch.device('cuda'))
        model.load_state_dict(state_dic['state_dict'])
        # model.load_state_dict(state_dic)
    return model



def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

def get_onehotlabel(x, class_nums = 2):
    return torch.eye(class_nums)[x, :]

if __name__ == '__main__':
    # # model = initModel(name='resnet', num_classes=2, use_pretrained=False)
    # model = classifier(path='', layers=64, repeat_num=6)
    # input = torch.randn(4, 3, 224, 224)
    # out = model(input)
    # print(out.shape)

    # Label Smoothening
    # real_labels = 0.7 + 0.3 * torch.rand(10, device='cuda:0')
    real_labels = torch.ones(10, device='cuda:0')
    # fake_labels = 0.3 * torch.rand(10, device='cuda:0')
    real_labels = get_onehotlabel(real_labels)
    print(real_labels)