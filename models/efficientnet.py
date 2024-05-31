from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch


def efficientnet(name, num_classes = 2):
    base_model = EfficientNet.from_name(name)  # 加载模型，使用b几的就改为b几
    state_dict = torch.load('./check_points/Classifier/efficientnet-b0-355c32eb.pth')
    base_model.load_state_dict(state_dict)
    # 修改全连接层
    num_ftrs = base_model._fc.in_features
    base_model._fc = nn.Linear(num_ftrs, num_classes)
    model = base_model

    return model



if __name__ == '__main__':
    model = efficientnet(name='efficientnet-b0', num_classes=2)

    input = torch.randn(32, 3, 256, 256)
    out = model(input)
    print(out.shape)
