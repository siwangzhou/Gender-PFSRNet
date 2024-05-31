import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model, ResNet
from utils import utils
from PIL import Image
from tqdm import tqdm
import torch
import time
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 4  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True

    dataset = create_dataset(opt)

    # print(dataset)
    psfr_model = create_model(opt)  # create a model given opt.model and other options
    psfr_model.load_pretrain_models()

    netP = psfr_model.netP
    netG = psfr_model.netG
    psfr_model.eval()

    resnet = ResNet.resnet18(num_classes=2).to(psfr_model.device)
    resnet.load_state_dict(torch.load('./check_points/Classifier/ResNet18_best.pth'))
    resnet.eval()

    correct_pred, num_examples = 0, 0
    for i, data in tqdm(enumerate(dataset), total=len(dataset) // opt.batch_size):
        inp = data['LR']
        hr = data['HR']
        label = data['Attr'].squeeze()

        with torch.no_grad():
            parse_map, _ = netP(inp)
            parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
            output_SR = netG(inp, parse_map_sm)

            logits, probas = resnet(output_SR)
            _, predicted_labels = torch.max(probas, 1)

            num_examples += label.size(0)
            correct_pred += (predicted_labels.cpu() == label).sum()

    attr_acc = correct_pred.float() / num_examples * 100

    print(f'Test accuracy: {attr_acc:.2f}%')