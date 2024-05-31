import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model, modelc, networks
from utils import utils
from PIL import Image
from tqdm import tqdm
import torch
import time
import numpy as np
from torchvision.utils import save_image

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True

    dataset = create_dataset(opt)

    # print(dataset)
    psfr_model = create_model(opt)  # create a model given opt.model and other options
    psfr_model.load_pretrain_models()

    netP = psfr_model.netP
    netG = psfr_model.netG
    psfr_model.eval()


    model = modelc.initModel(name='resnet', num_classes=2, feature_extract=False, use_pretrained=False).to(psfr_model.device )
    model.load_state_dict(torch.load('./check_points/224_x4/C/resnet50_224_99.30.pth'))
    model.eval()

    model1 = modelc.initModel(name='densenet', num_classes=2, feature_extract=False, use_pretrained=False).to(psfr_model.device)
    model1.load_state_dict(torch.load('./check_points/224_x4/C/densenet121_224_99.25.pth'))
    model1.eval()

    model2 = modelc.initModel(name='vgg19', num_classes=2, feature_extract=False,use_pretrained=True).to(psfr_model.device)
    model2.load_state_dict(torch.load('./check_points/224_x4/C/vgg19_bn_99.05.pth'))
    model2.eval()

    model3 = modelc.initModel(name='efficientnet-b0', num_classes=2, feature_extract=False, use_pretrained=False).to(psfr_model.device)
    model3.load_state_dict(torch.load('./check_points/224_x4/C/efficientnet-b0_99.20.pth'))
    model3.eval()

    model4 = modelc.initModel(name='squeezenet', num_classes=2, feature_extract=False, use_pretrained=False).to(psfr_model.device)
    model4.load_state_dict(torch.load('./check_points/224_x4/C/squeezenet_224_99.15.pth'))
    model4.eval()

    model5 = modelc.initModel(name='mobilenet', num_classes=2, feature_extract=False, use_pretrained=False).to(psfr_model.device )
    model5.load_state_dict(torch.load('./check_points/224_x4/C/mobilenet_99.05.pth'))
    model5.eval()


    # privacynetGen = networks.Generator(conv_dim=64, c_dim=4, repeat_num=6).to(opt.device)
    # privacynetGen.load_state_dict(torch.load('E:/LiuJia/Privacynet_D_FM/gender-an/models/1000000-G.ckpt'))

    save_dir = './test/generan_cc/'

    correct_pred, num_examples = 0, 0
    correct_pred1, correct_pred2 = 0, 0
    correct_pred3, correct_pred4, correct_pred5 = 0, 0, 0



    # for i, data in tqdm(enumerate(dataset), total=len(dataset) // opt.batch_size):
    #
    #     inp = data['LR'].to(psfr_model.device)
    #     hr = data['HR'].to(psfr_model.device)
    #     # label = data['label'].squeeze()
    #     label = data['Attr'].to(psfr_model.device)
    #
    #     tr_label = label.clone()
    #     tr_label[:, 0] = 0
    #
    #     # tr_label = 1-label
    #     f1, f2, f3, f4, f5, f6 = False, False, False, False, False, False
    #
    #     with torch.no_grad():
    #         # parse_map, _ = netP(inp)
    #         # parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
    #         # output_SR = netG(inp, parse_map_sm)
    #
    #
    #         # fake = privacynetGen(output_SR, tr_label)
    #         fake = hr
    #         # if (tr_label[0][0].cpu() == 0):
    #         #     fake = privacynetGen(output_SR, tr_label)
    #         #
    #         # else:
    #         #     fake = output_SR
    #
    #         # img_path = data['HR_paths'][0].split('\\')[-1]
    #         # save_path = os.path.join(save_dir, os.path.basename(img_path)[:-3]+'png')
    #         # os.makedirs(os.path.join(save_dir), exist_ok=True)
    #         # img = fake.cpu()
    #         # img = (img + 1) / 2
    #         # save_image(img, save_path)
    #
    #         # output_SR = inp
    #
    #         logits = model(fake)
    #         # logits = model(output_SR)
    #         probas = torch.nn.functional.softmax(logits, dim=1)
    #         _, predicted_labels = torch.max(probas, 1)
    #
    #         correct_pred += (predicted_labels.cpu() == label[0][0].cpu()).sum()
    #         if predicted_labels.cpu() != label[0][0].cpu():
    #             f1 = True
    #
    #
    #
    #         logits = model1(fake)
    #         # logits = model1(output_SR)
    #         probas = torch.nn.functional.softmax(logits, dim=1)
    #         _, predicted_labels = torch.max(probas, 1)
    #
    #         correct_pred1 += (predicted_labels.cpu() == label[0][0].cpu()).sum()
    #
    #         if predicted_labels.cpu() != label[0][0].cpu():
    #             f2 = True
    #
    #             # img_path = data['HR_paths'][0].split('\\')[-1]
    #             # save_path = os.path.join(save_dir, os.path.basename(img_path)[:-3] + 'png')
    #             # os.makedirs(os.path.join(save_dir), exist_ok=True)
    #             # img = output_SR.cpu()
    #             # img = (img + 1) / 2
    #             # save_image(img, save_path)
    #
    #
    #         logits = model2(fake)
    #         # logits = model2(output_SR)
    #         probas = torch.nn.functional.softmax(logits, dim=1)
    #         _, predicted_labels = torch.max(probas, 1)
    #
    #         correct_pred2 += (predicted_labels.cpu() == label[0][0].cpu()).sum()
    #
    #         if predicted_labels.cpu() != label[0][0].cpu():
    #             f3 = True
    #
    #             # img_path = data['HR_paths'][0].split('\\')[-1]
    #             # save_path = os.path.join(save_dir, os.path.basename(img_path)[:-3] + 'png')
    #             # os.makedirs(os.path.join(save_dir), exist_ok=True)
    #             # img = output_SR.cpu()
    #             # img = (img + 1) / 2
    #             # save_image(img, save_path)
    #
    #         logits = model3(fake)
    #         # logits = model3(output_SR)
    #         probas = torch.nn.functional.softmax(logits, dim=1)
    #         _, predicted_labels = torch.max(probas, 1)
    #
    #         correct_pred3 += (predicted_labels.cpu() == label[0][0].cpu()).sum()
    #
    #         if predicted_labels.cpu() != label[0][0].cpu():
    #             f4 = True
    #
    #         logits = model4(fake)
    #         # logits = model4(output_SR)
    #         probas = torch.nn.functional.softmax(logits, dim=1)
    #         _, predicted_labels = torch.max(probas, 1)
    #
    #         correct_pred4 += (predicted_labels.cpu() == label[0][0].cpu()).sum()
    #
    #         if predicted_labels.cpu() != label[0][0].cpu():
    #             f5 = True
    #
    #         logits = model5(fake)
    #         # logits = model5(output_SR)
    #         probas = torch.nn.functional.softmax(logits, dim=1)
    #         _, predicted_labels = torch.max(probas, 1)
    #
    #         correct_pred5 += (predicted_labels.cpu() == label[0][0].cpu()).sum()
    #
    #         if predicted_labels.cpu() != label[0][0].cpu():
    #             f6 = True
    #
    #         num_examples += label.size(0)
    #
    #         # if (f1 and f2 and f3 and f4 and f5 and f6):
    #         #     img_path = data['HR_paths'][0].split('\\')[-1]
    #         #     save_path = os.path.join(save_dir, os.path.basename(img_path)[:-3]+'png')
    #         #     os.makedirs(os.path.join(save_dir), exist_ok=True)
    #         #     img = output_SR.cpu()
    #         #     img = (img + 1) / 2
    #         #     save_image(img, save_path)
    #
    # attr_acc = correct_pred.float() / num_examples * 100
    # attr_acc1 = correct_pred1.float() / num_examples * 100
    # attr_acc2 = correct_pred2.float() / num_examples * 100
    # attr_acc3 = correct_pred3.float() / num_examples * 100
    # attr_acc4 = correct_pred4.float() / num_examples * 100
    # attr_acc5 = correct_pred5.float() / num_examples * 100
    #
    # print(f'Test accuracy: {attr_acc:.2f}%')
    # print(f'Test accuracy: {attr_acc1:.2f}%')
    # print(f'Test accuracy: {attr_acc2:.2f}%')
    # print(f'Test accuracy: {attr_acc3:.2f}%')
    # print(f'Test accuracy: {attr_acc4:.2f}%')
    # print(f'Test accuracy: {attr_acc5:.2f}%')