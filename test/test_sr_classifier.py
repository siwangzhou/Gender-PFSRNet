import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model, ResNet, efficientnet, modelc, networks
from utils import utils
from PIL import Image
from tqdm import tqdm
import torch
import time
import numpy as np
from torchvision.utils import save_image

import io
from urllib.request import urlopen
from alibabacloud_facebody20191230.client import Client
from alibabacloud_facebody20191230.models import RecognizeFaceAdvanceRequest
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions


config = Config(
  # "YOUR_ACCESS_KEY_ID", "YOUR_ACCESS_KEY_SECRET" 的生成请参考https://help.aliyun.com/document_detail/175144.html
  # 如果您是用的子账号AccessKey，还需要为子账号授予权限AliyunVIAPIFullAccess，请参考https://help.aliyun.com/document_detail/145025.html
  # 您的 AccessKey ID
  access_key_id='LTAI5tBJ75DFSf8kw8RoPmjK',
  # 您的 AccessKey Secret
  access_key_secret='VltPHSLwaAKSiEHCBA8ztCfiChGGoq',
  # 访问的域名
  endpoint='facebody.cn-shanghai.aliyuncs.com',
  # 访问的域名对应的region
  region_id='cn-shanghai'
)


def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

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

    # model = modelc.initModel(name='efficientnet-b0', use_pretrained=False).to(opt.device)
    # model.load_state_dict(torch.load('./check_points/224_x4/C/efficientnet-b0_99.20.pth'))
    # # # model.load_state_dict(torch.load('./check_points/224_x4/C/efficientnet_smiling_94.30.pth'))
    # # # model.load_state_dict(torch.load(r'E:\LiuJia\AdvFSRN_224\check_points\FSRN_Black_E\0.5\10_4.0_clr\iter_40000_net_C.pth'))
    # model.eval()

    # model = modelc.initModel(name='squeezenet', use_pretrained=False).to(opt.device)
    # model.load_state_dict(torch.load('./check_points/224_x4/C/squeezenet_224_99.15.pth'))
    # model.eval()

    # unet = networks.UNetGeneratorSN(num_in_ch=3, num_out_ch=3).to(opt.device)
    # unet.load_state_dict(torch.load('./check_points/FSRN_AdvDU/latest_net_U.pth'))
    # unet.eval()


    # model = modelc.initModel(name='resnet', num_classes=2, use_pretrained=False).to(opt.device)
    # model.load_state_dict(torch.load('./check_points/224_x4/C/resnet50_224_99.30.pth'))
    # model.eval()

    # model = modelc.initModel(name='densenet', use_pretrained=False).to(opt.device)
    # model.load_state_dict(torch.load('./check_points/224_x4/C/densenet121_224_99.25.pth'))
    # model.eval()

    # model = modelc.initModel(name='vgg19', use_pretrained=False).to(opt.device)
    # model.load_state_dict(torch.load('./check_points/224_x4/C/vgg19_99.25.pth'))
    # model.eval()

    # model = modelc.initModel(name='googlenet', use_pretrained=True).to(opt.device)
    # model.load_state_dict(torch.load('./check_points/224_x4/C/googlenet_99.15.pth'))
    # model.eval()

    # model = modelc.initModel(name='mobilenet', use_pretrained=False).to(opt.device)
    # model.load_state_dict(torch.load('./check_points/224_x4/C/mobilenet_99.05.pth'))
    # model.eval()

    # privacynetGen = networks.Generator(conv_dim=64, c_dim=1, repeat_num=6).to(opt.device)
    # privacynetGen.load_state_dict(torch.load('E:/LiuJia/PrivacyNet/privacyNet_male/models/240000-G.ckpt'))
    # privacynetGen.eval()

    # save_dir = './test/hr_pn_male/'

    # save_dir = './test/Black_E_0.5_10_4_clr/4w/'

    # save_dir = r'E:\LiuJia\AdvFSRN_224\test\privacynet_sr_24w_advMY_all'

    correct_pred, num_examples = 0, 0
    for i, data in tqdm(enumerate(dataset), total=len(dataset) // opt.batch_size):
        inp = data['LR'].to(psfr_model.device)
        hr = data['HR'].to(psfr_model.device)
        real_label = data['Attr'].to(psfr_model.device)
        save_path = data['HR_paths'][0]
        label = real_label.clone()

        # label[:, 0] = (label[:, 0] == 0)
        label[:, 0] = 0

        with torch.no_grad():

            # parse_map, _ = netP(inp)
            # parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
            # output_SR = netG(inp, parse_map_sm)

            # fake = privacynetGen(hr, label)
            fake = hr
            # # if (label[0][0].cpu() == 0):
            # #     fake = privacynetGen(output_SR, label)
            # #
            # # else:
            # #     fake = output_SR
            #
            # img_path = data['HR_paths'][0].split('\\')[-1]
            # save_path = os.path.join(save_dir, os.path.basename(img_path)[:-3] + 'png')
            # os.makedirs(os.path.join(save_dir), exist_ok=True)
            # # img = utils.color_parse_map(parse_map, 224)[0]
            # fake = (fake + 1) / 2
            # fake = fake.clamp_(0, 1)

            # import imageio
            #
            # imageio.imsave(save_path, img)
            # save_image(fake, save_path)

            # img = Image.open(save_path).convert('RGB')
            #
            # img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1)) / 255. * 2 - 1
            # img_tensor = img_tensor.unsqueeze(0).float().to(psfr_model.device)

            # logits = model(fake)
            # # logits = model(output_SR)
            # # probas = torch.nn.functional.sigmoid(logits)
            # probas = torch.nn.functional.softmax(logits, dim=1)
            # _, predicted_labels = torch.max(probas, 1)

            # 场景一：文件在本地
            stream = open(save_path, 'rb')


            # if (predicted_labels.cpu() != label):
            #     img_path = data['HR_paths'][0].split('\\')[-1]
            #     save_path = os.path.join(save_dir, os.path.basename(img_path)[:-3]+'png')
            #     os.makedirs(os.path.join(save_dir), exist_ok=True)
            #     img = output_SR.cpu()
            #     img = (img + 1) / 2
            #     save_image(img, save_path)

            recognize_face_request = RecognizeFaceAdvanceRequest(
                # 文件在本地写法
                image_urlobject=stream,
                # image_urlobject=io.BytesIO(fake),
                age=True,
                gender=True,
                hat=True,
                glass=True,
                beauty=True,
                expression=True,
                mask=True,
                quality=True,
                max_face_number=2
            )
            runtime = RuntimeOptions()

            try:            # 初始化Client
                client = Client(config)
                response = client.recognize_face_advance(recognize_face_request, runtime)
                # 获取整体结果
                predicted_labels = response.body.data.gender_list[0]

                num_examples += label.size(0)
                # print('pre: ', predicted_labels[0])
                # print('real: ', real_label[0][0])
                correct_pred += (predicted_labels == real_label[0].cpu()).sum()
                # if(predicted_labels.cpu() == real_label[0][0].cpu()):
                #     print(predicted_labels)
                #     print(real_label[0][0])

                num_examples += label.size(0)
                # print('pre: ', predicted_labels[0])
                # print('real: ', real_label[0][0])
                correct_pred += (predicted_labels == real_label[0][0].cpu()).sum()
            except Exception as error:
                # 获取整体报错信息
                print(error)
                # 获取单个字段
                print(error.code)
            #     # tips: 可通过error.__dict__查看属性名称

    attr_acc = correct_pred.float() / num_examples * 100

    print(f'Test accuracy: {attr_acc:.2f}%')