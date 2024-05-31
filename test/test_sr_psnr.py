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
from image_quality_assessment import PSNR, SSIM

from torchvision.utils import save_image
from utils.ssim_psnr import get_ssim, get_psnr

def getpsnr(xr, xf):
    xr = xr.cpu().detach().numpy()
    xf = xf.cpu().detach().numpy()
    mse = np.mean((xr/1.0-xf/1.0)**2)
    if mse < 1.0e-10:
        print('1')
        psnr = 100
    else:
        psnr = 10 * np.log10(2**2/mse)
    return psnr


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

    psnr = PSNR(1, only_test_y_channel=True)
    ssim = SSIM(1,True)

    psnr = psnr.to(device=psfr_model.device, non_blocking=True)
    ssim = ssim.to(device=psfr_model.device, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0

    # from models import networks
    # privacynetGen = networks.Generator(conv_dim=64, c_dim=2, repeat_num=6).to(opt.device)
    # privacynetGen.load_state_dict(torch.load('E:/LiuJia/PrivacyNet/privacyNet_male_young/models/240000-G.ckpt'))

    # save_dir = 'E:/LiuJia/CelebA_HQ/Valid/CelebA-HQ-ga/'
    # save_dir = './test/privacynet_c/'
    save_dir = './test/Black_E_0.5_10_4_clr/3.5W/'
    # save_dir = './test/sr_c/'
    # save_dir = './test/ga_c/'
    total_files = len(dataset)
    for i, data in tqdm(enumerate(dataset), total=len(dataset) // opt.batch_size):
        inp = data['LR'].to(psfr_model.device)
        hr = data['HR'].to(psfr_model.device)
        label = data['Attr'].to(psfr_model.device)

        # label[:, 0] = (label[:, 0] == 0)

        with torch.no_grad():
            # parse_map, pre_sr = netP(inp)
            # parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
            # output_SR = netG(inp, parse_map_sm)

            # if (label[0][0].cpu() == 0):
            #     fake = privacynetGen(output_SR, label)
            #
            # else:
            #     fake = output_SR

            img_path = data['HR_paths'][0].split('\\')[-1]
            save_path = os.path.join(save_dir, os.path.basename(img_path)[:-3] + 'png')

            img = Image.open(save_path).convert('RGB')

            img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1)) / 255. * 2 - 1
            img_tensor = img_tensor.unsqueeze(0).float().to(psfr_model.device)

            # if (label[0].cpu() == 1):
            #     img_path = data['HR_paths'][0].split('\\')[-1]
            #     save_path = os.path.join(save_dir, os.path.basename(img_path)[:-3] + 'png')
            #     os.makedirs(os.path.join(save_dir), exist_ok=True)
            #     img = output_SR.cpu()
            #     img = (img + 1) / 2
            #     save_image(img, save_path)

            # psnr_metrics += psnr(output_SR, hr).item()
            # ssim_metrics += ssim(output_SR, hr).item()
            psnr_metrics += getpsnr(img_tensor, hr)
            ssim_metrics += get_ssim(img_tensor, hr)

            # psnr_metrics += psnr(img_tensor, hr).item()
            # ssim_metrics += ssim(img_tensor, hr).item()

    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files

    print(f"PSNR: {avg_psnr:4.2f} [dB]\n"
          f"SSIM: {avg_ssim:4.4f} [u]")

