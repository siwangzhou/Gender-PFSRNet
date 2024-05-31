import os
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from models import loss
from models import networks
from .base_model import BaseModel
from utils import utils
from models import modelc, modelGD
from torchvision.transforms import transforms as T
from random import choices


class EnhanceModel(BaseModel):

    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--parse_net_weight', type=str, default='./check_points/224_x4/FPN/latest_net_P.pth',
                                help='parse model path')
            # parser.add_argument('--classisier_net_weight', type=str, default='./check_points/Classifier/resnet50.pth',
            #                     help='attr classifier model path')
            parser.add_argument('--preG_net_weight', type=str, default='./check_points/FSRN_CelebA/latest_net_G.pth',
                                help='pretrained G model path')
            parser.add_argument('--preD_net_weight', type=str, default='./check_points/FSRN_CelebA/latest_net_D.pth',
                                help='pretrained D model path')
            parser.add_argument('--lambda_pix', type=float, default=0.0, help='weight for parsing map')
            parser.add_argument('--lambda_pcp', type=float, default=10.0, help='weight for vgg perceptual loss')
            parser.add_argument('--lambda_fm', type=float, default=10.0, help='weight for sr')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for sr')
            parser.add_argument('--lambda_ss', type=float, default=1000., help='weight for global style')
            parser.add_argument('--lambda_attr', type=float, default=4.0, help='weight for attr loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.netP = networks.define_P(opt, weight_path=opt.parse_net_weight)
        self.netG = networks.define_G(opt, use_norm='spectral_norm')



        if self.isTrain:
            self.netD = networks.define_D(opt, opt.Dinput_nc, use_norm='spectral_norm')
            self.vgg_model = loss.PCPFeat(weight_path='./pretrain_models/vgg19-dcbb9e9d.pth').to(opt.device)
            # self.attr_cl = classifier.classifier(path=None, layers=64, repeat_num=6).to(opt.device)
            # self.netC1 = modelc.initModel(name='densenet', use_pretrained=False).to(opt.device)
            # self.netC1.load_state_dict(torch.load('./check_points/224_x4/C/densenet121_224_99.25.pth'))
            # self.netC1.eval()
            # # self.netC2 = modelc.initModel(name='squeezenet', use_pretrained=False).to(opt.device)
            # # self.netC2.load_state_dict(torch.load('./check_points/224_x4/C/squeezenet_224_99.15.pth'))
            # # self.netC2.eval()
            # self.netC2 = modelc.initModel(name='densenet', use_pretrained=False).to(opt.device)
            # self.netC2.load_state_dict(torch.load('./check_points/224_x4/C/densenet121_224_99.25.pth'))
            # self.netC2.eval()

            # self.netU = networks.UNetGeneratorSN(num_in_ch=3, num_out_ch=3).to(opt.device)
            # self.netU = modelGD.generator().to(opt.device)

            # self.netC = modelc.classifier().to(opt.device)
            self.netC = modelc.initModel(name='efficientnet-b0', use_pretrained=False).to(opt.device)
            self.netC.load_state_dict(torch.load('./check_points/224_x4/C/efficientnet-b0_99.20.pth'))
            self.netC.eval()
            # self.netC = modelc.initModel(name='vgg19', num_classes=2, feature_extract=False, use_pretrained=False).to(opt.device)
            # self.netC.load_state_dict(torch.load('./check_points/224_x4/C/vgg19_99.25.pth'))


            # self.Fm = vggface.VGGFace().to(opt.device)
            # self.Fm.load_state_dict(torch.load('./check_points/Classifier/vggface.pth'))
            # self.Fm.eval()
            if len(opt.gpu_ids) > 0:
                self.vgg_model = torch.nn.DataParallel(self.vgg_model, opt.gpu_ids, output_device=opt.device)

                # self.netU = torch.nn.DataParallel(self.netU, opt.gpu_ids, output_device=opt.device)
                self.netC = torch.nn.DataParallel(self.netC, opt.gpu_ids, output_device=opt.device)
                """ 读取预训练模型 """
                self.netG.module.load_state_dict(torch.load(self.opt.preG_net_weight), strict=False)
                self.netD.module.load_state_dict(torch.load(self.opt.preD_net_weight), strict=False)

        self.model_names = ['G']
        # self.loss_names = ['Pix', 'PCP', 'G', 'FM', 'D', 'SS', 'Attr', 'C'] # Generator loss, fm loss, parsing loss, discriminator loss, attr loss, realattr loss
        self.loss_names = ['Pix', 'PCP', 'G', 'FM', 'D', 'SS', 'Attr']
        self.visual_names = ['img_LR', 'img_HR', 'img_SR', 'ref_Parse', 'hr_mask']
        self.fm_weights = [1 ** x for x in range(opt.D_num)]

        if self.isTrain:

            self.g_lr = opt.g_lr
            self.d_lr = opt.d_lr
            self.c_lr = opt.c_lr
            # self.g_lr = 1e-4
            # self.d_lr = 4e-4
            # self.u_lr = 2e-4
            self.total_steps = opt.total_epochs * (28000 // opt.batch_size)
            # self.total_steps = 15 * (28000 // opt.batch_size)

            self.model_names = ['G', 'D', 'C']
            self.load_model_names = ['G', 'D', 'C']
            # self.model_names = ['G', 'D']
            # self.load_model_names = ['G', 'D']

            self.criterionParse = torch.nn.CrossEntropyLoss().to(opt.device)
            self.criterionFM = loss.FMLoss().to(opt.device)
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(opt.device)
            self.criterionPCP = loss.PCPLoss(opt)
            self.criterionPix = nn.L1Loss()
            # self.criterionPix = nn.MSELoss()
            self.criterionRS = loss.RegionStyleLoss()
            self.criterionAttr = torch.nn.CrossEntropyLoss().to(opt.device)
            # self.criterionAttr = torch.nn.BCEWithLogitsLoss().to(opt.device)
            # self.criterionSSIM = SSIM().to(opt.device)

            for name, param in self.netG.named_parameters():
                if param.requires_grad:
                    print('updata ', name)

            self.optimizer_G = optim.Adam([p for p in self.netG.parameters() if p.requires_grad], lr=self.g_lr,
                                          betas=(opt.beta1, 0.999))
            self.optimizer_D = optim.Adam([p for p in self.netD.parameters() if p.requires_grad], lr=self.d_lr,
                                          betas=(opt.beta1, 0.999))
            # self.optimizer_U = optim.Adam([p for p in self.netU.parameters() if p.requires_grad], lr=self.u_lr, betas=(opt.beta1, 0.999))
            self.optimizer_C = optim.Adam([p for p in self.netC.parameters() if p.requires_grad], lr=self.c_lr,
                                          betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D, self.optimizer_C]
            # self.optimizers = [self.optimizer_G, self.optimizer_D]

            self.sched_G = optim.lr_scheduler.LambdaLR(
                self.optimizer_G, lambda step: 1 - step / self.total_steps)
            self.sched_D = optim.lr_scheduler.LambdaLR(
                self.optimizer_D, lambda step: 1 - step / self.total_steps)
            # self.sched_U = optim.lr_scheduler.LambdaLR(
            #     self.optimizer_U, lambda step: 1 - step / self.total_steps)
            self.sched_C = optim.lr_scheduler.LambdaLR(
                self.optimizer_C, lambda step: 1 - step / self.total_steps)

    def eval(self):
        self.netG.eval()
        self.netP.eval()
        # self.netC.eval()

    def train(self):
        self.netG.train()
        self.netP.train()
        # self.netC.train()

    def updata_lr(self):
        # self.sched_G.step()
        # self.sched_D.step()
        self.sched_C.step()
        pass


    def load_pretrain_models(self, ):
        self.netP.eval()
        print('Loading pretrained LQ face parsing network from', self.opt.parse_net_weight)
        if len(self.opt.gpu_ids) > 0:
            self.netP.module.load_state_dict(torch.load(self.opt.parse_net_weight))
        else:
            self.netP.load_state_dict(torch.load(self.opt.parse_net_weight))
        self.netG.eval()
        print('Loading pretrained PSFRGAN from', self.opt.psfr_net_weight)
        if len(self.opt.gpu_ids) > 0:
            self.netG.module.load_state_dict(torch.load(self.opt.psfr_net_weight), strict=False)
        else:
            self.netG.load_state_dict(torch.load(self.opt.psfr_net_weight), strict=False)

    def set_input(self, input, cur_iters=None):
        self.cur_iters = cur_iters
        self.img_LR = input['LR'].to(self.opt.device)
        self.img_HR = input['HR'].to(self.opt.device)
        self.hr_mask = input['Mask'].to(self.opt.device)
        self.real_attr = input['Attr'].squeeze().to(self.opt.device)
        # self.hr_attr = input['Attr'].squeeze().to(self.opt.device)
        self.hr_attr = torch.zeros_like(self.real_attr).to(self.opt.device)
        # # self.hr_attr = 0.3 * torch.rand(self.real_attr.size(), device='cuda:0')

        if self.opt.debug:
            print('SRNet input shape:', self.img_LR.shape, self.img_HR.shape)


    def forward(self):

        # self.netU.eval()

        with torch.no_grad():
            ref_mask, self.img_pSR = self.netP(self.img_LR)
            self.ref_mask_onehot = (ref_mask == ref_mask.max(dim=1, keepdim=True)[0]).float().detach()

        if self.opt.debug:
            print('SRNet reference mask shape:', self.ref_mask_onehot.shape)
        self.img_SR = self.netG(self.img_LR, self.ref_mask_onehot)
        # self.img_SR = self.netG(self.img_pSR, self.ref_mask_onehot)

        self.real_D_results = self.netD(torch.cat((self.img_HR, self.hr_mask), dim=1), return_feat=True)
        self.fake_D_results = self.netD(torch.cat((self.img_SR.detach(), self.hr_mask), dim=1), return_feat=False)
        self.fake_G_results = self.netD(torch.cat((self.img_SR, self.hr_mask), dim=1), return_feat=True)

        self.img_SR_feats = self.vgg_model(self.img_SR)
        self.img_HR_feats = self.vgg_model(self.img_HR)

        # self.img_SR_unet = self.netU(self.img_SR)

        # self.pre_attr1 = self.netC1(self.img_SR)
        # self.pre_attr2 = self.netC2(self.img_SR_unet)

        self.pre_attr = self.netC(self.img_SR)

        # self.hr_fea = self.Fm(self.nor_vggface(self.img_HR))
        # self.sr_fea = self.Fm(self.nor_vggface(self.img_SR))

        # self.netU.train()

    def backward_G(self):
        # Pix Loss
        self.loss_Pix = self.criterionPix(self.img_SR, self.img_HR) * self.opt.lambda_pix
        # self.loss_Pix = self.criterionPix(self.hr_fea, self.sr_fea) * self.opt.lambda_pix
        # self.loss_Pix = 0.0

        # semantic style loss
        self.loss_SS = self.criterionRS(self.img_SR_feats, self.img_HR_feats, self.hr_mask) * self.opt.lambda_ss

        # perceptual loss
        self.loss_PCP = self.criterionPCP(self.img_SR_feats, self.img_HR_feats) * self.opt.lambda_pcp

        # Feature matching loss
        tmp_loss = 0
        for i, w in zip(range(self.opt.D_num), self.fm_weights):
            tmp_loss = tmp_loss + self.criterionFM(self.fake_G_results[i][1], self.real_D_results[i][1]) * w
        self.loss_FM = tmp_loss * self.opt.lambda_fm / self.opt.D_num

        # Generator loss
        tmp_loss = 0
        for i in range(self.opt.D_num):
            tmp_loss = tmp_loss + self.criterionGAN(self.fake_G_results[i][0], True, for_discriminator=False)
        self.loss_G = tmp_loss * self.opt.lambda_g / self.opt.D_num

        # Attr loss
        # self.loss_Attr1 = self.criterionAttr(self.pre_attr1, self.hr_attr)
        # # self.loss_Attr1 = 0.0
        # self.loss_Attr2 = self.criterionAttr(self.pre_attr2, self.hr_attr) * self.opt.lambda_attr
        # self.loss_Attr = 1.0 * (self.loss_Attr1 + self.loss_Attr2)

        self.loss_Attr  = self.criterionAttr(self.pre_attr, self.hr_attr) * self.opt.lambda_attr
        # self.loss_Attr = 0.0

        # SSIM loss
        # self.loss_SSIM = -self.criterionSSIM(self.img_HR, self.img_SR) * 1.0


        total_loss = self.loss_Pix + self.loss_PCP + self.loss_FM + self.loss_G + self.loss_SS + self.loss_Attr
        total_loss.backward()

    def backward_D(self, ):
        self.loss_D = 0
        for i in range(self.opt.D_num):
            self.loss_D += 0.5 * (
                        self.criterionGAN(self.fake_D_results[i], False) + self.criterionGAN(self.real_D_results[i][0],
                                                                                             True))
        self.loss_D /= self.opt.D_num
        self.loss_D.backward()



    def backward_C(self, ):

        # self.real_attr = 1 - self.hr_attr       # 反转标签，使用真实标签训练分类器

        pre_hr = self.netC(self.img_HR.detach())
        pre_sr = self.netC(self.img_SR.detach())

        # self.loss_C = self.criterionAttr(pre_hr, self.real_attr)
        self.loss_C = 0.5 * self.criterionAttr(pre_hr, self.real_attr) + 0.5 * self.criterionAttr(pre_sr, self.real_attr)
        self.loss_C.backward()
        # self.eval()

    def optimize_parameters_netC(self, ):
        # ---- Update U ------------
        # if (self.cur_iters - 1) % 10 == 0:
        self.optimizer_C.zero_grad()
        self.backward_C()
        self.optimizer_C.step()

    def optimize_parameters(self, ):

        # if (self.cur_iters-1) % 2 == 0:
        # ---- Update G ------------
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # ---- Update D ------------
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # if (self.cur_iters - 1) % 10 == 0:
        #     self.optimizer_C.zero_grad()
        #     self.backward_C()
        #     self.optimizer_C.step()



    def get_current_visuals(self, size=224):
        out = []
        visual_imgs = []
        out.append(utils.tensor_to_numpy(self.img_LR))
        out.append(utils.tensor_to_numpy(self.img_SR))
        # out.append(utils.tensor_to_numpy(self.img_SR_unet))
        out.append(utils.tensor_to_numpy(self.img_HR))

        out_imgs = [utils.batch_numpy_to_image(x, size) for x in out]

        visual_imgs += out_imgs
        visual_imgs.append(utils.color_parse_map(self.ref_mask_onehot, size))
        visual_imgs.append(utils.color_parse_map(self.hr_mask, size))

        return visual_imgs

