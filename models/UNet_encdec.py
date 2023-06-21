import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from copy import deepcopy

def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

def get_r_adv_t(mp, mp1, mp2, decoder1, decoder2, it=1, xi=1e-1, eps=10.0):
    # stop bn
    decoder1.eval()
    decoder2.eval()

    x = mp[4].clone()
    x_mp1 = deepcopy(mp1)
    x_mp1[4] = mp[4].clone()
    x_mp2 = deepcopy(mp2)
    x_mp2[4] = mp[4].clone()

    x_mp1_detached = []
    x_mp2_detached = []
    for elements in x_mp1:
        x_mp1_detached.append(elements.detach())
    for elements in x_mp2:
        x_mp2_detached.append(elements.detach())

    with torch.no_grad():
        # get the ensemble results from teacher
        decoder1_result = decoder1(x_mp1_detached)
        decoder2_result = decoder2(x_mp2_detached)
        pred = F.softmax(sum(decoder1_result) + sum(decoder2_result) / 6, dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    # assist students to find the effective va-noise
    for _ in range(it):
        d.requires_grad_()
        x_mp1_input = x_mp1_detached
        x_mp1_input[4] += xi *d
        x_mp2_input = x_mp2_detached
        x_mp2_input[4] += xi *d
        pred_hat = (sum(decoder1(x_mp1_input)) + sum(decoder2(x_mp2_input))) / 6
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder1.zero_grad()
        decoder2.zero_grad()

    r_adv = d * eps

    # reopen bn, but freeze other params.
    # https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/16
    decoder1.train()
    decoder2.train()
    return r_adv

def gaussian_filter(shape, sigma):
    x, y = [int(np.floor(edge / 2)) for edge in shape]
    grid = np.array([[((i ** 2 + j ** 2) / (2.0 * sigma ** 2)) for i in range(-x, x + 1)] for j in range(-y, y + 1)])
    filt = np.exp(-grid) / (2 * np.pi * sigma ** 2)
    filt /= np.sum(filt)
    var = np.zeros((1, 1, shape[0], shape[1]))
    var[0, 0, :, :] = filt
    return torch.tensor(np.float32(var))

class UNet_base(nn.Module):
    def __init__(self):
        super(UNet_base, self).__init__()

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class UNet_encoder(UNet_base):
    def __init__(self, in_channels=3, init_features = 32):
        super(UNet_encoder, self).__init__()

        features = init_features
        self.encoder1 = UNet_encoder._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet_encoder._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet_encoder._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet_encoder._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet_encoder._block(features * 8, features * 16, name="bottleneck")

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        return [enc1, enc2, enc3, enc4, bottleneck]

class UNet_decoder(UNet_base):
    def __init__(self, init_features = 32):
        super(UNet_decoder, self).__init__()

        features = init_features

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet_decoder._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_decoder._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_decoder._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_decoder._block(features * 2, features, name="dec1")

        self.fcEp = torch.nn.Conv2d(features, round(features/2), kernel_size=3, stride=1, padding=1)
        self.fcE = torch.nn.Conv2d(round(features/2), 1, 1, padding_mode='reflect', padding=0)
        self.gaussE = torch.nn.Conv2d(1, 1, (9, 9), bias=False, padding_mode='reflect', padding=4)
        self.gaussE.weight = torch.nn.Parameter(gaussian_filter((9, 9), 2), requires_grad=False)

        self.fcAp = torch.nn.Conv2d(features, round(features/2), kernel_size=3, stride=1, padding=1)
        self.fcA = torch.nn.Conv2d(round(features/2), 1, 1, padding_mode='reflect', padding=0)
        self.fcBp = torch.nn.Conv2d(features, round(features/2), kernel_size=3, stride=1, padding=1)
        self.fcB = torch.nn.Conv2d(round(features/2), 1, 1, padding_mode='reflect', padding=0)

    def forward(self, x, vat_models=None, vat_datas=None):
        enc1, enc2, enc3, enc4, bottleneck = x
        if vat_models is not None:
            ema1_midparams_labeled, ema2_midparams_labeled = vat_datas
            ema_teacher_model1_decoder, ema_teacher_model2_decoder = vat_models
            r_adv = get_r_adv_t(x, ema1_midparams_labeled, ema2_midparams_labeled, ema_teacher_model1_decoder,
                                ema_teacher_model2_decoder, it=1, xi=1e-6, eps=2.0)
            bottleneck = bottleneck + r_adv  # 我囸你的妈，bottleneck += r_adv 报错，这样写反而可以，逆天

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # 通道预测
        h_fcEp = self.fcEp(dec1)
        h_fcE = self.fcE(h_fcEp)
        filteredE = torch.nn.functional.conv2d(h_fcE, self.gaussE.weight, padding=4)

        h_fcAp = self.fcAp(dec1)
        h_fcA = self.fcA(h_fcAp)
        h_fcBp = self.fcBp(dec1)
        h_fcB = self.fcB(h_fcBp)

        return filteredE, h_fcA, h_fcB