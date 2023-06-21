import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

def gaussian_filter(shape, sigma):
    x, y = [int(np.floor(edge / 2)) for edge in shape]
    grid = np.array([[((i ** 2 + j ** 2) / (2.0 * sigma ** 2)) for i in range(-x, x + 1)] for j in range(-y, y + 1)])
    filt = np.exp(-grid) / (2 * np.pi * sigma ** 2)
    filt /= np.sum(filt)
    var = np.zeros((1, 1, shape[0], shape[1]))
    var[0, 0, :, :] = filt
    return torch.tensor(np.float32(var))

class BNLayerStorage:
    def __init__(self):
        self.enc1norm1 = None
        self.enc1norm2 = None
        self.enc2norm1 = None
        self.enc2norm2 = None
        self.enc3norm1 = None
        self.enc3norm2 = None
        self.enc4norm1 = None
        self.enc4norm2 = None
        self.bottlenecknorm1 = None
        self.bottlenecknorm2 = None
        self.dec4norm1 = None
        self.dec4norm2 = None
        self.dec3norm1 = None
        self.dec3norm2 = None
        self.dec2norm1 = None
        self.dec2norm2 = None
        self.dec1norm1 = None
        self.dec1norm2 = None

    def bnlayer_init(self, init_features=32):
        self.enc1norm1 = nn.BatchNorm2d(num_features = init_features)
        self.enc1norm2 = nn.BatchNorm2d(num_features = init_features)
        self.enc2norm1 = nn.BatchNorm2d(num_features = init_features * 2)
        self.enc2norm2 = nn.BatchNorm2d(num_features = init_features * 2)
        self.enc3norm1 = nn.BatchNorm2d(num_features = init_features * 4)
        self.enc3norm2 = nn.BatchNorm2d(num_features = init_features * 4)
        self.enc4norm1 = nn.BatchNorm2d(num_features = init_features * 8)
        self.enc4norm2 = nn.BatchNorm2d(num_features = init_features * 8)
        self.bottlenecknorm1 = nn.BatchNorm2d(num_features = init_features * 16)
        self.bottlenecknorm2 = nn.BatchNorm2d(num_features = init_features * 16)
        self.dec4norm1 = nn.BatchNorm2d(num_features = init_features * 8)
        self.dec4norm2 = nn.BatchNorm2d(num_features = init_features * 8)
        self.dec3norm1 = nn.BatchNorm2d(num_features = init_features * 4)
        self.dec3norm2 = nn.BatchNorm2d(num_features = init_features * 4)
        self.dec2norm1 = nn.BatchNorm2d(num_features = init_features * 2)
        self.dec2norm2 = nn.BatchNorm2d(num_features = init_features * 2)
        self.dec1norm1 = nn.BatchNorm2d(num_features = init_features)
        self.dec1norm2 = nn.BatchNorm2d(num_features = init_features)

    def pack_as_list(self):
        return [self.enc1norm1, self.enc1norm2, self.enc2norm1, self.enc2norm2, self.enc3norm1, self.enc3norm2, self.enc4norm1, self.enc4norm2,
                self.bottlenecknorm1, self.bottlenecknorm2,
                self.dec4norm1, self.dec4norm2, self.dec3norm1, self.dec3norm2, self.dec2norm1, self.dec2norm2, self.dec1norm1, self.dec1norm2]

    def list_to_storage(self,BN):
        self.enc1norm1 = BN[0]
        self.enc1norm2 = BN[1]
        self.enc2norm1 = BN[2]
        self.enc2norm2 = BN[3]
        self.enc3norm1 = BN[4]
        self.enc3norm2 = BN[5]
        self.enc4norm1 = BN[6]
        self.enc4norm2 = BN[7]
        self.bottlenecknorm1 = BN[8]
        self.bottlenecknorm2 = BN[9]
        self.dec4norm1 = BN[10]
        self.dec4norm2 = BN[11]
        self.dec3norm1 = BN[12]
        self.dec3norm2 = BN[13]
        self.dec2norm1 = BN[14]
        self.dec2norm2 = BN[15]
        self.dec1norm1 = BN[16]
        self.dec1norm2 = BN[17]


class UNet_DSBN(nn.Module):
    def __init__(self, in_channels=3, init_features=32):
        super(UNet_DSBN, self).__init__()

        self.dataset_bnstorage = []
        for i in range(3):  # index为3的为测试用site
            nowbnstorage = BNLayerStorage()
            if i>0: nowbnstorage.bnlayer_init(init_features)
            self.dataset_bnstorage.append(nowbnstorage)

        self.current_site_no = 0

        features = init_features
        self.encoder1 = UNet_DSBN._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet_DSBN._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet_DSBN._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet_DSBN._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet_DSBN._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet_DSBN._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_DSBN._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_DSBN._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_DSBN._block(features * 2, features, name="dec1")

        self.fcEp = torch.nn.Conv2d(features, round(features/2), kernel_size=3, stride=1, padding=1)
        self.fcE = torch.nn.Conv2d(round(features/2), 1, 1, padding_mode='reflect', padding=0)
        self.gaussE = torch.nn.Conv2d(1, 1, (9, 9), bias=False, padding_mode='reflect', padding=4)
        self.gaussE.weight = torch.nn.Parameter(gaussian_filter((9, 9), 2), requires_grad=False)

        self.fcAp = torch.nn.Conv2d(features, round(features/2), kernel_size=3, stride=1, padding=1)
        self.fcA = torch.nn.Conv2d(round(features/2), 1, 1, padding_mode='reflect', padding=0)
        self.fcBp = torch.nn.Conv2d(features, round(features/2), kernel_size=3, stride=1, padding=1)
        self.fcB = torch.nn.Conv2d(round(features/2), 1, 1, padding_mode='reflect', padding=0)


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

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

    def getall_BN(self):
        return [self.encoder1.enc1norm1, self.encoder1.enc1norm2, self.encoder2.enc2norm1, self.encoder2.enc2norm2, self.encoder3.enc3norm1, self.encoder3.enc3norm2, self.encoder4.enc4norm1, self.encoder4.enc4norm2, \
            self.bottleneck.bottlenecknorm1, self.bottleneck.bottlenecknorm2, \
            self.decoder4.dec4norm1, self.decoder4.dec4norm2, self.decoder3.dec3norm1, self.decoder3.dec3norm2, self.decoder2.dec2norm1, self.decoder2.dec2norm2, self.decoder1.dec1norm1, self.decoder1.dec1norm2]

    def setall_BN(self, BN):
        self.encoder1.enc1norm1 = BN[0]
        self.encoder1.enc1norm2 = BN[1]
        self.encoder2.enc2norm1 = BN[2]
        self.encoder2.enc2norm2 = BN[3]
        self.encoder3.enc3norm1 = BN[4]
        self.encoder3.enc3norm2 = BN[5]
        self.encoder4.enc4norm1 = BN[6]
        self.encoder4.enc4norm2 = BN[7]
        self.bottleneck.bottlenecknorm1 = BN[8]
        self.bottleneck.bottlenecknorm2 = BN[9]
        self.decoder4.dec4norm1 = BN[10]
        self.decoder4.dec4norm2 = BN[11]
        self.decoder3.dec3norm1 = BN[12]
        self.decoder3.dec3norm2 = BN[13]
        self.decoder2.dec2norm1 = BN[14]
        self.decoder2.dec2norm2 = BN[15]
        self.decoder1.dec1norm1 = BN[16]
        self.decoder1.dec1norm2 = BN[17]

    def BNShift(self, dataset_no):
        current_site_BNlist = self.getall_BN()
        self.dataset_bnstorage[self.current_site_no].list_to_storage(current_site_BNlist)

        self.current_site_no = dataset_no
        self.setall_BN(self.dataset_bnstorage[self.current_site_no].pack_as_list())

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
