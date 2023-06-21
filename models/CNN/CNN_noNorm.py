import torch
import numpy as np

def gaussian_filter(shape, sigma):
    x, y = [int(np.floor(edge / 2)) for edge in shape]
    grid = np.array([[((i ** 2 + j ** 2) / (2.0 * sigma ** 2)) for i in range(-x, x + 1)] for j in range(-y, y + 1)])
    filt = np.exp(-grid) / (2 * np.pi * sigma ** 2)
    filt /= np.sum(filt)
    var = np.zeros((1, 1, shape[0], shape[1]))
    var[0, 0, :, :] = filt
    return torch.tensor(np.float32(var))

class CNN_B1(torch.nn.Module):
    def __init__(self):
        super(CNN_B1, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 32, 7, padding_mode='reflect', padding=3)
        self.bn0 = torch.nn.BatchNorm2d(32)

        self.conv1 = torch.nn.Conv2d(32, 64, 5, padding_mode='reflect', padding=2)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding_mode='reflect', padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 128, 3, padding_mode='reflect', padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 256, 3, padding_mode='reflect', padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.conv5 = torch.nn.Conv2d(256, 256, 3, padding_mode='reflect', padding=1)
        self.bn5 = torch.nn.BatchNorm2d(256)

        self.mlpconv1 = torch.nn.Conv2d(832, 256, 1, padding_mode='reflect', padding=0)
        self.bn_mlp1 = torch.nn.BatchNorm2d(256)
        self.mlpconv2 = torch.nn.Conv2d(256, 64, 1, padding_mode='reflect', padding=0)
        self.bn_mlp2 = torch.nn.BatchNorm2d(64)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fcE = torch.nn.Conv2d(64, 1, 1, padding_mode='reflect', padding=0)
        self.gaussE = torch.nn.Conv2d(1, 1, (9, 9), bias=False, padding_mode='reflect', padding=4)
        self.gaussE.weight = torch.nn.Parameter(gaussian_filter((9, 9), 2), requires_grad=False)

        self.fcA = torch.nn.Conv2d(64, 1, 1, padding_mode='reflect', padding=0)
        self.fcB = torch.nn.Conv2d(64, 1, 1, padding_mode='reflect', padding=0)

    def forward(self, x):
        # 输入x size=[bs, channel=3, imx, imy], 此处为[32, 3, 128, 128]
        x = self.bn0(self.relu(self.conv0(x)))

        resized_out = []
        x = self.bn1(self.relu(self.conv1(x))) # size=[bs, 64, 128, 128]
        resized_out.append(x)
        x = self.bn2(self.relu(self.conv2(x)))
        resized_out.append(x)
        x = self.bn3(self.relu(self.conv3(x)))
        resized_out.append(x)
        x = self.bn4(self.relu(self.conv4(x)))
        resized_out.append(x)
        x = self.bn5(self.relu(self.conv5(x)))
        resized_out.append(x)

        h_concat = torch.cat(resized_out, dim=1) # size=[bs, 832, 128, 128]
        hcovd = self.bn_mlp1(self.relu(self.mlpconv1(h_concat)))
        hcovf = self.bn_mlp2(self.relu(self.mlpconv2(hcovd)))

        # 通道预测
        h_fcE = self.fcE(hcovf)
        filteredE = torch.nn.functional.conv2d(h_fcE, self.gaussE.weight, padding=4)

        h_fcA = self.fcA(hcovf)
        h_fcB = self.fcB(hcovf)

        return filteredE, h_fcA, h_fcB
