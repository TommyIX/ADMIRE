'''
CAT.py
修改：王锦宏
------------------------------
CATkernel的三个参数rx, ry, rz：rx和ry影响卷积核的大小（如输入128,128 卷积核就是127*127） 一般来说超过原图大小的卷积核能在图像边缘起到最好的效果
scipy的convolve2d提供了镜像补边（以图像边缘为轴镜像填充 也就是  cba| abcdefg | gfe）
'''


import numpy as np
import torch
from scipy.signal import convolve2d

def CATkernel(rx, ry, rz):
    Rx = int(np.floor(rx * 0.5) - 1)
    Ry = int(np.floor(ry * 0.5) - 1)
    n = Rx + Rx + 1
    m = Ry + Ry + 1
    Mx = np.zeros([n, m])
    My = np.zeros([n, m])

    for i in range(-Ry, Ry + 1):
        for j in range(-Rx, Rx + 1):
            if i==0 and j==0:
                Mx[i + Ry, j + Rx] = 0
                My[i + Ry, j + Rx] = 0
                continue
            Mx[i + Ry, j + Rx] = -j / np.power(np.sqrt(i * i + j * j + rz),3.7)
            My[i + Ry, j + Rx] = -i / np.power(np.sqrt(i * i + j * j + rz),3.7)

    return torch.from_numpy(np.ascontiguousarray(Mx)), torch.from_numpy(np.ascontiguousarray(My))

def conv2(x, y, mode='same'): # 据锦宏说，火炬的卷积是没有对称补边的，只有scipy里的才有。所以先变成np，用scipy的卷积，再变回去。
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return torch.from_numpy(np.ascontiguousarray(np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode, boundary='symm'), 2)))
    # 旋转90°这些细节，锦宏考虑得非常不错。

# def conv2(x, y):
#     可能使用的思路，通过torch的方法进行补边
#     # Pad x,y and convolute x,y using torch
#     x = x.numpy()
#     y = y.numpy()
#     x = np.pad(x, ((y.shape[0] - 1, y.shape[0] - 1), (y.shape[1] - 1, y.shape[1] - 1)), 'symmetric')
#     x = torch.from_numpy(x)
#     y = torch.from_numpy(y)
#     return torch.nn.functional.conv2d(x.unsqueeze(0).unsqueeze(0), y.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
#
#     xr = torch.rot90(x, 2).unsqueeze(0).unsqueeze(0)
#     yr = torch.rot90(y, 2).unsqueeze(0).unsqueeze(0)
#     return torch.rot90(torch.nn.functional.conv2d(xr, yr.to(torch.float32), padding='same'),2)

def ConVEF_model(f, Mx, My):
    fmin = torch.min(f[:, :])
    fmax = torch.max(f[:, :])
    f = (f - fmin) / (fmax - fmin)  # % Normalize f to the range [0,1]

    fx = conv2(f, Mx)
    fy = conv2(f, My)

    if len(fx.shape)==2:
        return [fx, fy]  # if use convolve2d
    else:
        return [fx[0,0,:,:], fy[0,0,:,:]]
