"""
作者：Haing
日期：2021-01-03
把金标准掩膜变成蛇上的点，输入image实际上是掩膜，大小为(图像长, 图像宽)

PyTorch适配：王锦宏
2022.12 更新：对掩膜使用Canny算子后容易产生断裂轮廓，因此新版getGT直接对二值mask进行轮廓提取
2022.12 更新2：换用了contour数据集，这个文件因此弃用
"""

import cv2
import torch
import numpy as np
from scipy import interpolate

def getGT(image):  # 输入的image应该是掩膜。
    ret, image1 = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
    # 上句，掩膜变成binary（黑白二值）图。第二个输入80是阈值，低于阈值的都置零，高于阈值的都置第三个输入（255）。

    image1 = image1.astype(np.uint8)
    contours, thresh = cv2.findContours(image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    th = contours[0]  # ★直接通过二值轮廓提取的话，若标注无误只会有一条轮廓
    nnum = len(th)
    n1 = np.zeros(((nnum + 1), 2))
    n1[0:nnum, 0] = th[:, 0, 0]
    n1[0:nnum, 1] = th[:, 0, 1]  # 前面nnum个点。
    n1[nnum, 0] = th[0, 0, 0]
    n1[nnum, 1] = th[0, 0, 1]  # 最后加一个点，让他和第0个点一样。
    # 以上，把那个contours里的点坐标，转化成(L, 2)的形式。
    return np.array(n1)

def getGT_single(img, L):
    contourGT = np.zeros([L, 2])
    # 上句，准备用金标准掩膜生成金标准的蛇位置。初始化为全0的了，L是蛇上的点数，2对应u/v坐标。
    thisGT1 = getGT(np.squeeze(img))  # 从掩膜生成金标准图像。
    [tck, u] = interpolate.splprep([thisGT1[:, 1], thisGT1[:, 0]], s=2, k=1, per=1)  # 为多边形插值做准备。
    [gt_u, gt_v] = interpolate.splev(np.linspace(0, 1, L), tck)  # 插值到L个点。（现在L=200，记得说再加多点数用处不大了？）
    contourGT[:, 0] = gt_u
    contourGT[:, 1] = gt_v

    return contourGT
def getGT_batch(batchim, L, batch_size):  # batchim是一批次的掩膜。
    contourGT = np.zeros([batch_size, L, 2])
    # 上句，准备用金标准掩膜生成金标准的蛇位置。初始化为全0的了，L是蛇上的点数，2对应u/v坐标。
    for i in range(batch_size):
        thisGT1 = getGT(np.squeeze(batchim[i, :, :]))  # 从掩膜生成金标准图像。
        [tck, u] = interpolate.splprep([thisGT1[:, 1], thisGT1[:, 0]], s=2, k=1, per=1)  # 为多边形插值做准备。
        [gt_u, gt_v] = interpolate.splev(np.linspace(0, 1, L), tck)  # 插值到L个点。（现在L=200，记得说再加多点数用处不大了？）
        contourGT[i, :, 0] = gt_u
        contourGT[i, :, 1] = gt_v
    return torch.from_numpy(contourGT)