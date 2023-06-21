'''
snake_initialization.py

王一康编写
通过成型的mapE对轮廓进行更加精准的初始化，有效提高模型的性能
'''
import torch
import numpy as np

def initialize_snake(shape, imglen, scale, L, adaptive=False, Emap=None, device='cpu'):
    """
    Initialize the snake contour.
    Args:
        shape (str): 'circle' or 'square'
        imglen (int): the length of the image
        scale (float): the scale of the imglen
        L (int): the number of points on the snake contour
        adaptive (bool): whether to initialize the snake contour adaptively
        Emap (torch.Tensor): the energy map
    Returns:
        init_snake (torch.Tensor): the initialized snake contour
    """

    assert shape in ['circle', 'square'], 'shape must be circle or square'
    if adaptive and Emap is None:
        raise ValueError('Emap must be provided when adaptive is True')

    if shape == 'circle':
        s = torch.linspace(0, 2 * np.pi, L)
        # 确定初始化中心
        if adaptive:
            init_mid_u, init_mid_v = torch.where(Emap < torch.mean(Emap) * 0.5)
            init_mid_u = torch.mean(init_mid_u.float().to('cpu'))
            init_mid_v = torch.mean(init_mid_v.float().to('cpu'))
        else:
            init_mid_u = imglen // 2
            init_mid_v = imglen // 2

        # 初始化为圆
        init_u = init_mid_u + scale * imglen * 0.5 * torch.cos(s)
        init_v = init_mid_v + scale * imglen * 0.5 * torch.sin(s)

        # 处理边界问题
        if adaptive:
            # 防止边界问题，对初始位置进行限制
            # 这里是把超出部分限制到图像边缘
            # init_u[init_u > imglen-1] = imglen-1
            # init_v[init_v > imglen-1] = imglen-1
            # init_u[init_u < 0] = 0
            # init_v[init_v < 0] = 0
            # 这里是担心轮廓在图像边缘会不会表现不好，所以往回缩5个像素
            # NOTE 还不确定这个改动是否有效果，待后续验证
            init_u[init_u > imglen-6] = imglen-6
            init_v[init_v > imglen-6] = imglen-6
            init_u[init_u < 5] = 5
            init_v[init_v < 5] = 5

        init_u = init_u.reshape([L, 1])
        init_v = init_v.reshape([L, 1])
        init_snake = torch.zeros([L, 2])
        init_snake[:, 0] = init_u.squeeze()
        init_snake[:, 1] = init_v.squeeze()
        init_snake = init_snake.to(torch.float32).to(device)

    elif shape == 'square':
        # 确定初始化中心
        if adaptive:
            init_mid_u, init_mid_v = torch.where(Emap < torch.mean(Emap) * 0.5)
            init_mid_u = torch.mean(init_mid_u.float().to('cpu'))
            init_mid_v = torch.mean(init_mid_v.float().to('cpu'))
        else:
            init_mid_u = imglen // 2
            init_mid_v = imglen // 2

        # 初始化为正方形
        L_min = (init_mid_u - scale * imglen * 0.5).item()
        L_max = (init_mid_u + scale * imglen * 0.5).item()
        init_u = np.hstack((np.ones(L // 4) * L_min, np.linspace(L_min, L_max, L // 4 + 1)[:-1], np.ones(L // 4) * L_max, np.linspace(L_max, L_min, L // 4 + 1)[:-1]))
        init_v = np.hstack((np.linspace(L_min, L_max, L // 4 + 1)[:-1], np.ones(L // 4) * L_max, np.linspace(L_max, L_min, L // 4 + 1)[:-1], np.ones(L // 4) * L_min))

        # 处理边界问题
        if adaptive:
            init_u[init_u > imglen-1] = imglen-1
            init_v[init_v > imglen-1] = imglen-1
            init_u[init_u < 0] = 0
            init_v[init_v < 0] = 0

        init_u = init_u.reshape([L, 1])
        init_v = init_v.reshape([L, 1])
        init_snake = torch.zeros([L, 2])
        init_snake[:, 0] = torch.from_numpy(init_u).squeeze()
        init_snake[:, 1] = torch.from_numpy(init_v).squeeze()
        init_snake = init_snake.to(torch.float32).to(device)

    return init_snake
