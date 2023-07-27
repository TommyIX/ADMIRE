import torch
import numpy as np

def active_contour_process(snake, Fu, Fv, mA, mB, mCATu, mCATv, iteration, delta_s, MAP_force_weight, CAT_force_weight, max_pixel_move=1, gamma=2, device='cpu'):
    '''
    这一版本的Active_contour_step彻底去除了气球力相关项
    同时翻译成了PyTorch版本
    '''
    L = snake.shape[0]  # 蛇上点数。
    M = Fu.shape[0]  # 蛇参数图长。
    N = Fu.shape[1]  # 蛇参数图宽。

    u = snake[:,0:1]  # 所有蛇上点横坐标（u）。
    v = snake[:,1:2]  # 所有蛇上点纵坐标（v）。

    snake_hist = []
    snake_hist.append(np.array([u[:, 0].detach().cpu().numpy(), v[:, 0].detach().cpu().numpy()]))
    # 以上snake_hist是历史上的蛇位置。

    du = 0.0  # 速度参量
    dv = 0.0

    for i in range(iteration):
        # 每次蛇迭代步骤的开始（蛇迭代），都把坐标取整
        u = torch.round(u).int()
        v = torch.round(v).int()

        # 以下，计算蛇上各点处的受力情况。
        fu1 = torch.reshape(Fu, [M*N])
        fv1 = torch.reshape(Fv, [M*N])
        fu = torch.gather(fu1, 0, (u*M+v).to(torch.int64).squeeze())
        fv = torch.gather(fv1, 0, (u*M+v).to(torch.int64).squeeze())
        # 上面4句是每个点上的图像能量，导致蛇上每个点在u/v方向受的力。本来Fu/Fv是力场图，现在取出来蛇上点对应的力。

        a1 = torch.reshape(mA, [M*N])
        b1 = torch.reshape(mB, [M*N])
        a = torch.gather(a1, 0, (u*M+v).to(torch.int64).squeeze()).squeeze()
        b = torch.gather(b1, 0, (u*M+v).to(torch.int64).squeeze()).squeeze()
        # 以上4句，mapA（mA）和mapB（mB）在蛇上每一点的值。
        # 总之，以上8句，从输入的各种（指Fu/Fv/alpha/beta）矩阵中，拿出来蛇轮廓位置处的值。比如说，如果蛇上第i个点是(u_i, v_i)，
        #     那就是从这四个矩阵中，取出来(u_i, v_i)点处的值（u_i和v_i是经过插值的，如果不是整数，会四舍五入），
        #         也就是说，输出的四个值都是L*1的，且反应了蛇上每一点的u/v方向外力、alpha值、beta值。
        #     然后那个reshape是把原来[M, N]的矩阵给变成[M*N, 1]的，便于后面去取出蛇上点的值。

        am1 = torch.cat([a[L - 1:L], a[0:L - 1]], dim=0)  # am1是把a的最后一个数放到最前，其他往后移位。
        a0d0 = torch.diag(a)  # a0d0是以向量a中的元素为对角元素的对角矩阵，长度是L*L。
        am1d0 = torch.diag(am1)  # 类似于上面，以向量am1中的元素为对角元素的对角矩阵，长度是L*L。
        a0d1 = torch.cat([a0d0[0:L, L - 1:L], a0d0[0:L, 0:L - 1]], 1)  # a0d0（对角元素为a的矩阵）的最后一列放到第0列，然后其他的往后移位。
        am1dm1 = torch.cat([am1d0[0:L, 1:L], am1d0[0:L, 0:1]], 1)  # am1d0（对角元素为am1的矩阵）第0列放到最后一列，然后其他的往前移位。
        # 以上5句是处理一阶导数项（mA矩阵、即外面的mapA，对应的蛇参数）。移位应该是要在后面通过矩阵相加和相减，构建差分的形式。
        bm1 = torch.cat([b[L - 1:L], b[0:L - 1]], 0)  # bm1是把b的最后一个数放到最前，其他往后移位。
        b1 = torch.cat([b[1:L], b[0:1]], 0)  # b1是把b的第0个数放到最后，其他往前移位。
        b0d0 = torch.diag(b)  # b0d0是以向量b中的元素为对角元素的对角矩阵，长度是L*L。
        bm1d0 = torch.diag(bm1)  # 类似于上面，以向量bm1中的元素为对角元素的对角矩阵，长度是L*L。
        b1d0 = torch.diag(b1)  # 类似于上面，以向量b1中的元素为对角元素的对角矩阵，长度是L*L。
        b0dm1 = torch.cat([b0d0[0:L, 1:L], b0d0[0:L, 0:1]], 1)  # b0d0（对角元素为b的矩阵）的第0列放到最后一列，然后其他的往前移位。
        b0d1 = torch.cat([b0d0[0:L, L - 1:L], b0d0[0:L, 0:L - 1]], 1)  # b0d0（对角元素为b的矩阵）的最后一列放到第0列，然后其他的往后移位。
        bm1dm1 = torch.cat([bm1d0[0:L, 1:L], bm1d0[0:L, 0:1]], 1)  # bm1d0（对角元素为bm1的矩阵）的第0列放到最后一列，然后其他的往前移位。
        b1d1 = torch.cat([b1d0[0:L, L - 1:L], b1d0[0:L, 0:L - 1]], 1)  # b1d0（对角元素为b1的矩阵）的最后一列放到第0列，然后其他的往后移位。
        bm1dm2 = torch.cat([bm1d0[0:L, 2:L], bm1d0[0:L, 0:2]], 1)  # bm1d0（对角元素为bm1的矩阵）的第0/1列放到最后两列，然后其他的往前移2位。
        b1d2 = torch.cat([b1d0[0:L, L - 2:L], b1d0[0:L, 0:L - 2]], 1)  # b1d0（对角元素为b1的矩阵）的最后两列放到第0/1列，然后其他的往后移2位。
        # 以上11句，类似于前面处理一阶导数项的方法，只不过这里是处理二阶导数项。因为现在有移2位的情况，所以移位之后，可以构建二阶差分的形式。

        A = -am1dm1 + (a0d0 + am1d0) - a0d1
        # 上句，014文章中的3式（也是4和11式中的A矩阵），[−α_{s−1}, α_{s−1}+α_s, −α_s]这一项，这儿写成矩阵形式了
        B = bm1dm2 - 2 * (bm1dm1 + b0dm1) + (bm1d0 + 4 * b0d0 + b1d0) - 2 * (b0d1 + b1d1) + b1d2
        # 上句，014文章中的3式（也是文中11式的B矩阵），[βs−1, −2βs − 2βs−1, βs−1 + 4βs + βs+1,− 2βs+1 − 2βs, βs+1]
        #     这一项，这儿写成矩阵形式了。
        # 这个跟001文章中，或者我写的文章里是一样的：
        #     文章a32版2.2节的491行中的“1+A(y_{i-1})+A(y_i)+B(y_{i-1})+4B(y_i)+B{y_{i+1}}”就对应此处A中的(a0d0 + am1d0)、
        #         B中的(bm1d0 + 4 * b0d0 + b1d0)、以及下面加的那个torch.eye(L).to(device)。
        #     文章a32版2.2节的492-493行的“-A(y_{i-1}) - 2B(y_i) - 2B(y_{i-1})”就对应此处A中的-am1dm1和B中的2 * (bm1dm1 + b0dm1)，
        #         然后“-A(y_i) - 2B(y_i) - 2B(y_{i+1})”就对应此处A中的-a0d1和B中的2 * (b0d1 + b1d1)。
        #     这样，后面torch.inverse()里面的东西，相当于是那个M_1矩阵，就是反映内力的矩阵。

        # CAT力(外力接口)
        Cu = torch.gather(torch.reshape(mCATu, [M * N]), 0, (u*M+v).to(torch.int64).squeeze())
        Cv = torch.gather(torch.reshape(mCATv, [M * N]), 0, (u*M+v).to(torch.int64).squeeze())
        # 以上2句，每个点受到的u/v方向的CAT力。

        # 当前速度
        du = -max_pixel_move * torch.tanh((MAP_force_weight * fu - CAT_force_weight * Cu) * gamma * 0.1) * 0.5
        dv = -max_pixel_move * torch.tanh((MAP_force_weight * fv - CAT_force_weight * Cv) * gamma * 0.1) * 0.5
        # 以上2句，计算图像力和CAT力之和，也就是外力之和。

        # 以下，蛇迭代过程，014文章中的12式，或者我们文章中的5式。
        u = torch.matmul(torch.inverse(torch.eye(L).to(device) + 2 * gamma * (A / delta_s + B / (delta_s * delta_s))), (u + gamma * du.unsqueeze(1)).to(torch.float32))
        v = torch.matmul(torch.inverse(torch.eye(L).to(device) + 2 * gamma * (A / delta_s + B / (delta_s * delta_s))), (v + gamma * dv.unsqueeze(1)).to(torch.float32))
        # 以上2句，torch.matmul中第一项用torch.inverse求逆的，对应我们文章5式中的M_1矩阵的逆矩阵，反映内力作用；
        #     第二项含du的，对应的是5式括号里的位移项，反映外力作用。

        u = torch.minimum(u, torch.tensor(M - 1))  # 以下，裁剪与保存。
        v = torch.minimum(v, torch.tensor(N - 1))
        u = torch.maximum(u, torch.tensor(1))
        v = torch.maximum(v, torch.tensor(1))
        snake_hist.append(np.array([u[:, 0].detach().cpu().numpy(), v[:, 0].detach().cpu().numpy()]))

    return u, v, snake_hist
