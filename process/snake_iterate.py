# Functionalized snake iteration process, only used in specified programs
import torch
import numpy as np
from models.ACStep import active_contour_process
from models.CAT import ConVEF_model
from process.snake_initialization import initialize_snake
from process.auxiliary_evolve_module import auxevolvehandler


from config import *


def snake_handler(i, mapE, mapA, mapB, batch_shape, CATkernelset, device, train_status, CATkernelset_dsp = None, CAT_dsp = None, CAT_usp = None):
    snake_result = np.zeros([batch_shape, L, 2])
    snake_result_list = []

    Mx, My = CATkernelset[0], CATkernelset[1]
    if use_dsp_CAT and i < dsp_stop_epoch:
        dsp_mapE = CAT_dsp(mapE)
        Mx_dsp, My_dsp = CATkernelset_dsp[0], CATkernelset_dsp[1]

    batch_gx1 = np.zeros([mapE.shape[0], mapE.shape[2], mapE.shape[3]])
    batch_gy1 = np.zeros([mapE.shape[0], mapE.shape[2], mapE.shape[3]])
    batch_Fu = np.zeros([mapE.shape[0], mapE.shape[2], mapE.shape[3]])
    batch_Fv = np.zeros([mapE.shape[0], mapE.shape[2], mapE.shape[3]])

    for b in range(batch_shape): # 对于bs中每一张map，进行单独的蛇演化
        if use_located_snake_init and i >= located_init_start_epoch:  # 蛇的自适应初始化规则
            now_snake = initialize_snake(snake_type, image_size, snake_init_scale, L,
                                          adaptive=True, Emap=mapE[b,0,:,:], device=device)
        else:
            now_snake = initialize_snake(snake_type, image_size, snake_init_scale, L,
                                          adaptive=False, device=device)

        # MapE计算图像梯度
        Fu = torch.gradient(mapE[b,0,:,:],dim=0)[0]
        Fv = torch.gradient(mapE[b,0,:,:],dim=1)[0]

        if not use_dsp_CAT or i >= dsp_stop_epoch:  # 计算CAT方向力
            gx0, gy0 = ConVEF_model(mapE[b,0,:,:], Mx, My)
        else:  # 计算CAT方向力（下采样版），为了加速，用的是这一版。
            gx0, gy0 = ConVEF_model(dsp_mapE[b,0,:,:], Mx_dsp, My_dsp)

        # 进行图均一化，使得CAT在各个方向上的绝对值均为1，这有助于加速蛇演化，并且真实的去做了capture range的扩大
        gx1, gy1 = gx0, gy0
        for ikk in range(0, gx0.shape[0]):
            for jkk in range(0, gx0.shape[1]):
                n_valsum = gx0[ikk, jkk] * gx0[ikk, jkk] + gy0[ikk, jkk] * gy0[ikk, jkk]  # CAT力的幅值
                franum = torch.sqrt(1 / n_valsum)
                gx1[ikk, jkk] *= franum
                gy1[ikk, jkk] *= franum

        if not use_dsp_CAT or i >= dsp_stop_epoch:  # 如果没用下采样，那就直接用归一化的CAT力。
            gx1 = gx1.to(device)
            gy1 = gy1.to(device)
        else:  # 计算CAT方向力（下采样版）：如果用了下采样，现在把CAT力上采样回来。
            gx1 = CAT_usp(gx1.unsqueeze(0))[0].to(device)
            gy1 = CAT_usp(gy1.unsqueeze(0))[0].to(device)

        if draw_force_field:
            batch_gx1[b,:,:] = gx1.cpu().detach().numpy()
            batch_gy1[b,:,:] = gy1.cpu().detach().numpy()
            batch_Fu[b,:,:] = Fu.cpu().detach().numpy()
            batch_Fv[b,:,:] = Fv.cpu().detach().numpy()

        if (adaptive_ACM_mode == 'train_only' and train_status) or (
                adaptive_ACM_mode == 'test_only' and not train_status) or (adaptive_ACM_mode == 'yes'):
            shistall = []
            last_evolve_rate = 0.0
            evolve_tries = 0
            while evolve_tries < max_ACM_reiter:
                su, sv, shist = active_contour_process(now_snake, Fu, Fv, mapA[b, 0, :, :], mapB[b, 0, :, :],
                                                       mCATu=-gx1, mCATv=gy1, iteration=ACM_iteration_base,
                                                       delta_s=ACM_paramset['delta_s'],
                                                       CAT_force_weight=ACM_paramset['CAT_forceweight'],
                                                       max_pixel_move=ACM_paramset['max_pixel_move'],
                                                       gamma=ACM_paramset['gamma'], device=device)

                now_snake[:, 0] = su[:, 0]
                now_snake[:, 1] = sv[:, 0]
                shistall += shist
                evolve_tries += 1

                coincide_rate = auxevolvehandler(mapE[b, 0, :, :], now_snake, b)
                if coincide_rate > 0.9:  # 判定为基本收敛
                    print("[Converge:%d]" % evolve_tries, end='')
                    break
                elif abs(coincide_rate - last_evolve_rate) < 0.01 and evolve_tries > 10:
                    print("[StopMove:%d]" % evolve_tries, end='')
                    break
                else:
                    last_evolve_rate = coincide_rate
            snake_result[b, :, 0] = now_snake[:, 0].detach().cpu().numpy()
            snake_result[b, :, 1] = now_snake[:, 1].detach().cpu().numpy()
            snake_result_list.append(shistall)
        else:  # 常规演化情况
            su, sv, shist = active_contour_process(now_snake, Fu, Fv, mapA[b, 0, :, :], mapB[b, 0, :, :],
                                                   mCATu=-gx1, mCATv=gy1, iteration=ACM_iterations,
                                                   delta_s=ACM_paramset['delta_s'],
                                                   CAT_force_weight=ACM_paramset['CAT_forceweight'],
                                                   max_pixel_move=ACM_paramset['max_pixel_move'],
                                                   gamma=ACM_paramset['gamma'], device=device)

            snake_result[b, :, 0] = su.detach().cpu().numpy()[:, 0]
            snake_result[b, :, 1] = sv.detach().cpu().numpy()[:, 0]
            snake_result_list.append(shist)

        snake_result[b, :, 0] = su.detach().cpu().numpy()[:,0]
        snake_result[b, :, 1] = sv.detach().cpu().numpy()[:,0]
        snake_result_list.append(shist)

    return snake_result, snake_result_list, batch_gx1, batch_gy1, batch_Fu, batch_Fv