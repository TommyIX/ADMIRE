import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# 防止forrtl: error (200)代替KeyboardInterrupt，保证中断时可以保存权重

import cv2
import torch
import signal
import pickle
import numpy as np

from config.config_single import *  # config.py里的所有变量都可以像写在这个py文件里一样，直接去用。
from dataset import build_dataloader, build_dataloader_alldataset
from models.ACStep import active_contour_process
from models.UNet_head import UNet
from models.CAT import CATkernel, ConVEF_model
from models.CCQLoss import CCQLoss
from models.PolyProcess import draw_poly_fill, batch_mask_convert
from process.mapplot import plot_result
from process.mapprocess import map_normalization
from process.metrics import FBound_metric, WCov_metric
from process.snake_initialization import initialize_snake
from process.auxiliary_evolve_module import auxevolvehandler

from torchvision.transforms import Resize

# Ctrl+C处理，中断时保存数据
def emergency_stop(signum, frame):
    global force_stop
    force_stop = True
    print("捕获到Ctrl+C，正在保存当前权重")
signal.signal(signal.SIGINT, emergency_stop)
signal.signal(signal.SIGTERM, emergency_stop)  # 这个设计相当不错。

result_file = open("./result.txt", "a", encoding="utf-8")
result_full_file = open("./result_full.txt", "a", encoding="utf-8")

# Step1 - 准备数据
assert data_loadmode in ['folder', 'npy']
if 'ff' in divide_mode:
    print("当前正在运行数据集五折交叉验证，模式为", divide_mode,"，当前为第", ff_fold_num, "折")
    result_file.write("当前正在运行数据集五折交叉验证，模式为"+divide_mode+"，当前为第"+str(ff_fold_num)+"折\n")
    result_full_file.write("当前正在运行数据集五折交叉验证，模式为"+divide_mode+"，当前为第"+str(ff_fold_num)+"折\n")

if data_loadmode == 'npy':  # 现在采用的是npy格式的数据。
    dataset_train, dataset_test, dataloader_train, dataloader_test = build_dataloader(npy_dir, image_size, divide_mode, batch_size, preload_mode=True, ff_fold_num=ff_fold_num, ff_random_seed=ffrad_seed, L_Points=L, ACDC_mode=True)
else:
    print("正在从图像文件夹加载数据集")
    dataset_train, dataset_test, dataloader_train, dataloader_test = build_dataloader(folder_dir, image_size, divide_mode, batch_size, preload_mode=False, imnum=image_num, ff_fold_num=ff_fold_num, ff_random_seed=ffrad_seed, L_Points=L)
# 以上，构建PyTorch中的数据集类型和数据加载器类型。

if os.path.exists(im_save_path) is False:
    os.mkdir(im_save_path)
if os.path.exists(model_save_path) is False:
    os.mkdir(model_save_path)

# Step2 - 构建网络
model = UNet()
resume_epoch = -1
if resume_training:
    model.load_state_dict(torch.load(load_ckpt_dir))  # load和load_state_dict连用加载节点文件，火炬中的套路。
    resume_epoch = int(load_ckpt_dir.split('/')[-1].split('.')[0].split('_')[2])
    if len(load_ckpt_dir.split('/')[-1].split('.')[0].split('_'))>3:
        print("模型权重加载成功(从上次中断的模型加载，从当前epoch重新训练)，从epoch %d继续训练" % resume_epoch)
        resume_epoch -= 1
    else:
        print("模型权重加载成功，从epoch %d继续训练" % resume_epoch)

device = torch.device(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch)  # 学习率余弦退火，保证效果
force_stop = False

model.train()
model.to(device)

# Step3 - 训练网络
Mx, My = CATkernel(image_size, image_size, CAT_Sharpness)
# 上句，计算CAT滤波的卷积核。如果image_size是128，那么Mx和My都是127*127的卷积核，这样执行卷积的时候效率会比较低。
if use_dsp_CAT:  # 下采样CAT模型，有助于提高训练速度(此处为卷积核定义)
    print("下采样CAT加速训练已设置，原尺寸：%d，下采样map尺寸：%d，下采样设置结束epoch：%d" % (image_size, image_size // dsp_CAT_scale, dsp_stop_epoch))
    Mx_dsp, My_dsp = CATkernel(image_size/dsp_CAT_scale, image_size/dsp_CAT_scale, CAT_Sharpness)
    # 上句，设置下采样后CAT滤波的卷积核，卷积核小了，执行卷积的时候效率会提高一些。
    CAT_dsp = Resize([int(image_size/dsp_CAT_scale), int(image_size/dsp_CAT_scale)])  # 这个只是定义下采样操作（未放入数据），缩放到image_size/dsp_CAT_scale大小。
    CAT_usp = Resize([image_size, image_size])  # 这个是定义上采样操作，缩放到image_size大小。

print("准备开始训练")
for i in range(resume_epoch + 1, epoch):
    train_status = False
    for dataset_idx, now_dataloader in enumerate([dataloader_train, dataloader_test[0], dataloader_test[1]]):
        # 使用代码复用的训练-测试转换，只在训练时执行反向传播
        full_iou = []
        full_dice = []
        full_wcov = []
        full_mbf = []

        if dataset_idx % 3 == 0:
            train_status = not train_status
            dataset_name = 'ALL'
        elif dataset_idx % 3 == 1:
            train_status = not train_status
            dataset_name = 'ED'
        else:
            dataset_name = 'ES'

        if divide_mode == 'no' and train_status == False:  # 这种情况下不划分训练/测试集，此时测试集为None，直接开启下一个循环即可
            continue

        if result_save_rule == 'data':  # 在保存data数据的情况下，会使用RAM暂存图片，并统一压缩保存
            save_datalist = []

        if not do_train and train_status:
            continue

        for j, data in enumerate(now_dataloader):
            optimizer.zero_grad()
            image, contour = data
            image = image.to(device)
            contour = contour.to(device)
            batch_shape = image.shape[0]

            modetitle = 'Train ' if train_status else 'Test '
            if batch_shape > 1:
                print(modetitle + dataset_name + " Epoch %d, Batch %d" % (i, j), end='')
            else:
                print(modetitle + dataset_name + " Epoch %d, Image %d" % (i, j), end='')

            mapEo, mapAo, mapBo = model(image)  # 输出三个蛇参数图。

            with torch.no_grad():
                if morph_op_train and train_status or morph_op_test and not train_status:
                    mapE = map_normalization(mapEo,batch_shape).cpu().detach().numpy().squeeze()
                    mapB = map_normalization(mapBo,batch_shape).cpu().detach().numpy().squeeze()
                    mapA = map_normalization(mapAo,batch_shape).cpu().detach().numpy().squeeze()
                    mapE = cv2.dilate(mapE, np.ones((3, 3), np.uint8), iterations=1)
                    mapB = cv2.dilate(mapB, np.ones((3, 3), np.uint8), iterations=1)
                    mapA = cv2.dilate(mapA, np.ones((3, 3), np.uint8), iterations=1)
                    mapE = cv2.erode(mapE, np.ones((2, 2), np.uint8), iterations=2)
                    mapB = cv2.erode(mapB, np.ones((2, 2), np.uint8), iterations=2)
                    mapA = cv2.erode(mapA, np.ones((2, 2), np.uint8), iterations=2)
                    mapE = torch.from_numpy(mapE.reshape((1,1,128,128))).to(device) * 12
                    mapB = torch.from_numpy(mapB.reshape((1,1,128,128))).to(device)
                    mapA = torch.from_numpy(mapA.reshape((1,1,128,128))).to(device)
                else:
                    mapE = map_normalization(mapEo,batch_shape) * 12
                    mapB = map_normalization(mapBo,batch_shape)
                    mapA = map_normalization(mapAo,batch_shape)
                # 以上，蛇参数图归一化，并且缩放到合适的大小。12 0.7 0.65都是超参数。（注：这里修改了一下参数）

            snake_result = np.zeros([batch_shape, L, 2])
            snake_result_list = []

            if use_dsp_CAT and i< dsp_stop_epoch:
                dsp_mapE = CAT_dsp(mapE)
                # 上句，把蛇参数中的图像能量E下采样（前面定义了下采样操作，现在放入数据），
                #     下采样后，蛇参数图像能量会和下采样的卷积核Mx_dsp/My_dsp一样大小。
                # (在若干个时代后，会取消下采样，这样子就可以回复原有精度，保证最好效果了)

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
                # 以上，从u/v两个方向计算图像能量的导数。
                # 另外，可否把mapE小于某一定值的地方设为1，而mapE较大的地方设为0，从而得到一个0-1的掩膜，
                #     用这个掩膜乘以原图在u/v的导数（不是特征图的导数，因为原图在真实边界处似乎更精确），作为一个外力去演化蛇？
                #     这个可以之后再试，或者如果需要让结果变好的时候再试。

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
                # 以上，归一化CAT力场，每个点的CAT力都除以幅值。

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


                if (adaptive_ACM_mode == 'train_only' and train_status) or (adaptive_ACM_mode == 'test_only' and not train_status) or (adaptive_ACM_mode == 'yes'):
                    shistall = []
                    last_evolve_rate = 0.0
                    evolve_tries = 0
                    while evolve_tries < max_ACM_reiter:
                        su, sv, shist = active_contour_process(now_snake, Fu, Fv, mapA[b,0,:,:], mapB[b,0,:,:],
                                                               mCATu=-gx1, mCATv=gy1, iteration=ACM_iteration_base, delta_s=ACM_paramset['delta_s'],
                                                               CAT_force_weight=ACM_paramset['CAT_forceweight'], MAP_force_weight=ACM_paramset['Map_forceweight'], max_pixel_move=ACM_paramset['max_pixel_move'],
                                                               gamma=ACM_paramset['gamma'], device=device)

                        now_snake[:,0] = su[:, 0]
                        now_snake[:,1] = sv[:, 0]
                        shistall += shist
                        evolve_tries += 1

                        coincide_rate = auxevolvehandler(mapE[b, 0, :, :], now_snake, b)
                        if coincide_rate > 0.9:  # 判定为基本收敛
                            print("[Converge:%d]"%evolve_tries, end='')
                            break
                        elif abs(coincide_rate - last_evolve_rate) < 0.01 and evolve_tries > 10:
                            print("[StopMove:%d]"%evolve_tries, end='')
                            break
                        else:
                            last_evolve_rate = coincide_rate
                    snake_result[b, :, 0] = now_snake[:,0].detach().cpu().numpy()
                    snake_result[b, :, 1] = now_snake[:,1].detach().cpu().numpy()
                    snake_result_list.append(shistall)
                else: # 常规演化情况
                    su, sv, shist = active_contour_process(now_snake, Fu, Fv, mapA[b, 0, :, :], mapB[b, 0, :, :],
                                                           mCATu=-gx1, mCATv=gy1, iteration=ACM_iterations,
                                                           delta_s=ACM_paramset['delta_s'],
                                                           CAT_force_weight=ACM_paramset['CAT_forceweight'],
                                                           MAP_force_weight = ACM_paramset['Map_forceweight'],
                                                           max_pixel_move=ACM_paramset['max_pixel_move'],
                                                           gamma=ACM_paramset['gamma'], device=device)

                    snake_result[b, :, 0] = su.detach().cpu().numpy()[:, 0]
                    snake_result[b, :, 1] = sv.detach().cpu().numpy()[:, 0]
                    snake_result_list.append(shist)

            # 以下，计算损失、参数更新
            batch_mask = batch_mask_convert(contour, [image_size, image_size])
            if train_status:
                total_loss = CCQLoss.apply(mapEo, mapAo, mapBo, snake_result, contour, image_size, batch_shape, batch_mask)
                # 上句，计算CCQ损失。发现有些地方，好像CNN弄的是对的，但是蛇轮廓演化不对，比如说epoch-2-num-4268这张。
                #     有没有可能弄蛇上点损失呢？之前锦宏说需要for循环特别慢，但是好像不一定需要吧，矩阵操作应该就可以的。之前tf的MRCNN里有计算所有金标准和预测外接矩形的IOU的，这个应该也可以类似的方法。。
                #     然后又看了一些不太好的，比如说epoch-3-num-764、epoch-3-num-765等等，感觉这是蛇演化的问题啊，基本上E图是对的，
                #     而，如果初始化在蛇里面的，就不太行（应该对应锦宏说的椎骨在边上那些），是不是因为CAT力太小了，弄不过去。。
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                # 以上，计算损失，并且按照火炬的方式损失backward、优化子step、学习率规划子step。

            # 计算模型指标
            with torch.no_grad():
                ioulist = []
                dicelist = []
                boundflist = []
                wcovlist = []

                for m in range(batch_shape):
                    mask_snake = draw_poly_fill(snake_result[m,:,:], [image_size, image_size])
                    # batch_mask就是现在的mask_gt，取出即可
                    mask_gt = batch_mask[m,:,:].detach().cpu().numpy()

                    intersection = (mask_gt + mask_snake) == 2  # 金标准与预测值的并集
                    union = (mask_gt + mask_snake) >= 1  # 金标准与预测值的交集
                    iou = np.sum(intersection) / np.sum(union)  # 用定义计算的IoU值。
                    dice = 2 * iou / (iou + 1)  # F1-Score定义的Dice求法，与论文一致
                    boundf = FBound_metric(mask_snake, mask_gt)
                    wc = WCov_metric(mask_snake, mask_gt)
                    # 以上4句，计算各种指标。

                    ioulist.append(iou)
                    full_iou.append(iou)
                    dicelist.append(dice)
                    full_dice.append(dice)
                    boundflist.append(boundf)
                    full_mbf.append(boundf)
                    wcovlist.append(wc)
                    full_wcov.append(wc)

                    if result_save_rule == 'img' or os.path.exists("./override_result_save_rule.txt"):
                        plot_result(im_save_path, iou, i, j*batch_size+m, 'train' if train_status else 'test',
                                    snake_result[m,:,:], snake_result_list[m], contour[m,:,:].cpu().numpy(),
                                    mapE[m,0,:,:].cpu().numpy(), mapA[m,0,:,:].cpu().numpy(), mapB[m,0,:,:].cpu().numpy(),
                                    image[m,:,:,:].cpu().numpy(), plot_force=draw_force_field,
                                    gx=batch_gx1[m,:,:], gy=batch_gy1[m,:,:], Fu=batch_Fu[m,:,:], Fv=batch_Fv[m,:,:])
                    elif result_save_rule == 'data':
                        now_savedata = {
                            "iou": iou,
                            "epoch": i,
                            "imnum": j*batch_size+m,
                            "status": 'train' if train_status else 'test' + dataset_name,
                            "mapE": mapE[m,0,:,:].cpu().numpy(),
                            "mapA": mapA[m,0,:,:].cpu().numpy(),
                            "mapB": mapB[m,0,:,:].cpu().numpy(),
                            "snake_result": snake_result[m,:,:],
                            "snake_result_list": snake_result_list[m],
                            "GTContour": contour[m,:,:].cpu().numpy(),
                            "image": image[m,:,:,:].cpu().numpy(),
                            "gx": batch_gx1[m,:,:] if draw_force_field else None,
                            "gy": batch_gy1[m,:,:] if draw_force_field else None,
                            "Fu": batch_Fu[m,:,:] if draw_force_field else None,
                            "Fv": batch_Fv[m,:,:] if draw_force_field else None
                        }
                        save_datalist.append(now_savedata)
                    else:
                        raise KeyError("Unknown result_save_rule! Aborting.")

                if batch_shape > 1:
                    print(", Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f" % (sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist), sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist)), end='')
                    if train_status:
                        print(", loss: %.2f\n" % (total_loss.item()), end='')
                    else:
                        print("\n", end='')
                    result_file.write(modetitle + dataset_name + " Epoch %d, Batch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f" % (i, j, sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist),sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist)))
                    if train_status:
                        result_file.write(", loss: %.2f\n" % (total_loss.item()))
                    else:
                        result_file.write("\n")
                else:
                    print(", IoU: %.4f, Dice: %.4f, mBF: %.4f, WCov: %.4f" % (sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist), sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist)), end='')
                    if train_status:
                        print(", loss: %.2f\n" % (total_loss.item()), end='')
                    else:
                        print("\n", end='')
                    result_file.write(modetitle + dataset_name + " Epoch %d, Im %d, IoU: %.4f, Dice: %.4f, mBF: %.4f, WCov: %.4f" % (i, j, sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist) ,sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist)))
                    if train_status:
                        result_file.write(", loss: %.2f\n" % (total_loss.item()))
                    else:
                        result_file.write("\n")
            if force_stop:
                torch.save(model.state_dict(), model_save_path + 'ADMIRE_model_%d_forcestop_%s_batchim_%d.pth' % (i, modetitle.strip()+dataset_name, (j+1)*batch_size))
                pickle.dump(save_datalist, open(im_save_path + 'ADMIRE_resultdata_epoch_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip()+dataset_name, (j+1)*batch_size), 'wb'))
                print("\n(*)中断权重文件已保存至", model_save_path + 'ADMIRE_model_%d_forcestop_%s_batchim_%d.pth' % (i, modetitle.strip()+dataset_name, (j + 1) * batch_size))
                print("\n(*)中断结果数据已保存至", im_save_path + 'ADMIRE_resultdata_epoch_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip()+dataset_name, (j + 1) * batch_size))
                exit(500)

        pickle.dump(save_datalist, open(im_save_path + 'ADMIRE_resultdata_epoch_%d_%s_data.pkl' % (i, modetitle.strip()+dataset_name), 'wb'))
        save_datalist.clear()

        if train_status and do_train:  # 结束训练后保存模型checkpoint
            torch.save(model.state_dict(), model_save_path+'ADMIRE_model_%d.pth'%i)
            print("(*)权重文件已保存至",model_save_path+'ADMIRE_model_%d.pth'%i)

        result_file.write("--------------------------------------------------------\n")
        if train_status:
            result_file.write("Train")
            result_full_file.write("Train")
        else:
            result_file.write("Test " + dataset_name)
            result_full_file.write("Test " + dataset_name)
        result_file.write(" Epoch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (i, sum(full_iou)/len(full_iou), sum(full_dice)/len(full_dice), sum(full_mbf)/len(full_mbf), sum(full_wcov)/len(full_wcov)))
        result_full_file.write(" Epoch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (i, sum(full_iou)/len(full_iou), sum(full_dice)/len(full_dice), sum(full_mbf)/len(full_mbf), sum(full_wcov)/len(full_wcov)))
        result_file.write("--------------------------------------------------------\n\n\n")
        result_full_file.flush()
