import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# 防止forrtl: error (200)代替KeyboardInterrupt，保证中断时可以保存权重

import torch
import signal
import pickle
import numpy as np

from config.mixed_config.config import *  # config.py里的所有变量都可以像写在这个py文件里一样，直接去用。
from dataset import build_dataloader_alldataset
from models.ACStep import active_contour_process
from models.UNet_head import UNet
from models.CAT import CATkernel, ConVEF_model
from models.PolyProcess import draw_poly_fill, batch_mask_convert
from process.mapplot import plot_result
from process.mapprocess import map_normalization
from process.metrics import FBound_metric, WCov_metric
from process.snake_initialization import initialize_snake

from torchvision.transforms import Resize

# Ctrl+C处理，中断时保存数据
def emergency_stop(signum, frame):
    global force_stop
    force_stop = True
    print("捕获到Ctrl+C，正在保存当前数据")
signal.signal(signal.SIGINT, emergency_stop)
signal.signal(signal.SIGTERM, emergency_stop)  # 这个设计相当不错。

result_file = open("result.txt", "a", encoding="utf-8")
result_full_file = open("result_full.txt", "a", encoding="utf-8")

# Step1 - 准备数据
assert data_loadmode in ['folder', 'npy']
if 'ff' in divide_mode:
    print("当前正在运行数据集五折交叉验证，模式为", divide_mode,"，当前为第", ff_fold_num, "折")
    result_file.write("当前正在运行数据集五折交叉验证，模式为"+divide_mode+"，当前为第"+str(ff_fold_num)+"折\n")
    result_full_file.write("当前正在运行数据集五折交叉验证，模式为"+divide_mode+"，当前为第"+str(ff_fold_num)+"折\n")

if data_loadmode == 'npy':  # 现在采用的是npy格式的数据。
    print("正在直接从numpy文件中载入三个数据集")
    datasets_train, datasets_test, dataloaders_train, dataloaders_test = build_dataloader_alldataset(npy_dir, image_size, divide_mode, batch_size, preload_mode=True, ff_fold_num=ff_fold_num, ff_random_seed=ffrad_seed, L_Points=L)
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
force_stop = False

model.train()
model.to(device)

total_loss = torch.tensor(-1.0)  # 防止纯test模式下出错

# Step3 - 训练网络
Mx, My = CATkernel(image_size, image_size, CAT_Sharpness)
if use_dsp_CAT:  # 下采样CAT模型，有助于提高训练速度(此处为卷积核定义)
    print("下采样CAT加速训练已设置，原尺寸：%d，下采样map尺寸：%d，下采样设置结束epoch：%d" % (image_size, image_size // dsp_CAT_scale, dsp_stop_epoch))
    Mx_dsp, My_dsp = CATkernel(image_size/dsp_CAT_scale, image_size/dsp_CAT_scale, CAT_Sharpness)
    # 上句，设置下采样后CAT滤波的卷积核，卷积核小了，执行卷积的时候效率会提高一些。
    CAT_dsp = Resize([int(image_size/dsp_CAT_scale), int(image_size/dsp_CAT_scale)])  # 这个只是定义下采样操作（未放入数据），缩放到image_size/dsp_CAT_scale大小。
    CAT_usp = Resize([image_size, image_size])  # 这个是定义上采样操作，缩放到image_size大小。

print("准备开始执行全数据训练模型在不同数据集上的分别测试")
train_status = False

now_dataloaders = dataloaders_test
i = resume_epoch
for dni in range(len(dataset_names)):
    now_dataloader = now_dataloaders[dni]
    now_data_name = dataset_names[dni]

    full_iou = []
    full_dice = []
    full_wcov = []
    full_mbf = []

    if divide_mode == 'no' and train_status == False:  # 这种情况下不划分训练/测试集，此时测试集为None，直接开启下一个循环即可
        continue

    if result_save_rule == 'data':  # 在保存data数据的情况下，会使用RAM暂存图片，并统一压缩保存
        save_datalist = []

    if not do_train and train_status:
        continue

    for j, data in enumerate(now_dataloader):
        image, contour = data
        image = image.to(device)
        contour = contour.to(device)
        batch_shape = image.shape[0]

        modetitle = 'Train' if train_status else 'Test'
        modetitle += '_' + now_data_name
        if batch_shape > 1:
            print(modetitle + " Epoch %d, Batch %d" % (i, j), end='')
        else:
            print(modetitle + " Epoch %d, Image %d" % (i, j), end='')

        mapEo, mapAo, mapBo = model(image)  # 输出三个蛇参数图。

        with torch.no_grad():  # 不太确定为什么要用no_grad()？这个请教一下锦宏。
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
                init_snake = initialize_snake(snake_type, image_size, snake_init_scale, L,
                                              adaptive=True, Emap=mapE[b,0,:,:], device=device)
            else:
                init_snake = initialize_snake(snake_type, image_size, snake_init_scale, L,
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

            su, sv, shist = active_contour_process(init_snake, Fu, Fv, mapA[b,0,:,:], mapB[b,0,:,:],
                                                   mCATu=-gx1, mCATv=gy1, iteration=ACM_iterations, delta_s=ACM_paramset['delta_s'],
                                                   CAT_force_weight=ACM_paramset['CAT_forceweight'], max_pixel_move=ACM_paramset['max_pixel_move'],
                                                   gamma=ACM_paramset['gamma'], device=device)

            # 上句，蛇演化过程。确实火炬版本的放在这里，是带着数值演化的，比起tf版本更好看一些。锦宏修改的程序还是不错的。

            snake_result[b, :, 0] = su.detach().cpu().numpy()[:,0]
            snake_result[b, :, 1] = sv.detach().cpu().numpy()[:,0]
            snake_result_list.append(shist)
            # 以上，把蛇演化结果存放到snake_result里去，把蛇的历史结果存放到snake_result_list里去。

        batch_mask = batch_mask_convert(contour, [image_size, image_size])

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
                    plot_result(im_save_path, iou, i, j*batch_size+m, 'train' if train_status else 'test' + "_" + now_data_name,
                                snake_result[m,:,:], snake_result_list[m], contour[m,:,:].cpu().numpy(),
                                mapE[m,0,:,:].cpu().numpy(), mapA[m,0,:,:].cpu().numpy(), mapB[m,0,:,:].cpu().numpy(),
                                image[m,:,:,:].cpu().numpy(), plot_force=draw_force_field,
                                gx=batch_gx1[m,:,:], gy=batch_gy1[m,:,:], Fu=batch_Fu[m,:,:], Fv=batch_Fv[m,:,:])
                elif result_save_rule == 'data':
                    now_savedata = {
                        "iou": iou,
                        "epoch": i,
                        "imnum": j*batch_size+m,
                        "status": 'train' if train_status else 'test' + "_" + now_data_name,
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
                print(", Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f, loss: %.2f\n" % (sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist), sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist), total_loss.item()), end='')
                result_file.write(modetitle + "Epoch %d, Batch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f, loss: %.2f\n" % (i, j, sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist),sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist), total_loss.item()))
            else:
                print(", IoU: %.4f, Dice: %.4f, mBF: %.4f, WCov: %.4f, loss: %.2f\n" % (sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist), sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist), total_loss.item()), end='')
                result_file.write(modetitle + "Epoch %d, Im %d, IoU: %.4f, Dice: %.4f, mBF: %.4f, WCov: %.4f, loss: %.2f\n" % (i, j, sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist) ,sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist), total_loss.item()))

        if force_stop:
            pickle.dump(save_datalist, open(im_save_path + 'ADMIRE_resultdata_epoch_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j+1)*batch_size), 'wb'))
            print("\n(*)中断结果数据已保存至", im_save_path + 'ADMIRE_resultdata_epoch_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j + 1) * batch_size))
            exit(500)

    pickle.dump(save_datalist, open(im_save_path + 'ADMIRE_resultdata_epoch_%d_%s_data.pkl' % (i, modetitle.strip()), 'wb'))
    save_datalist.clear()

    result_file.write("--------------------------------------------------------\n")
    result_file.write(modetitle.strip())
    result_full_file.write(modetitle.strip())
    result_file.write(" Epoch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (i, sum(full_iou)/len(full_iou), sum(full_dice)/len(full_dice), sum(full_mbf)/len(full_mbf), sum(full_wcov)/len(full_wcov)))
    result_full_file.write(" Epoch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (i, sum(full_iou)/len(full_iou), sum(full_dice)/len(full_dice), sum(full_mbf)/len(full_mbf), sum(full_wcov)/len(full_wcov)))
    result_file.write("--------------------------------------------------------\n\n\n")
    result_full_file.flush()
