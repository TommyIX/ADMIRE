import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# 防止forrtl: error (200)代替KeyboardInterrupt，保证中断时可以保存权重

import torch
import signal
import pickle
import numpy as np

from config.config_dsbn_mskt import *  # config.py里的所有变量都可以像写在这个py文件里一样，直接去用。
from dataset import build_dataloader_alldataset_DSBN

from models.UNet_head import UNet
from models.UNet_DSBN import UNet_DSBN
from models.ACStep import active_contour_process
from models.CAT import CATkernel, ConVEF_model
from models.CCQLoss import CCQLoss
from models.KnowledgeTransLoss import KnowledgeTransLoss
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

def snake_handler(i, mapE, mapA, mapB, batch_shape, CATkernelset, device, train_status, CATkernelset_dsp = None, CAT_dsp = None, CAT_usp = None, Config_file = None):
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
                                                       MAP_force_weight = ACM_paramset['Map_forceweight'],
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
                                                   MAP_force_weight=ACM_paramset['Map_forceweight'],
                                                   max_pixel_move=ACM_paramset['max_pixel_move'],
                                                   gamma=ACM_paramset['gamma'], device=device)

            snake_result[b, :, 0] = su.detach().cpu().numpy()[:, 0]
            snake_result[b, :, 1] = sv.detach().cpu().numpy()[:, 0]
            snake_result_list.append(shist)

        snake_result[b, :, 0] = su.detach().cpu().numpy()[:,0]
        snake_result[b, :, 1] = sv.detach().cpu().numpy()[:,0]
        snake_result_list.append(shist)

    return snake_result, snake_result_list, batch_gx1, batch_gy1, batch_Fu, batch_Fv

result_file = open("result.txt", "a", encoding="utf-8")
result_full_file = open("result_full.txt", "a", encoding="utf-8")

# Step1 - 准备数据
assert data_loadmode in ['folder', 'npy']
if 'ff' in divide_mode:
    print("当前正在运行数据集五折交叉验证，模式为", divide_mode,"，当前为第", ff_fold_num, "折")
    result_file.write("当前正在运行数据集五折交叉验证，模式为"+divide_mode+"，当前为第"+str(ff_fold_num)+"折\n")
    result_full_file.write("当前正在运行数据集五折交叉验证，模式为"+divide_mode+"，当前为第"+str(ff_fold_num)+"折\n")

if data_loadmode == 'npy':  # 现在采用的是npy格式的数据。
    print("正在直接从numpy文件中载入三个数据集并合并")
    assert isinstance(npy_dir[0], list)
    dataset_train, dataset_test, dataloader_train, dataloader_test = build_dataloader_alldataset_DSBN(npy_dir, image_size, divide_mode, batch_size, preload_mode=True, ff_fold_num=ff_fold_num, ff_random_seed=ffrad_seed, L_Points=L, instant_shuffle=shuffle_after_combine)
    print("载入数据集：训练集长度：", len(dataset_train), "，测试集长度：", len(dataset_test), "，训练集迭代器长度：", len(dataloader_train), "，测试集迭代器长度：", len(dataloader_test))

if os.path.exists(im_save_path) is False:
    os.mkdir(im_save_path)
if os.path.exists(model_save_path) is False:
    os.mkdir(model_save_path)

# Step2 - 构建网络
model = UNet_DSBN()
model_universal = UNet()

# ktloss = KnowledgeTransLoss()
ktloss = torch.nn.MSELoss()

resume_epoch = -1
if resume_training:
    model = pickle.load(open(load_ckpt_dir, 'rb'))
    model_universal.load_state_dict(torch.load(load_ckpt_dir_uni))
    resume_epoch = int(load_ckpt_dir.split('/')[-1].split('.')[0].split('_')[2])
    if len(load_ckpt_dir.split('/')[-1].split('.')[0].split('_'))>3:
        print("模型权重加载成功(从上次中断的模型加载，从当前epoch重新训练)，从epoch %d继续训练" % resume_epoch)
        resume_epoch -= 1
    else:
        print("模型权重加载成功，从epoch %d继续训练" % resume_epoch)

device = torch.device(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch)
optimizer_universal = torch.optim.Adam(model_universal.parameters(), lr=lr_uni)
scheduler_universal = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_universal, T_max=epoch)
force_stop = False

model.train()
model.to(device)
model_universal.train()
model_universal.to(device)

# Step3 - 训练网络
Mx, My = CATkernel(image_size, image_size, CAT_Sharpness)
if use_dsp_CAT:
    print("下采样CAT加速训练已设置，原尺寸：%d，下采样map尺寸：%d，下采样设置结束epoch：%d" % (image_size, image_size // dsp_CAT_scale, dsp_stop_epoch))
    Mx_dsp, My_dsp = CATkernel(image_size/dsp_CAT_scale, image_size/dsp_CAT_scale, CAT_Sharpness)
    CAT_dsp = Resize([int(image_size/dsp_CAT_scale), int(image_size/dsp_CAT_scale)])
    CAT_usp = Resize([image_size, image_size])

lastfield = 0
now_data_name = dataset_names[0]

for i in range(resume_epoch + 1, epoch):
    train_status = False
    for now_dataloader in [dataloader_train, dataloader_test]:
        train_status = not train_status

        full_iou = []
        full_dice = []
        full_wcov = []
        full_mbf = []
        full_iouu = []
        full_diceu = []
        full_wcovu = []
        full_mbfu = []

        if divide_mode == 'no' and train_status == False:  # 这种情况下不划分训练/测试集，此时测试集为None，直接开启下一个循环即可
            continue

        if result_save_rule == 'data':  # 在保存data数据的情况下，会使用RAM暂存图片，并统一压缩保存
            save_datalist = []
            save_datalistu = []

        if not do_train and train_status:
            continue

        for j, data in enumerate(now_dataloader):
            optimizer.zero_grad()
            optimizer_universal.zero_grad()  # 两个优化器都要清零梯度，这个应该是之前结果不对的原因
            image, contour, nowfield = data

            if nowfield != lastfield:
                model.BNShift(nowfield)
                now_data_name = dataset_names[nowfield]
                print("(BNShift to %d)"%nowfield, end='')
                model.train()
                model.to(device)
                lastfield = nowfield

            image = image.to(device)
            contour = contour.to(device)
            batch_shape = image.shape[0]

            modetitle = 'Train' if train_status else 'Test'
            if batch_shape > 1:
                print(modetitle + " Epoch %d, Batch %d" % (i, j), end='')
            else:
                print(modetitle + " Epoch %d, Image %d" % (i, j), end='')

            mapEo, mapAo, mapBo = model(image)
            mapEou, mapAou, mapBou = model_universal(image)

            with torch.no_grad():
                mapE = map_normalization(mapEo,batch_shape) * 12
                mapB = map_normalization(mapBo,batch_shape)
                mapA = map_normalization(mapAo,batch_shape)
                mapEu = map_normalization(mapEou,batch_shape) * 12
                mapBu = map_normalization(mapBou,batch_shape)
                mapAu = map_normalization(mapAou,batch_shape)

            if use_dsp_CAT and i< dsp_stop_epoch:
                dsp_mapE = CAT_dsp(mapE)
                dsp_mapEu = CAT_dsp(mapEu)
                # 上句，把蛇参数中的图像能量E下采样（前面定义了下采样操作，现在放入数据），
                #     下采样后，蛇参数图像能量会和下采样的卷积核Mx_dsp/My_dsp一样大小。
                # (在若干个时代后，会取消下采样，这样子就可以回复原有精度，保证最好效果了)

            snake_result, snake_result_list, batch_gx1, batch_gy1, batch_Fu, batch_Fv = \
                snake_handler(i, mapE, mapA, mapB, batch_shape, [Mx, My], device, train_status, [Mx_dsp, My_dsp], CAT_dsp, CAT_usp)

            snake_resultu, snake_result_listu, batch_gx1u, batch_gy1u, batch_Fuu, batch_Fvu = \
                snake_handler(i, mapEu, mapAu, mapBu, batch_shape, [Mx, My], device, train_status, [Mx_dsp, My_dsp], CAT_dsp, CAT_usp)

            # 以下，计算Auxiliary Branch的训练损失L_aux (CCQLoss版本) 并回传
            batch_mask = batch_mask_convert(contour, [image_size, image_size])
            if train_status:
                # Multi-site模型：正常更新Loss
                total_loss = CCQLoss.apply(mapEo, mapAo, mapBo, snake_result, contour, image_size, batch_shape, batch_mask)
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                model.eval()
                model_universal.train()
                ccqloss_u = CCQLoss.apply(mapEou, mapAou, mapBou, snake_resultu, contour, image_size, batch_shape, batch_mask)

                # ktloss_u = torch.mean(mapEuo*mapEo.data*2/(mapEuo**2+mapEo.data**2) + mapAuo*mapAo.data*2/(mapAuo**2+mapAo.data**2) + mapBuo*mapBo.data*2/(mapBuo**2+mapBo.data**2))
                ktloss_e = ktloss(mapEou, mapEo.data)
                ktloss_a = ktloss(mapAou, mapAo.data)
                ktloss_b = ktloss(mapBou, mapBo.data)
                ktloss_u = ktloss_e + ktloss_a + ktloss_b
                # ktloss_u = torch.tensor(0.0)
                total_lossu = ccqloss_u + ktloss_u * ktweight
                total_lossu.backward()
                optimizer_universal.step()
                scheduler_universal.step()
                model.train()

            # 计算模型指标
            with torch.no_grad():
                ioulist = []
                dicelist = []
                boundflist = []
                wcovlist = []
                ioulistu = []
                dicelistu = []
                boundflistu = []
                wcovlistu = []

                for m in range(batch_shape):
                    mask_snake = draw_poly_fill(snake_result[m, :, :], [image_size, image_size])
                    mask_snakeu = draw_poly_fill(snake_resultu[m, :, :], [image_size, image_size])
                    # batch_mask就是现在的mask_gt，取出即可
                    mask_gt = batch_mask[m, :, :].detach().cpu().numpy()

                    intersection = (mask_gt + mask_snake) == 2  # 金标准与预测值的并集
                    union = (mask_gt + mask_snake) >= 1  # 金标准与预测值的交集
                    iou = np.sum(intersection) / np.sum(union)  # 用定义计算的IoU值。
                    dice = 2 * iou / (iou + 1)  # F1-Score定义的Dice求法，与论文一致
                    boundf = FBound_metric(mask_snake, mask_gt)
                    wc = WCov_metric(mask_snake, mask_gt)

                    intersectionu = (mask_gt + mask_snakeu) == 2  # 金标准与预测值的并集
                    unionu = (mask_gt + mask_snakeu) >= 1  # 金标准与预测值的交集
                    iouu = np.sum(intersectionu) / np.sum(unionu)  # 用定义计算的IoU值。
                    diceu = 2 * iouu / (iouu + 1)  # F1-Score定义的Dice求法，与论文一致
                    boundfu = FBound_metric(mask_snakeu, mask_gt)
                    wcu = WCov_metric(mask_snakeu, mask_gt)

                    ioulist.append(iou)
                    full_iou.append(iou)
                    dicelist.append(dice)
                    full_dice.append(dice)
                    boundflist.append(boundf)
                    full_mbf.append(boundf)
                    wcovlist.append(wc)
                    full_wcov.append(wc)
                    ioulistu.append(iouu)
                    full_iouu.append(iouu)
                    dicelistu.append(diceu)
                    full_diceu.append(diceu)
                    boundflistu.append(boundfu)
                    full_mbfu.append(boundfu)
                    wcovlistu.append(wcu)
                    full_wcovu.append(wcu)

                    if result_save_rule == 'img' or os.path.exists("./override_result_save_rule.txt"):
                        plot_result(im_save_path, iou, i, j * batch_size + m,
                                    'train' if train_status else 'test' + "_" + now_data_name,
                                    snake_result[m, :, :], snake_result_list[m], contour[m, :, :].cpu().numpy(),
                                    mapE[m, 0, :, :].cpu().numpy(), mapA[m, 0, :, :].cpu().numpy(),
                                    mapB[m, 0, :, :].cpu().numpy(),
                                    image[m, :, :, :].cpu().numpy(), plot_force=draw_force_field,
                                    gx=batch_gx1[m, :, :], gy=batch_gy1[m, :, :], Fu=batch_Fu[m, :, :],
                                    Fv=batch_Fv[m, :, :])
                        plot_result(im_save_path, iou, i, j * batch_size + m,
                                    'universal_train' if train_status else 'universal_test' + "_" + now_data_name,
                                    snake_resultu[m, :, :], snake_result_listu[m], contour[m, :, :].cpu().numpy(),
                                    mapEu[m, 0, :, :].cpu().numpy(), mapAu[m, 0, :, :].cpu().numpy(),
                                    mapBu[m, 0, :, :].cpu().numpy(),
                                    image[m, :, :, :].cpu().numpy(), plot_force=draw_force_field,
                                    gx=batch_gx1u[m, :, :], gy=batch_gy1u[m, :, :], Fu=batch_Fuu[m, :, :],
                                    Fv=batch_Fvu[m, :, :])
                    elif result_save_rule == 'data':
                        now_savedata = {
                            "iou": iou,
                            "epoch": i,
                            "imnum": j * batch_size + m,
                            "status": 'train' if train_status else 'test' + "_" + now_data_name,
                            "mapE": mapE[m, 0, :, :].cpu().numpy(),
                            "mapA": mapA[m, 0, :, :].cpu().numpy(),
                            "mapB": mapB[m, 0, :, :].cpu().numpy(),
                            "snake_result": snake_result[m, :, :],
                            "snake_result_list": snake_result_list[m],
                            "GTContour": contour[m, :, :].cpu().numpy(),
                            "image": image[m, :, :, :].cpu().numpy(),
                            "gx": batch_gx1[m, :, :] if draw_force_field else None,
                            "gy": batch_gy1[m, :, :] if draw_force_field else None,
                            "Fu": batch_Fu[m, :, :] if draw_force_field else None,
                            "Fv": batch_Fv[m, :, :] if draw_force_field else None
                        }
                        save_datalist.append(now_savedata)
                        now_savedatau = {
                            "iou": iouu,
                            "epoch": i,
                            "imnum": j * batch_size + m,
                            "status": 'train_universal' if train_status else 'test_universal' + "_" + now_data_name,
                            "mapE": mapEu[m, 0, :, :].cpu().numpy(),
                            "mapA": mapAu[m, 0, :, :].cpu().numpy(),
                            "mapB": mapBu[m, 0, :, :].cpu().numpy(),
                            "snake_result": snake_resultu[m, :, :],
                            "snake_result_list": snake_result_listu[m],
                            "GTContour": contour[m, :, :].cpu().numpy(),
                            "image": image[m, :, :, :].cpu().numpy(),
                            "gx": batch_gx1u[m, :, :] if draw_force_field else None,
                            "gy": batch_gy1u[m, :, :] if draw_force_field else None,
                            "Fu": batch_Fuu[m, :, :] if draw_force_field else None,
                            "Fv": batch_Fvu[m, :, :] if draw_force_field else None
                        }
                        save_datalistu.append(now_savedatau)
                    else:
                        raise KeyError("Unknown result_save_rule! Aborting.")

                # 打印Aux Model数据
                if batch_shape > 1:
                    print(", Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f, loss: %.2f\n" % (
                    sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist), sum(boundflist) / len(boundflist),
                    sum(wcovlist) / len(wcovlist), total_loss.item()), end='')
                    result_file.write(
                        modetitle + "Epoch %d, Batch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f, loss: %.2f\n" % (
                        i, j, sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist),
                        sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist), total_loss.item()))
                else:
                    print(", IoU: %.4f, Dice: %.4f, mBF: %.4f, WCov: %.4f, loss: %.2f\n" % (
                    sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist), sum(boundflist) / len(boundflist),
                    sum(wcovlist) / len(wcovlist), total_loss.item()), end='')
                    result_file.write(
                        modetitle + "Epoch %d, Im %d, IoU: %.4f, Dice: %.4f, mBF: %.4f, WCov: %.4f, loss: %.2f\n" % (
                        i, j, sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist),
                        sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist), total_loss.item()))

                # 打印Universal Model数据
                if batch_shape > 1:
                    print("Uni Stat: Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f, ccqloss_u: %.2f, ktloss: %.2f" % (
                        sum(ioulistu) / len(ioulistu), sum(dicelistu) / len(dicelistu),
                        sum(boundflistu) / len(boundflistu), sum(wcovlistu) / len(wcovlistu), ccqloss_u.item(), ktloss_u.item()))
                    result_file.write(
                        modetitle + "(Uni) Epoch %d, Batch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f, ccqloss_u: %.2f, ktloss: %.2f\n" % (
                        i, j, sum(ioulistu) / len(ioulistu), sum(dicelistu) / len(dicelistu),
                        sum(boundflistu) / len(boundflistu), sum(wcovlistu) / len(wcovlistu), ccqloss_u.item(), ktloss_u.item()))
                else:
                    print("Uni Stat: IoU: %.4f, Dice: %.4f, mBF: %.4f, WCov: %.4f, ccqloss_u: %.2f, ktloss: %.2f" % (
                    sum(ioulistu) / len(ioulistu), sum(dicelistu) / len(dicelistu),
                    sum(boundflistu) / len(boundflistu), sum(wcovlistu) / len(wcovlistu), ccqloss_u.item(), ktloss_u.item()))
                    result_file.write(
                        modetitle + "(Uni) Epoch %d, Im %d, IoU: %.4f, Dice: %.4f, mBF: %.4f, WCov: %.4f, ccqloss_u: %.2f, ktloss: %.2f\n" % (
                        i, j, sum(ioulistu) / len(ioulistu), sum(dicelistu) / len(dicelistu),
                        sum(boundflistu) / len(boundflistu), sum(wcovlistu) / len(wcovlistu), ccqloss_u.item(), ktloss_u.item()))

            if force_stop:
                pickle.dump(model, open(model_save_path + 'ADMIRE_modelaux_%d_forcestop_%s_batchim_%d.pkl' % (
                           i, modetitle.strip(), (j + 1) * batch_size), 'wb'))
                torch.save(model_universal.state_dict(),
                           model_save_path + 'ADMIRE_modeluniversal_%d_forcestop_%s_batchim_%d.pth' % (
                           i, modetitle.strip(), (j + 1) * batch_size))
                pickle.dump(save_datalist, open(
                    im_save_path + 'ADMIREaux_resultdata_epoch_%d_forcestop_%s_batchim_%d.pkl' % (
                    i, modetitle.strip(), (j + 1) * batch_size), 'wb'))
                pickle.dump(save_datalistu, open(
                    im_save_path + 'ADMIREuniversal_resultdata_epoch_%d_forcestop_%s_batchim_%d.pkl' % (
                    i, modetitle.strip(), (j + 1) * batch_size), 'wb'))
                print("\n(*)中断权重文件与结果文件已保存")
                exit(500)

        pickle.dump(save_datalist, open(im_save_path + 'ADMIREaux_resultdata_epoch_%d_%s_data.pkl' % (i, modetitle.strip()), 'wb'))
        pickle.dump(save_datalistu, open(im_save_path + 'ADMIREuniversal_resultdata_epoch_%d_%s_data_universal.pkl' % (i, modetitle.strip()),'wb'))
        save_datalist.clear()
        save_datalistu.clear()

        result_file.write("--------------------------------------------------------\n")
        result_file.write(modetitle.strip())
        result_full_file.write(modetitle.strip())
        result_file.write(" Epoch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (
        i, sum(full_iou) / len(full_iou), sum(full_dice) / len(full_dice), sum(full_mbf) / len(full_mbf),
        sum(full_wcov) / len(full_wcov)))
        result_full_file.write(" Epoch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (
        i, sum(full_iou) / len(full_iou), sum(full_dice) / len(full_dice), sum(full_mbf) / len(full_mbf),
        sum(full_wcov) / len(full_wcov)))
        result_file.write(modetitle.strip() + "(Uni)")
        result_full_file.write(modetitle.strip() + "(Uni)")
        result_file.write(" Epoch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (
        i, sum(full_iouu) / len(full_iouu), sum(full_diceu) / len(full_diceu), sum(full_mbfu) / len(full_mbfu),
        sum(full_wcovu) / len(full_wcovu)))
        result_full_file.write(" Epoch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (
        i, sum(full_iouu) / len(full_iouu), sum(full_diceu) / len(full_diceu), sum(full_mbfu) / len(full_mbfu),
        sum(full_wcovu) / len(full_wcovu)))
        result_file.write("--------------------------------------------------------\n\n\n")
        result_full_file.flush()

        if train_status and do_train:  # 结束训练后保存模型checkpoint
            pickle.dump(model, open(model_save_path + 'ADMIRE_modelaux_%d.pkl' % i, 'wb'))
            torch.save(model_universal.state_dict(), model_save_path + 'ADMIRE_modeluniversal_%d.pth' % i)
            print("(*)权重文件已保存")