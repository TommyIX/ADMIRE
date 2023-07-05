import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# 防止forrtl: error (200)代替KeyboardInterrupt，保证中断时可以保存权重

import torch
import signal
import pickle
import numpy as np

from config.config_mt import *  # config.py里的所有变量都可以像写在这个py文件里一样，直接去用。
from dataset import build_dataloader

from models.UNet_head import UNet
from models.CAT import CATkernel
from models.CCQLoss import CCQLoss
from models.PolyProcess import draw_poly_fill, batch_mask_convert
from process.mapplot import plot_result
from process.mapprocess import map_normalization
from process.metrics import FBound_metric, WCov_metric
from process.snake_iterate import snake_handler

from process.MTprocess import get_current_consistency_weight, update_ema_variables

from torchvision.transforms import Resize

# Ctrl+C处理，中断时保存数据
def emergency_stop(signum, frame):
    global force_stop
    force_stop = True
    print("捕获到Ctrl+C，正在保存当前权重")
signal.signal(signal.SIGINT, emergency_stop)
signal.signal(signal.SIGTERM, emergency_stop)  # 这个设计相当不错。

result_file = open("result.txt", "a", encoding="utf-8")
result_full_file = open("result_full.txt", "a", encoding="utf-8")

# Step1 - 准备数据
assert data_loadmode in ['folder', 'npy']
result_file.write("正在运行Mean Teacher版本ADMIRE，当前batch_size为" + str(batch_size) + "，每批次有监督图片数量：" + str(labeled_batch_size))
result_full_file.write("正在运行Mean Teacher版本ADMIRE，当前batch_size为" + str(batch_size) + "，每批次有监督图片数量：" + str(labeled_batch_size))

if 'ff' in divide_mode:
    print("当前正在运行数据集五折交叉验证，模式为", divide_mode,"，当前为第", ff_fold_num, "折")
    result_file.write("当前正在运行数据集五折交叉验证，模式为"+divide_mode+"，当前为第"+str(ff_fold_num)+"折\n")
    result_full_file.write("当前正在运行数据集五折交叉验证，模式为"+divide_mode+"，当前为第"+str(ff_fold_num)+"折\n")

if data_loadmode == 'npy':  # 现在采用的是npy格式的数据。
    print("正在直接从numpy文件中载入数据集")
    dataset_train, dataset_test, dataloader_train, dataloader_test = build_dataloader(npy_dir, image_size, divide_mode, batch_size, preload_mode=True, ff_fold_num=ff_fold_num, ff_random_seed=ffrad_seed, L_Points=L)
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
ema_model = UNet()
for param in ema_model.parameters():
    param.detach_()

resume_epoch = -1
if resume_training:
    model.load_state_dict(torch.load(load_ckpt_dir))
    ema_model.load_state_dict(torch.load(load_ckpt_ema_dir))
    resume_epoch = int(load_ckpt_dir.split('/')[-1].split('.')[0].split('_')[-1])
    if len(load_ckpt_dir.split('/')[-1].split('.')[0].split('_')) > 4:
        print("模型权重加载成功(从上次中断的模型加载)，从epoch %d执行测试" % resume_epoch)
        resume_epoch -= 1
    else:
        print("模型权重加载成功，从epoch %d执行测试" % resume_epoch)

device = torch.device(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch)  # 学习率余弦退火，保证效果
force_stop = False

model.train()  # model为使用有标注数据训练的模型，而ema_model为使用无标注数据训练的
model.to(device)
ema_model.train()
ema_model.to(device)

# Step3 - 训练网络
Mx, My = CATkernel(image_size, image_size, CAT_Sharpness)
# 上句，计算CAT滤波的卷积核。如果image_size是128，那么Mx和My都是127*127的卷积核，这样执行卷积的时候效率会比较低。
if use_dsp_CAT:  # 下采样CAT模型，有助于提高训练速度(此处为卷积核定义)
    print("下采样CAT加速训练已设置，原尺寸：%d，下采样map尺寸：%d，下采样设置结束epoch：%d" % (image_size, image_size // dsp_CAT_scale, dsp_stop_epoch))
    Mx_dsp, My_dsp = CATkernel(image_size/dsp_CAT_scale, image_size/dsp_CAT_scale, CAT_Sharpness)
    # 上句，设置下采样后CAT滤波的卷积核，卷积核小了，执行卷积的时候效率会提高一些。
    CAT_dsp = Resize([int(image_size/dsp_CAT_scale), int(image_size/dsp_CAT_scale)])  # 这个只是定义下采样操作（未放入数据），缩放到image_size/dsp_CAT_scale大小。
    CAT_usp = Resize([image_size, image_size])  # 这个是定义上采样操作，缩放到image_size大小。

iter_num = 0
total_loss = torch.tensor(0.0)  # 纯测试模式下防止报错
consistency_loss = torch.tensor(0.0)
consistency_weight = 0.0

print("准备开始训练")
for i in range(resume_epoch + 1, epoch):
    train_status = False
    for now_dataloader in [dataloader_train, dataloader_test]:
        # 使用代码复用的训练-测试转换，只在训练时执行反向传播
        full_iou = []
        full_dice = []
        full_wcov = []
        full_mbf = []
        full_ema_iou = []
        full_ema_dice = []
        full_ema_wcov = []
        full_ema_mbf = []

        train_status = not train_status

        if not do_train and train_status:
            continue
        if skip_test_before_mean_teacher and not train_status and i < mean_teacher_epochs:
            continue
        if divide_mode == 'no' and train_status == False:  # 这种情况下不划分训练/测试集，此时测试集为None，直接开启下一个循环即可
            continue

        if result_save_rule == 'data':  # 在保存data数据的情况下，会使用RAM暂存图片，并统一压缩保存
            save_datalist = []
            save_datalist_ema = []

        for j, data in enumerate(now_dataloader):
            iter_num += 1
            optimizer.zero_grad()
            image, contour = data
            image = image.to(device)
            contour = contour.to(device)
            batch_shape = image.shape[0]

            if batch_shape - labeled_batch_size <= 0 and train_status:  # 如果无标注数据不足一个batch，直接跳过本次循环
                continue

            if i >= mean_teacher_epochs:  # 时代数大于mean_teacher_epochs才开始用无标签数据。
                image1 = image.clone()
                noise = torch.clamp(torch.randn_like(image1) * 0.1, -3, 3)
                teacher_model_inputs = image1 + noise  # 不用标签的数据加上个噪声，构建教师模型的输入。

            labeled_image = image[:labeled_batch_size,:,:,:]  # 后续lb对应的这些，这些数据是用标签的。
            unlabeled_image = image[labeled_batch_size:,:,:,:]  # 后续ulb对应的这些，这些数据是不用标签的，作为无标签数据。

            if train_status:
                if i >= mean_teacher_epochs:  # 使用MT机制训练，这个是有无标签数据都给他放到模型里去。
                    mapEo_lb, mapAo_lb, mapBo_lb = model(labeled_image)  # 用标签数据的各种蛇参数图
                    mapEo_ulb, mapAo_ulb, mapBo_ulb = model(unlabeled_image)  # 不用标签的数据的各种蛇参数图
                    ema_mapEo, ema_mapAo, ema_mapBo = ema_model(teacher_model_inputs)  # 加噪声的不用标签的数据的蛇参数图
                else:  # else里是说，前面mean_teacher_epochs个时代，只用有标签数据。
                    mapEo_lb, mapAo_lb, mapBo_lb = model(labeled_image)  # 在前面mean_teacher_epochs个时代是只有有标签数据的蛇参数图。
                    batch_shape = labeled_batch_size  # 如果时代数小于mean_teacher_epochs，就缩小批大小，也就是说只保留有标签部分。
            else:  # 测试模式下使用全部数据做测试
                mapEo_lb, mapAo_lb, mapBo_lb = model(image)  # 测试数据是全部有标签的，复用了这部分代码
                if i >= mean_teacher_epochs:
                    ema_mapEo, ema_mapAo, ema_mapBo = ema_model(image)  # 加噪声的不用标签的数据的蛇参数图

            modetitle = 'Train ' if train_status else 'Test '
            if batch_shape > 1:
                print(modetitle + "Epoch %d, Batch %d" % (i, j), end='')
            else:
                print(modetitle + "Epoch %d, Image %d" % (i, j), end='')

            with torch.no_grad():
                if i >= mean_teacher_epochs:
                    mapE_lb = map_normalization(mapEo_lb, labeled_batch_size if train_status else batch_shape) * 12
                    mapA_lb = map_normalization(mapAo_lb, labeled_batch_size if train_status else batch_shape)
                    mapB_lb = map_normalization(mapBo_lb, labeled_batch_size if train_status else batch_shape)
                    ema_mapE = map_normalization(ema_mapEo, batch_shape) * 12
                    ema_mapB = map_normalization(ema_mapBo, batch_shape)
                    ema_mapA = map_normalization(ema_mapAo, batch_shape)
                    if train_status:
                        mapE_ulb = map_normalization(mapEo_ulb, batch_shape - labeled_batch_size) * 12
                        mapA_ulb = map_normalization(mapAo_ulb, batch_shape - labeled_batch_size)
                        mapB_ulb = map_normalization(mapBo_ulb, batch_shape - labeled_batch_size)
                else:
                    mapE_lb = map_normalization(mapEo_lb, batch_shape) * 12
                    mapA_lb = map_normalization(mapAo_lb, batch_shape)
                    mapB_lb = map_normalization(mapBo_lb, batch_shape)
                    # 以上，蛇参数图归一化，并且缩放到合适的大小。12 0.7 0.65都是超参数。（注：这里修改了一下参数）
                    # 也是在时代数大于的时候mean_teacher_epochs，就分别对用标签的和不用标签的数据，分别做的蛇参数图归一化。否则就只处理有标签的数据。

            # 有标注部分蛇演化
            snake_result, snake_result_list, batch_gx1, batch_gy1, batch_Fu, batch_Fv = \
                snake_handler(i, mapE_lb, mapA_lb, mapB_lb, labeled_batch_size if train_status else batch_shape,
                              [Mx, My], device, train_status, [Mx_dsp, My_dsp], CAT_dsp, CAT_usp)  # 学生模型，用标签的数据，做了蛇演化

            if i >= mean_teacher_epochs:
                # 半监督模型蛇演化
                ema_snake_result, ema_snake_result_list, ema_batch_gx1, ema_batch_gy1, ema_batch_Fu, ema_batch_Fv = \
                    snake_handler(i, ema_mapE, ema_mapA, ema_mapB, batch_shape, [Mx, My], device, train_status, [Mx_dsp, My_dsp],
                                    CAT_dsp, CAT_usp)  # 教师模型，有噪声的、不用标签的数据，做了蛇演化。
                #  为什么还要把教师模型的mapE A B也用来演化蛇呢？是因为，要用教师模型的演化结果计算指标。

                # 无标注部分蛇演化
                if train_status:
                    ulb_snake_result, ulb_snake_result_list, ulb_batch_gx1, ulb_batch_gy1, ulb_batch_Fu, ulb_batch_Fv = \
                        snake_handler(i, mapE_ulb, mapA_ulb, mapB_ulb, batch_shape - labeled_batch_size, [Mx, My], device, train_status,
                                      [Mx_dsp, My_dsp], CAT_dsp, CAT_usp)  # 学生模型，不用标签的数据，做了蛇演化

            # 以下，计算损失、参数更新
            batch_mask = batch_mask_convert(contour, [image_size, image_size])
            if train_status:
                supervised_loss = CCQLoss.apply(mapEo_lb, mapAo_lb, mapBo_lb, snake_result, contour, image_size,
                                                labeled_batch_size, batch_mask)  # 用标签的数据，监督损失，这个是一直都有的。
                if i < mean_teacher_epochs:  # 前面mean_teacher_epochs个时代，只用用了标签的数据，不计算一致性损失。
                    consistency_loss = 0.0
                else:
                    consistency_loss = torch.mean((mapEo_lb - ema_mapEo[:labeled_batch_size]) ** 2 + \
                                                  (mapAo_lb - ema_mapAo[:labeled_batch_size]) ** 2 + \
                                                  (mapBo_lb - ema_mapBo[:labeled_batch_size]) ** 2) + \
                                       torch.mean((mapEo_ulb - ema_mapEo[labeled_batch_size:]) ** 2 + \
                                                  (mapAo_ulb - ema_mapAo[labeled_batch_size:]) ** 2 + \
                                                  (mapBo_ulb - ema_mapBo[labeled_batch_size:]) ** 2)
                    """这个地方可以先按照mapEo的损失去搞，但是我感觉，可能真的要有无标签的同时计算一致性损失。
                    我不太确定刚你说加了一致性损失后，map会逐渐变淡，会不会是因为没有加有标签数据的一致性损失导致的，
                        但我感觉有可能是，因为在用无标记数据的时候，有标签数据似乎也不应该浪费了，似乎，有标签数据的一致性损失，可以更好地让教师模型去逐渐靠拢学生模型学到的东西？
                    一步一步来，我们的步骤是：1、首先把表填掉，弄出来差不多的数据，这之后，再考虑其他的思路：
                    2、有标签数据也加上一致性损失
                    3、看看是不是可以用mapE什么的计算损失了。
                    4、那个150改一改能不能更好一些。
                    """
                consistency_weight = get_current_consistency_weight(i - mean_teacher_epochs)  # 这个150到底怎么改呢？？ -———改成epoch num试试

                total_loss = supervised_loss + consistency_weight * consistency_loss
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                if i >= mean_teacher_epochs:
                    update_ema_variables(model, ema_model, ema_decay, iter_num)

            # 特定断电
            if j == 5:
                print("cut")

            # 计算模型指标
            with torch.no_grad():
                ioulist = []
                dicelist = []
                boundflist = []
                wcovlist = []
                ema_ioulist = []
                ema_dicelist = []
                ema_boundflist = []
                ema_wcovlist = []

                for m in range(batch_shape):
                    if train_status:
                        if m < labeled_batch_size:
                            mask_snake = draw_poly_fill(snake_result[m,:,:], [image_size, image_size])
                        else:
                            mask_snake = draw_poly_fill(ulb_snake_result[m-labeled_batch_size,:,:], [image_size, image_size])
                    else:
                        mask_snake = draw_poly_fill(snake_result[m,:,:], [image_size, image_size])
                    # batch_mask就是现在的mask_gt，取出即可
                    mask_gt = batch_mask[m,:,:].detach().cpu().numpy()

                    intersection = (mask_gt + mask_snake) == 2  # 金标准与预测值的并集
                    union = (mask_gt + mask_snake) >= 1  # 金标准与预测值的交集
                    iou = np.sum(intersection) / np.sum(union)  # 用定义计算的IoU值。
                    dice = 2 * iou / (iou + 1)  # F1-Score定义的Dice求法，与论文一致
                    boundf = FBound_metric(mask_snake, mask_gt)
                    wc = WCov_metric(mask_snake, mask_gt)

                    if i >= mean_teacher_epochs:
                        ema_mask_snake = draw_poly_fill(ema_snake_result[m,:,:], [image_size, image_size])
                        ema_intersection = (mask_gt + ema_mask_snake) == 2
                        ema_union = (mask_gt + ema_mask_snake) >= 1
                        ema_iou = np.sum(ema_intersection) / np.sum(ema_union)
                        ema_dice = 2 * ema_iou / (ema_iou + 1)
                        ema_boundf = FBound_metric(ema_mask_snake, mask_gt)
                        ema_wcov = WCov_metric(ema_mask_snake, mask_gt)
                        ema_ioulist.append(ema_iou)
                        full_ema_iou.append(ema_iou)
                        ema_dicelist.append(ema_dice)
                        full_ema_dice.append(ema_dice)
                        ema_boundflist.append(ema_boundf)
                        full_ema_mbf.append(ema_boundf)
                        ema_wcovlist.append(ema_wcov)
                        full_ema_wcov.append(ema_wcov)

                    ioulist.append(iou)
                    full_iou.append(iou)
                    dicelist.append(dice)
                    full_dice.append(dice)
                    boundflist.append(boundf)
                    full_mbf.append(boundf)
                    wcovlist.append(wc)
                    full_wcov.append(wc)

                    if result_save_rule == 'img' or os.path.exists("./override_result_save_rule.txt"):
                        if m < labeled_batch_size or not train_status:
                            plot_result(im_save_path, iou, i, j*batch_size+m, 'train' if train_status else 'test',
                                        snake_result[m,:,:], snake_result_list[m], contour[m,:,:].cpu().numpy(),
                                        mapE_lb[m,0,:,:].cpu().numpy(), mapA_lb[m,0,:,:].cpu().numpy(), mapB_lb[m,0,:,:].cpu().numpy(),
                                        image[m,:,:,:].cpu().numpy(), plot_force=draw_force_field,
                                        gx=batch_gx1[m,:,:], gy=batch_gy1[m,:,:], Fu=batch_Fu[m,:,:], Fv=batch_Fv[m,:,:])
                        else:
                            plot_result(im_save_path, iou, i, j*batch_size+m, 'train' if train_status else 'test',
                                        ulb_snake_result[m-labeled_batch_size,:,:], ulb_snake_result_list[m-labeled_batch_size],
                                        contour[m,:,:].cpu().numpy(), mapE_ulb[m-labeled_batch_size,0,:,:].cpu().numpy(),
                                        mapA_ulb[m-labeled_batch_size,0,:,:].cpu().numpy(), mapB_ulb[m-labeled_batch_size,0,:,:].cpu().numpy(),
                                        image[m,:,:,:].cpu().numpy(), plot_force=draw_force_field,
                                        gx=ulb_batch_gx1[m-labeled_batch_size,:,:], gy=ulb_batch_gy1[m-labeled_batch_size,:,:], Fu=ulb_batch_Fu[m-labeled_batch_size,:,:], Fv=ulb_batch_Fv[m-labeled_batch_size,:,:])

                        if i >= mean_teacher_epochs:
                            plot_result(im_save_path, ema_iou, i, j*batch_size+m, 'train_ema' if train_status else 'test_ema',
                                        ema_snake_result[m,:,:], ema_snake_result_list[m],
                                        contour[m,:,:].cpu().numpy(), ema_mapE[m-labeled_batch_size,0,:,:].cpu().numpy(),
                                        ema_mapA[m,0,:,:].cpu().numpy(), ema_mapB[m,0,:,:].cpu().numpy(),
                                        image[m,:,:,:].cpu().numpy(), plot_force=draw_force_field,
                                        gx=ema_batch_gx1[m,:,:], gy=ema_batch_gy1[m,:,:], Fu=ema_batch_Fu[m,:,:], Fv=ema_batch_Fv[m,:,:])

                    elif result_save_rule == 'data':
                        now_savedata = {
                            "iou": iou,
                            "epoch": i,
                            "imnum": j*batch_size+m,
                            "status": 'train' if train_status else 'test',
                            "mapE": mapE_lb[m,0,:,:].cpu().numpy() if m < labeled_batch_size or not train_status else mapE_ulb[m-labeled_batch_size,0,:,:].cpu().numpy(),
                            "mapA": mapA_lb[m,0,:,:].cpu().numpy() if m < labeled_batch_size or not train_status else mapA_ulb[m-labeled_batch_size,0,:,:].cpu().numpy(),
                            "mapB": mapB_lb[m,0,:,:].cpu().numpy() if m < labeled_batch_size or not train_status else mapB_ulb[m-labeled_batch_size,0,:,:].cpu().numpy(),
                            "snake_result": snake_result[m,:,:] if m < labeled_batch_size or not train_status else ulb_snake_result[m-labeled_batch_size,:,:],
                            "snake_result_list": snake_result_list[m] if m < labeled_batch_size or not train_status else ulb_snake_result_list[m-labeled_batch_size],
                            "GTContour": contour[m,:,:].cpu().numpy(),
                            "image": image[m,:,:,:].cpu().numpy(),
                            "gx": batch_gx1[m,:,:] if m < labeled_batch_size or not train_status else ulb_batch_gx1[m-labeled_batch_size,:,:] if draw_force_field else None,
                            "gy": batch_gy1[m,:,:] if m < labeled_batch_size or not train_status else ulb_batch_gy1[m-labeled_batch_size,:,:] if draw_force_field else None,
                            "Fu": batch_Fu[m,:,:] if m < labeled_batch_size or not train_status else ulb_batch_Fu[m-labeled_batch_size,:,:] if draw_force_field else None,
                            "Fv": batch_Fv[m,:,:] if m < labeled_batch_size or not train_status else ulb_batch_Fv[m-labeled_batch_size,:,:] if draw_force_field else None
                        }
                        save_datalist.append(now_savedata)
                        if i >= mean_teacher_epochs:
                            now_savedata = {
                                "iou": ema_iou,
                                "epoch": i,
                                "imnum": j*batch_size+m,
                                "status": 'train_ema' if train_status else 'test_ema',
                                "mapE": ema_mapE[m,0,:,:].cpu().numpy(),
                                "mapA": ema_mapA[m,0,:,:].cpu().numpy(),
                                "mapB": ema_mapB[m,0,:,:].cpu().numpy(),
                                "snake_result": ema_snake_result[m,:,:],
                                "snake_result_list": ema_snake_result_list[m],
                                "GTContour": contour[m,:,:].cpu().numpy(),
                                "image": image[m,:,:,:].cpu().numpy(),
                                "gx": ema_batch_gx1[m,:,:] if draw_force_field else None,
                                "gy": ema_batch_gy1[m,:,:] if draw_force_field else None,
                                "Fu": ema_batch_Fu[m,:,:] if draw_force_field else None,
                                "Fv": ema_batch_Fv[m,:,:] if draw_force_field else None
                            }
                            save_datalist_ema.append(now_savedata)
                    else:
                        raise KeyError("Unknown result_save_rule! Aborting.")

                if batch_shape > 1:
                    print(", Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f" % (sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist), sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist)), end='')
                    if train_status:
                        print(", loss: %.2f\n" % (total_loss.item()), end='')
                    else:
                        print("\n", end='')
                    result_file.write(modetitle + "Epoch %d, Batch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f" % (i, j, sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist),sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist)))
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
                    result_file.write(modetitle + "Epoch %d, Im %d, IoU: %.4f, Dice: %.4f, mBF: %.4f, WCov: %.4f" % (i, j, sum(ioulist) / len(ioulist), sum(dicelist) / len(dicelist) ,sum(boundflist) / len(boundflist), sum(wcovlist) / len(wcovlist)))
                    if train_status:
                        result_file.write(", loss: %.2f\n" % (total_loss.item()))
                    else:
                        result_file.write("\n")

                if i >= mean_teacher_epochs:
                    print("EMA Status:, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f" % (sum(ema_ioulist) / len(ema_ioulist), sum(ema_dicelist) / len(ema_dicelist), sum(ema_boundflist) / len(ema_boundflist), sum(ema_wcovlist) / len(ema_wcovlist)), end='')
                    if train_status:
                        print(", consistency loss: %.2f, consistency weight: %.2f\n"%(consistency_loss.item(), consistency_weight))
                    else:
                        print("\n")

            if force_stop:
                torch.save(model.state_dict(), model_save_path + 'ADMIRE_model_%d_forcestop_%s_batchim_%d.pth' % (i, modetitle.strip(), (j+1)*batch_size))
                torch.save(ema_model.state_dict(), model_save_path + 'ADMIRE_ema_model_%d_forcestop_%s_batchim_%d.pth' % (i, modetitle.strip(), (j+1)*batch_size))
                pickle.dump(save_datalist, open(im_save_path + 'ADMIRE_resultdata_epoch_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j+1)*batch_size), 'wb'))
                pickle.dump(save_datalist_ema, open(im_save_path + 'ADMIRE_resultdata_ema_epoch_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j+1)*batch_size), 'wb'))
                print("(*)中断权重文件已保存至", model_save_path + 'ADMIRE_model_%d_forcestop_%s_batchim_%d.pth' % (i, modetitle.strip(), (j + 1) * batch_size))
                print("(*)中断结果数据已保存至", im_save_path + 'ADMIRE_resultdata_epoch_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j + 1) * batch_size))
                exit(500)

        pickle.dump(save_datalist, open(im_save_path + 'ADMIRE_resultdata_epoch_%d_%s_data.pkl' % (i, modetitle.strip()), 'wb'))
        pickle.dump(save_datalist_ema, open(im_save_path + 'ADMIRE_resultdata_ema_epoch_%d_%s_data.pkl' % (i, modetitle.strip()), 'wb'))
        save_datalist.clear()
        save_datalist_ema.clear()

        if train_status:  # 结束训练后保存模型checkpoint
            torch.save(model.state_dict(), model_save_path+'ADMIRE_model_%d.pth'%i)
            torch.save(ema_model.state_dict(), model_save_path+'ADMIRE_model_ema_%d.pth'%i)
            print("(*)权重文件已保存至",model_save_path+'ADMIRE_model_%d.pth'%i,"和",model_save_path+'ADMIRE_model_ema_%d.pth'%i)

        result_file.write("--------------------------------------------------------\n")
        if train_status:
            result_file.write("Train")
            result_full_file.write("Train")
        else:
            result_file.write("Test")
            result_full_file.write("Test")
        result_file.write(" Epoch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (i, sum(full_iou)/len(full_iou), sum(full_dice)/len(full_dice), sum(full_mbf)/len(full_mbf), sum(full_wcov)/len(full_wcov)))
        result_full_file.write(" Epoch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (i, sum(full_iou)/len(full_iou), sum(full_dice)/len(full_dice), sum(full_mbf)/len(full_mbf), sum(full_wcov)/len(full_wcov)))

        if i>=mean_teacher_epochs:
            result_file.write(
                "EMA Status:, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f, consistency loss: %.2f, consistency weight: %.2f\n" % (
                sum(ema_ioulist) / len(ema_ioulist), sum(ema_dicelist) / len(ema_dicelist),
                sum(ema_boundflist) / len(ema_boundflist), sum(ema_wcovlist) / len(ema_wcovlist), consistency_loss.item(),
                consistency_weight))
            result_full_file.write("EMA Status: Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (sum(full_ema_iou)/len(full_ema_iou), sum(full_ema_dice)/len(full_ema_dice), sum(full_ema_mbf)/len(full_ema_mbf), sum(full_ema_wcov)/len(full_ema_wcov)))
        result_file.write("--------------------------------------------------------\n\n\n")
        result_full_file.flush()
