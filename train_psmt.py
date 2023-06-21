# coding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# 防止forrtl: error (200)代替KeyboardInterrupt，保证中断时可以保存权重

import torch
import signal
import pickle
import numpy as np

from config import *  # config.py里的所有变量都可以像写在这个py文件里一样，直接去用。
from dataset import build_dataloader
from models.UNet_encdec import UNet_encoder, UNet_decoder
from models.CAT import CATkernel
from models.CCQLoss import CCQLoss
from models.PolyProcess import draw_poly_fill, batch_mask_convert
from process.mapplot import plot_result
from process.mapprocess import map_normalization
from process.metrics import FBound_metric, WCov_metric
from process.MTprocess import update_ema_variables, consistency_weight, get_confidence_weight
from process.snake_iterate import snake_handler
from torchvision.transforms import Resize

# Ctrl+C处理，中断时保存数据
def emergency_stop(signum, frame):
    global force_stop
    force_stop = True
    print("捕获到Ctrl+C，正在保存当前权重")
signal.signal(signal.SIGINT, emergency_stop)
signal.signal(signal.SIGTERM, emergency_stop)

result_file = open("result.txt", "a", encoding="utf-8")
result_full_file = open("result_full.txt", "a", encoding="utf-8")

map_cons_loss = torch.nn.MSELoss()

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
model_encoder = UNet_encoder()
model_decoder = UNet_decoder()
ema_teacher_model1_encoder = UNet_encoder()
ema_teacher_model1_decoder = UNet_decoder()
ema_teacher_model2_encoder = UNet_encoder()
ema_teacher_model2_decoder = UNet_decoder()

for param in ema_teacher_model1_encoder.parameters():
    param.detach_()
for param in ema_teacher_model1_decoder.parameters():
    param.detach_()
for param in ema_teacher_model2_encoder.parameters():
    param.detach_()
for param in ema_teacher_model2_decoder.parameters():
    param.detach_()

resume_epoch = -1
iter_num = 0
if resume_training:
    load_ckpt_dir_decoder = load_ckpt_dir_encoder.replace("modelenc", "modeldec")
    load_ckpt_dir_ema1_encoder = load_ckpt_dir_encoder.replace("modelenc", "modelenc_ema1")
    load_ckpt_dir_ema1_decoder = load_ckpt_dir_encoder.replace("modeldec", "modeldec_ema1")
    load_ckpt_dir_ema2_encoder = load_ckpt_dir_encoder.replace("modelenc", "modelenc_ema2")
    load_ckpt_dir_ema2_decoder = load_ckpt_dir_encoder.replace("modeldec", "modeldec_ema2")

    model_encoder = pickle.load(open(load_ckpt_dir_encoder, 'rb'))
    model_decoder = pickle.load(open(load_ckpt_dir_decoder, 'rb'))
    ema_teacher_model1_encoder = pickle.load(open(load_ckpt_dir_encoder, 'rb'))
    ema_teacher_model1_decoder = pickle.load(open(load_ckpt_dir_decoder, 'rb'))
    ema_teacher_model2_encoder = pickle.load(open(load_ckpt_dir_encoder, 'rb'))
    ema_teacher_model2_decoder = pickle.load(open(load_ckpt_dir_decoder, 'rb'))

    resume_epoch = int(load_ckpt_dir_encoder.split('/')[-1].split('.')[0].split('_')[2])
    if len(load_ckpt_dir_encoder.split('/')[-1].split('.')[0].split('_'))>3:
        print("模型权重加载成功(从上次中断的模型加载，从当前epoch重新训练)，从epoch %d继续训练" % resume_epoch)
        resume_epoch -= 1
    else:
        print("模型权重加载成功，从epoch %d继续训练" % resume_epoch)
    iter_num = resume_epoch * len(dataloader_train)

device = torch.device(device)
optimizer1 = torch.optim.Adam(model_encoder.parameters(), lr=lr)
optimizer2 = torch.optim.Adam(model_decoder.parameters(), lr=lr)
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer1, T_max=epoch)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer2, T_max=epoch)
force_stop = False

model_encoder.train()  # model为使用有标注数据训练的模型，而ema_model为使用无标注数据训练的
model_encoder.to(device)
model_decoder.train()
model_decoder.to(device)
ema_teacher_model1_encoder.train()
ema_teacher_model1_encoder.to(device)
ema_teacher_model1_decoder.train()
ema_teacher_model1_decoder.to(device)
ema_teacher_model2_encoder.train()
ema_teacher_model2_encoder.to(device)
ema_teacher_model2_decoder.train()
ema_teacher_model2_decoder.to(device)

consistency_w_func = consistency_weight(
    final_w=3.0,  # PSMT示例代码中config中的unsupervised_w
    iters_per_epoch=len(dataloader_train),
    rampup_starts=0,
    rampup_ends=epoch,
    ramp_type="cosine_rampup"
)

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
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            image, contour = data
            image = image.to(device)
            contour = contour.to(device)
            batch_shape = image.shape[0]

            if batch_shape - labeled_batch_size <= 0 and train_status:  # 如果无标注数据不足一个batch，直接跳过本次循环
                continue

            if train_status:
                iter_num += 1

            if i >= mean_teacher_epochs:  # 时代数大于mean_teacher_epochs才开始用无标签数据。
                image1 = image.clone()
                noise = torch.clamp(torch.randn_like(image1) * 0.1, -3, 3)
                teacher_model_inputs = image1 + noise  # 不用标签的数据加上个噪声，构建教师模型的输入。

            labeled_image = image[:labeled_batch_size,:,:,:]  # 后续lb对应的这些，这些数据是用标签的。
            unlabeled_image = image[labeled_batch_size:,:,:,:]  # 后续ulb对应的这些，这些数据是不用标签的，作为无标签数据。

            if i >= mean_teacher_epochs:  # 训练模式，PSMT
                # Teacher 1 模型输出
                ema1_midparams = ema_teacher_model1_encoder(teacher_model_inputs)
                ema1_mapEo, ema1_mapAo, ema1_mapBo = ema_teacher_model1_decoder(ema1_midparams)
                # Teacher 2 模型输出
                ema2_midparams = ema_teacher_model2_encoder(teacher_model_inputs)
                ema2_mapEo, ema2_mapAo, ema2_mapBo = ema_teacher_model2_decoder(ema2_midparams)

                ema1_midparams_labeled = []
                ema1_midparams_unlabeled = []
                ema2_midparams_labeled = []
                ema2_midparams_unlabeled = []
                for param in ema1_midparams:
                    ema1_midparams_labeled.append(param[:labeled_batch_size])
                    ema1_midparams_unlabeled.append(param[labeled_batch_size:])
                for param in ema2_midparams:
                    ema2_midparams_labeled.append(param[:labeled_batch_size])
                    ema2_midparams_unlabeled.append(param[labeled_batch_size:])

            if train_status:
                if i >= mean_teacher_epochs:
                    # Student模型输出(有标注部分)
                    midparams_lb = model_encoder(labeled_image)
                    # Student模型输出(无标注部分)
                    midparams_ulb = model_encoder(unlabeled_image)
                else: # 训练模式，只使用有标注图片
                    midparams_lb = model_encoder(labeled_image)
                    batch_shape = labeled_batch_size
            else: # 测试模式下，全部数据做测试
                midparams_lb = model_encoder(image)

            # 获取到batch_shape 后做一下标题输出
            modetitle = 'Train ' if train_status else 'Test '
            if batch_shape > 1:
                print(modetitle + "Epoch %d, Batch %d" % (i, j), end='')
            else:
                print(modetitle + "Epoch %d, Image %d" % (i, j), end='')


            if use_PSMT_T_VAT and i >= mean_teacher_epochs and train_status:  # 测试模式下也不算了
                mapEo_lb, mapAo_lb, mapBo_lb = model_decoder(midparams_lb, vat_models=[ema_teacher_model1_decoder, ema_teacher_model2_decoder],
                                                    vat_datas=[ema1_midparams_labeled, ema2_midparams_labeled])
                if train_status:
                    mapEo_ulb, mapAo_ulb, mapBo_ulb = model_decoder(midparams_ulb, vat_models=[ema_teacher_model1_decoder, ema_teacher_model2_decoder],
                                                                    vat_datas=[ema1_midparams_unlabeled, ema2_midparams_unlabeled])
            else:
                # 如果不计算T-VAT，则使用以下的decoder代码
                mapEo_lb, mapAo_lb, mapBo_lb = model_decoder(midparams_lb)
                if train_status and i >= mean_teacher_epochs:
                    mapEo_ulb, mapAo_ulb, mapBo_ulb = model_decoder(midparams_ulb)

            if i >= mean_teacher_epochs:
                ema_mapEo = (ema1_mapEo + ema2_mapEo) / 2  # 这三个值对应PS-MT论文中的公式(4)，对应伪标签
                ema_mapBo = (ema1_mapBo + ema2_mapBo) / 2
                ema_mapAo = (ema1_mapAo + ema2_mapAo) / 2

            with torch.no_grad():
                mapE_lb = map_normalization(mapEo_lb, labeled_batch_size if train_status else batch_shape) * 12
                mapA_lb = map_normalization(mapAo_lb, labeled_batch_size if train_status else batch_shape)
                mapB_lb = map_normalization(mapBo_lb, labeled_batch_size if train_status else batch_shape)

                if train_status and i>= mean_teacher_epochs:
                    mapE_ulb = map_normalization(mapEo_ulb, batch_shape - labeled_batch_size) * 12
                    mapA_ulb = map_normalization(mapAo_ulb, batch_shape - labeled_batch_size)
                    mapB_ulb = map_normalization(mapBo_ulb, batch_shape - labeled_batch_size)

                if i >= mean_teacher_epochs:
                    ema1_mapE = map_normalization(ema1_mapEo,batch_shape) * 12
                    ema1_mapB = map_normalization(ema1_mapBo,batch_shape)
                    ema1_mapA = map_normalization(ema1_mapAo,batch_shape)
                    ema2_mapE = map_normalization(ema2_mapEo,batch_shape) * 12
                    ema2_mapB = map_normalization(ema2_mapBo,batch_shape)
                    ema2_mapA = map_normalization(ema2_mapAo,batch_shape)

                    # ema1和ema2的数据进行融合
                    ema_mapE = (ema1_mapE + ema2_mapE) / 2
                    ema_mapB = (ema1_mapB + ema2_mapB) / 2
                    ema_mapA = (ema1_mapA + ema2_mapA) / 2

                    ema_mapE_predict = torch.abs(torch.tensor(12.0, dtype=torch.float32, device=device) - ema_mapE)
                    ema_mapB_predict = torch.abs(torch.tensor(1.0, dtype=torch.float32, device=device) - ema_mapB)
                    ema_mapA_predict = torch.abs(torch.tensor(1.0, dtype=torch.float32, device=device) - ema_mapA)
                    ema_mapE_onehot_p = torch.where(ema_mapE_predict > ema_mapE_predict.mean(), torch.ones_like(ema_mapE_predict), torch.zeros_like(ema_mapE_predict))
                    ema_mapB_onehot_p = torch.where(ema_mapB_predict > ema_mapB_predict.mean(), torch.ones_like(ema_mapB_predict), torch.zeros_like(ema_mapB_predict))
                    ema_mapA_onehot_p = torch.where(ema_mapA_predict > ema_mapA_predict.mean(), torch.ones_like(ema_mapA_predict), torch.zeros_like(ema_mapA_predict))

            # 原模型蛇演化
            snake_result, snake_result_list, batch_gx1, batch_gy1, batch_Fu, batch_Fv = \
                snake_handler(i, mapE_lb, mapA_lb, mapB_lb, labeled_batch_size if train_status else batch_shape,
                              [Mx, My], device, [Mx_dsp, My_dsp], CAT_dsp, CAT_usp)  # 学生模型，用标签的数据，做了蛇演化

            if i >= mean_teacher_epochs:
                # 半监督模型蛇演化，但是忽然有点没明白，为什么还要把教师模型的mapE A B也用来演化蛇呢？--要用教师模型的演化结果计算指标。
                ema_snake_result, ema_snake_result_list, ema_batch_gx1, ema_batch_gy1, ema_batch_Fu, ema_batch_Fv = \
                    snake_handler(i, ema_mapE, ema_mapA, ema_mapB, batch_shape, [Mx, My], device, [Mx_dsp, My_dsp],
                                    CAT_dsp, CAT_usp)  # 教师模型，有噪声的、不用标签的数据，做了蛇演化。

                # 无标注部分蛇演化
                if train_status:
                    ulb_snake_result, ulb_snake_result_list, ulb_batch_gx1, ulb_batch_gy1, ulb_batch_Fu, ulb_batch_Fv = \
                        snake_handler(i, mapE_ulb, mapA_ulb, mapB_ulb, batch_shape - labeled_batch_size, [Mx, My], device,
                                      [Mx_dsp, My_dsp], CAT_dsp, CAT_usp)  # 学生模型，不用标签的数据，做了蛇演化


            # 计算损失、参数更新
            batch_mask = batch_mask_convert(contour, [image_size, image_size])

            if train_status:
                # PSMT1-Supervised Part：有标记数据正常训练student
                supervised_loss = CCQLoss.apply(mapEo_lb, mapAo_lb, mapBo_lb, snake_result, contour, image_size,
                                                labeled_batch_size, batch_mask)  # 用标签的数据，监督损失，这个是一直都有的。

                # PSMT2-Consistency Loss
                if i >= mean_teacher_epochs:
                    consistency_loss_sup = torch.mean(get_confidence_weight(ema_mapE_onehot_p[:labeled_batch_size], ema_mapE_predict[:labeled_batch_size]).data * map_cons_loss(mapEo_lb, ema_mapE[:labeled_batch_size].data / 12) +\
                        get_confidence_weight(ema_mapA_onehot_p[:labeled_batch_size], ema_mapA_predict[:labeled_batch_size]).data * map_cons_loss(mapAo_lb, ema_mapA[:labeled_batch_size].data / 1) +\
                        get_confidence_weight(ema_mapB_onehot_p[:labeled_batch_size], ema_mapB_predict[:labeled_batch_size]).data * map_cons_loss(mapBo_lb, ema_mapB[:labeled_batch_size].data / 1))

                    consistency_loss_unsup = torch.mean(get_confidence_weight(ema_mapE_onehot_p[labeled_batch_size:], ema_mapE_predict[labeled_batch_size:]).data * map_cons_loss(mapEo_ulb, ema_mapE[labeled_batch_size:].data / 12) +\
                        get_confidence_weight(ema_mapA_onehot_p[labeled_batch_size:], ema_mapA_predict[labeled_batch_size:]).data * map_cons_loss(mapAo_ulb, ema_mapA[labeled_batch_size:].data / 1) +\
                        get_confidence_weight(ema_mapB_onehot_p[labeled_batch_size:], ema_mapB_predict[labeled_batch_size:]).data * map_cons_loss(mapEo_ulb, ema_mapB[labeled_batch_size:].data / 1))

                    unsupervised_loss = consistency_w_func(epoch=i - mean_teacher_epochs, curr_iter=iter_num) * (consistency_loss_sup + consistency_loss_unsup)
                    # unsupervised_loss = get_current_consistency_weight(i) * (consistency_loss_sup * (labeled_batch_size / batch_shape) + consistency_loss_unsup * ((batch_shape - labeled_batch_size) / batch_shape))
                else:
                    unsupervised_loss = torch.tensor(0.0)

                total_loss = supervised_loss + beta_for_Lcon * unsupervised_loss  # PS-MT论文中公式(1)
                total_loss.backward()
                optimizer1.step()
                optimizer2.step()
                scheduler1.step()
                scheduler2.step()

                # PSMT4-基于Student模型的参数，以EMA的方式更新一个Teacher模型
                # PSMT论文中为了增加多样性，在这里是随机选择一个Teacher模型用ema方式进行更新的
                if np.random.random() > 0.5:
                    update_ema_variables(model_encoder, ema_teacher_model1_encoder, ema_decay, iter_num)
                    update_ema_variables(model_decoder, ema_teacher_model1_decoder, ema_decay, iter_num)
                else:
                    update_ema_variables(model_encoder, ema_teacher_model2_encoder, ema_decay, iter_num)
                    update_ema_variables(model_decoder, ema_teacher_model2_decoder, ema_decay, iter_num)

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
                                        contour[m,:,:].cpu().numpy(), ema_mapE[m,0,:,:].cpu().numpy(),
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
                        print(", unsupervised loss: %.2f\n" % (unsupervised_loss.item()), end='')
                    else:
                        print("\n", end='')


            if force_stop:
                pickle.dump(model_encoder, open(model_save_path + 'ADMIRE_modelenc_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j+1)*batch_size), 'wb'))
                pickle.dump(model_decoder, open(model_save_path + 'ADMIRE_modeldec_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j+1)*batch_size), 'wb'))
                pickle.dump(ema_teacher_model1_encoder, open(model_save_path + 'ADMIRE_ema_model1enc_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j+1)*batch_size), 'wb'))
                pickle.dump(ema_teacher_model1_decoder, open(model_save_path + 'ADMIRE_ema_model1dec_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j+1)*batch_size), 'wb'))
                pickle.dump(ema_teacher_model2_encoder, open(model_save_path + 'ADMIRE_ema_model2enc_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j+1)*batch_size), 'wb'))
                pickle.dump(ema_teacher_model2_decoder, open(model_save_path + 'ADMIRE_ema_model2dec_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j+1)*batch_size), 'wb'))
                pickle.dump(save_datalist, open(im_save_path + 'ADMIRE_resultdata_epoch_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j+1)*batch_size), 'wb'))
                pickle.dump(save_datalist_ema, open(im_save_path + 'ADMIRE_resultdata_ema_epoch_%d_forcestop_%s_batchim_%d.pkl' % (i, modetitle.strip(), (j+1)*batch_size), 'wb'))
                print("(*)中断权重文件和结果数据已保存")
                exit(500)

        pickle.dump(save_datalist, open(im_save_path + 'ADMIRE_resultdata_epoch_%d_%s_data.pkl' % (i, modetitle.strip()), 'wb'))
        pickle.dump(save_datalist_ema, open(im_save_path + 'ADMIRE_resultdata_ema_epoch_%d_%s_data.pkl' % (i, modetitle.strip()), 'wb'))
        save_datalist.clear()
        save_datalist_ema.clear()

        if train_status:  # 结束训练后保存模型checkpoint
            pickle.dump(model_encoder, open(model_save_path + 'ADMIRE_modelenc_%d.pkl' % i, 'wb'))
            pickle.dump(model_decoder, open(model_save_path + 'ADMIRE_modeldec_%d.pkl' % i, 'wb'))
            pickle.dump(ema_teacher_model1_encoder, open(model_save_path + 'ADMIRE_modelenc_ema1_%d.pkl' % i, 'wb'))
            pickle.dump(ema_teacher_model1_decoder, open(model_save_path + 'ADMIRE_modeldec_ema1_%d.pkl' % i, 'wb'))
            pickle.dump(ema_teacher_model2_encoder, open(model_save_path + 'ADMIRE_modelenc_ema2_%d.pkl' % i, 'wb'))
            pickle.dump(ema_teacher_model2_decoder, open(model_save_path + 'ADMIRE_modeldec_ema2_%d.pkl' % i, 'wb'))
            print("(*)权重文件已保存")

        result_file.write("--------------------------------------------------------\n")
        if train_status:
            result_file.write("Train")
            result_full_file.write("Train")
        else:
            result_file.write("Test")
            result_full_file.write("Test")
        result_file.write(" Epoch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (i, sum(full_iou)/len(full_iou), sum(full_dice)/len(full_dice), sum(full_mbf)/len(full_mbf), sum(full_wcov)/len(full_wcov)))
        result_full_file.write(" Epoch %d, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (
        i, sum(full_iou) / len(full_iou), sum(full_dice) / len(full_dice), sum(full_mbf) / len(full_mbf),
        sum(full_wcov) / len(full_wcov)))

        if i>=mean_teacher_epochs:
            result_file.write(
                "EMA Status:, Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f, unsupervised loss: %.2f\n" % (
                sum(ema_ioulist) / len(ema_ioulist), sum(ema_dicelist) / len(ema_dicelist),
                sum(ema_boundflist) / len(ema_boundflist), sum(ema_wcovlist) / len(ema_wcovlist), unsupervised_loss.item()))
            result_full_file.write("EMA Status: Avg IoU: %.4f, Avg Dice: %.4f, Avg mBF: %.4f, Avg WCov: %.4f\n" % (sum(full_ema_iou)/len(full_ema_iou), sum(full_ema_dice)/len(full_ema_dice), sum(full_ema_mbf)/len(full_ema_mbf), sum(full_ema_wcov)/len(full_ema_wcov)))
        result_file.write("--------------------------------------------------------\n\n\n")
        result_full_file.flush()
