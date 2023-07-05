# 运行参数 ----------------------------------------------------------------------------
device = "cuda"  # 可选择："cuda"为NVIDIA显卡，"mps"为Apple ARM架构芯片，其他情况下请使用 "cpu"

# 训练参数 ----------------------------------------------------------------------------
epoch = 20  # 运行时代数
batch_size = 1  # 运行批次大小(1为单张图片训练，接受大batch_size)
lr = 0.001  # 学习率

do_train = True  # 为False时，模型仅执行测试

use_dsp_CAT = True  # 是否使用下采样CAT模型，有助于提高训练速度
dsp_CAT_scale = 4  # 下采样CAT模型的缩放比例
dsp_stop_epoch = 20  # 下采样CAT模型停止使用的epoch

snake_type = 'circle'  # 初始化蛇的形状，'circle'或者'square'
snake_init_scale = 0.8  # 蛇的直径占据边长的比例

use_located_snake_init = True  # 是否通过mapE定位进行蛇初始化 (wyk蛇定位代码)
located_init_start_epoch = 0   # 开始使用定位蛇初始化的epoch

# ACM参数 ----------------------------------------------------------------------------
L = 200  # 蛇算法采样点数量

# 自适应演化参数
adaptive_ACM_mode = 'yes'  # 'no'表示不使用，'train_only'仅训练使用，'test_only'仅测试使用，'yes'保持常开
ACM_iteration_base = 50  # 自适应ACM基础演化次数
max_ACM_reiter = 20  # 自适应ACM最多重试演化次数

# 常规演化参数
ACM_iterations = 300  # Emap-ACM 蛇演化次数
CAT_Sharpness = 3  # 3.7在测试中是一个比较好的参数，适当增大锐度有助于提高性能
ACM_paramset = {
    "CAT_forceweight": 1,  # CAT力场的权重
    "delta_s": 1.8,  # Emap-ACM delta_s 参量
    "max_pixel_move": 2,  # Emap-ACM 最大允许运行长度
    "gamma": 2.2  # Emap-ACM gamma 参量
}

# 数据读取 ----------------------------------------------------------------------------
data_loadmode = 'npy'  # concat version does not support folder mode
image_size = 128  # 图片尺寸

divide_mode = 'ff-seq'  # 'old'为最开始的按顺序划分图片的方式，'ff-seq'为顺序五折，'ff-rad'为随机打乱五折，'no'为完全不划分 全用于训练
shuffle_after_combine = True  # 是否在合并数据集后打乱数据集，种子为ffrad_seed

ff_fold_num = 0  # 使用五折划分的情况下，当前程序运行的五折折数，0-4
ffrad_seed = 233  # 随机打乱五折情况下的种子

# Load multiple npy files and preprocess them
npy_dir = [[r"C:/Users/jhong/Documents/Datasets/ADMcontour1.0/ADM_images_128.npy",
           r"C:/Users/jhong/Documents/Datasets/ADMcontour1.0/ADM_contour_200.npy"],
           [r"C:/Users/jhong/Documents/Datasets/MRSpine1.1_remove_bad_images/MRSpineSeg_images_128.npy",
           r"C:/Users/jhong/Documents/Datasets/MRSpine1.1_remove_bad_images/MRSpineSeg_contour_200.npy"],
           [r"C:/Users/jhong/Documents/Datasets/VerSe_MICCAI_1.0/Verse20_MICCAI_images_128.npy",
            r"C:/Users/jhong/Documents/Datasets/VerSe_MICCAI_1.0/Verse20_MICCAI_contour_200.npy"]]

if data_loadmode is not 'npy':
    raise NotImplementedError("Mixture training only support npy mode right now.")

dataset_names = ['MR_AVBCE', 'MRSpine', 'Verse']  # Names of multiple dataset, required for specific display

# 模型权重读取 -------------------------------------------------------------------------
resume_training = False  # 是否加载保存的权重，继续训练
load_ckpt_dir = './checkpoints/ADMIRE_model_19.pth'

# 保存结果 ----------------------------------------------------------------------------
result_save_rule = 'data'   # 或者'img'，img为在result文件夹中保存所有图片数据，data直接保存训练时输出的图片等等，可以使用imviewer来查看
                            # 备注：img模式下的保存规则已移除，每一张都会保存

im_save_path = "./result/"  # 保存训练效果图片或者训练数据(根据save_rule)的路径
model_save_path = "./checkpoints/"  # 保存模型权重文件的路径
draw_force_field = True  # 在保存结果中加画CAT力场图与合力图

# Inaccessable parameters for this training mode ------------------------------------
folder_dir = r"./MR_AVBCE_dataset"
image_num = 4601