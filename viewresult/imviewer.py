#coding:utf-8
'''
imviewer.py
读取指定的pickle文件，画图并以单独文件的方式保存
适用于无GUI环境，在允许情况下可以使用ipynb notebook方式查看
'''

# 目标文件，图片与保存路径
pklfile_dir = '纯模型测试1/result/ADMIRE_resultdata_epoch_19_Test_data.pkl'
save_dir = pklfile_dir[:-4]

ims = [range(0,200)]
# 备注：ims支持range表示，会自动转换，如[range(2,5),8,10]这样的输入是允许的

import os
import pickle
from process.mapplot import plot_result

os.makedirs(save_dir, exist_ok=True)

save_imlist = []
for m in ims:
    if type(m) == range:
        save_imlist += list(m)
    else:
        save_imlist.append(m)
load_datalist = pickle.load(open(pklfile_dir, 'rb'))
print("读取数据文件：", pklfile_dir)
print("正在保存的图片序号：", save_imlist)

for i in save_imlist:
    datainfo = load_datalist[i]
    draw_force_field = True
    if datainfo['gx'] is None: draw_force_field = False

    plot_result(save_dir + '/', datainfo['iou'], datainfo['epoch'], datainfo['imnum'], datainfo['status'],
                datainfo['snake_result'], datainfo['snake_result_list'], datainfo['GTContour'],
                datainfo['mapE'], datainfo['mapA'], datainfo['mapB'],
                datainfo['image'], plot_force=draw_force_field,
                gx=datainfo['gx'], gy=datainfo['gy'], Fu=datainfo['Fu'], Fv=datainfo['Fv'], compressed_hist=True)

print("所需数据列表的结果已保存到：", save_dir)