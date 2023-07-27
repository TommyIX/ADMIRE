'''
CCQLoss.py
Jinhong Wang
'''

import torch
from models.PolyProcess import GTpoly, draw_poly, derivatives_poly

class CCQLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mapE, mapA, mapB, snake, thisGT, image_shape, batch_size, batch_mask):
        grads_arrayE = torch.zeros_like(mapE).cpu()
        grads_arrayA = torch.zeros_like(mapA).cpu()
        grads_arrayB = torch.zeros_like(mapB).cpu()

        for i in range(batch_size):
            grads_arrayE[i, 0, :, :] -= GTpoly(snake[i,:,:], [image_shape, image_shape], 2, batch_mask[i,:,:]*255)
            grads_arrayE[i, 0, :, :] += draw_poly(thisGT[i,:,:], 5, [image_shape, image_shape], 2)
            der1, der2 = derivatives_poly(snake[i,:,:])  # 预测的蛇轮廓上，每一点的一阶和二阶导数模长（模长是考虑u和v两个方向的导数计算的）
            der1_GT, der2_GT = derivatives_poly(thisGT[i,:,:].cpu().numpy())  # 这个是用金标准蛇计算的一阶和二阶导数
            # grads_arrayA[i, 0, :, :] -= (draw_poly(snake[i,:,:], der1, [image_shape, image_shape], 2) - draw_poly(thisGT[i,:,:], der1_GT, [image_shape, image_shape], 2))
            grads_arrayB[i, 0, :, :] -= (draw_poly(snake[i,:,:], der2, [image_shape, image_shape], 2) - draw_poly(thisGT[i,:,:], der2_GT, [image_shape, image_shape], 2))

        device = mapE.device
        grads_arrayE = grads_arrayE.to(device)
        grads_arrayA = grads_arrayA.to(device)
        grads_arrayB = grads_arrayB.to(device)
        total_loss = (torch.sum(grads_arrayE*mapE) + torch.sum(grads_arrayA*mapA) + torch.sum(grads_arrayB*mapB))/batch_size
        ctx.save_for_backward(grads_arrayE, grads_arrayA, grads_arrayB)
        return total_loss

    @staticmethod
    def backward(ctx, total_loss):
        gradE, gradA, gradB = ctx.saved_tensors
        return gradE, gradA, gradB, None, None, None, None, None
