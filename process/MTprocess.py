import torch
import numpy as np
from process import PSMTramps
from config import consistency, consistency_rampup

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        # ema_param.data.mul_(alpha).add_(param.data, *, 1 - alpha)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

class consistency_weight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """

    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(PSMTramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup


# def semi_ce_loss(inputs, targets,
#                  conf_mask=True, threshold=None,
#                  threshold_neg=None, temperature_value=1):
#     # target => logit, input => logit
#     pass_rate = {}
#     if conf_mask:
#         # for negative
#         targets_prob = F.softmax(targets / temperature_value, dim=1)
#
#         # for positive
#         targets_real_prob = F.softmax(targets, dim=1)
#
#         weight = targets_real_prob.max(1)[0]
#         total_number = len(targets_prob.flatten(0))
#         boundary = ["< 0.1", "0.1~0.2", "0.2~0.3",
#                     "0.3~0.4", "0.4~0.5", "0.5~0.6",
#                     "0.6~0.7", "0.7~0.8", "0.8~0.9",
#                     "> 0.9"]
#
#         rate = [torch.sum((torch.logical_and((i - 1) / 10 < targets_real_prob, targets_real_prob < i / 10)) == True)
#                 / total_number for i in range(1, 11)]
#
#         max_rate = [torch.sum((torch.logical_and((i - 1) / 10 < weight, weight < i / 10)) == True)
#                     / weight.numel() for i in range(1, 11)]
#
#         pass_rate["entire_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, rate)]
#         pass_rate["max_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, max_rate)]
#
#         mask = (weight >= threshold)
#
#         mask_neg = (targets_prob < threshold_neg)
#
#         # temp negative label * mask_neg, which mask down the positive labels.
#         # neg_label = torch.ones(targets.shape, dtype=targets.dtype, device=targets.device) * mask_neg
#         neg_label = torch.nn.functional.one_hot(torch.argmax(targets_prob, dim=1)).type(targets.dtype)
#         if neg_label.shape[-1] != 19:
#             neg_label = torch.cat((neg_label, torch.zeros([neg_label.shape[0], neg_label.shape[1],
#                                                            neg_label.shape[2], 19 - neg_label.shape[-1]]).cuda()),
#                                   dim=3)
#         neg_label = neg_label.permute(0, 3, 1, 2)
#         neg_label = 1 - neg_label
#
#         if not torch.any(mask):
#             neg_prediction_prob = torch.clamp(1 - F.softmax(inputs, dim=1), min=1e-7, max=1.)
#             negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))
#             # zero = torch.tensor(0., dtype=torch.float, device=negative_loss_mat.device)
#             return inputs.sum() * .0, pass_rate, negative_loss_mat[mask_neg].mean()
#         else:
#             positive_loss_mat = F.cross_entropy(inputs, torch.argmax(targets, dim=1), reduction="none")
#             positive_loss_mat = positive_loss_mat * weight
#
#             neg_prediction_prob = torch.clamp(1 - F.softmax(inputs, dim=1), min=1e-7, max=1.)
#             negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))
#
#             return positive_loss_mat[mask].mean(), pass_rate, negative_loss_mat[mask_neg].mean()
#     else:
#         raise NotImplementedError

# 这个是计算学生模型无监督部分损失\omega(w)的实现，我自己搞得
def get_confidence_weight(y_onehot, y_pred, tau_confidence = 0.5):
    y_pred_confidence = y_pred / y_pred.max()
    param1 = y_onehot.transpose(2,3) * y_pred_confidence
    param2 = torch.where(param1 > tau_confidence, torch.ones_like(param1), torch.zeros_like(param1))
    for i in range(param1.shape[0]):
        param1[i, 0, :, :] = torch.mm(param1[i, 0, :, :], param2[i, 0, :, :])

    return param1

# 老版本的一致性损失：
# consistency_loss = torch.mean((mapEo_lb - ema_mapEo[:labeled_batch_size]) ** 2 + \
#                               (mapAo_lb - ema_mapAo[:labeled_batch_size]) ** 2 + \
#                               (mapBo_lb - ema_mapBo[:labeled_batch_size]) ** 2) + \
#                    torch.mean((mapEo_ulb - ema_mapEo[labeled_batch_size:]) ** 2 + \
#                               (mapAo_ulb - ema_mapAo[labeled_batch_size:]) ** 2 + \
#                               (mapBo_ulb - ema_mapBo[labeled_batch_size:]) ** 2)

