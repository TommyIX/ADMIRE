# Note: Only used in DSBN/MSKT
import torch

class KnowledgeTransLoss(torch.nn.Module):
    def __init__(self):
        super(KnowledgeTransLoss, self).__init__()

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        # dsc = 2*y_pred*y_true/(y_pred**2 + y_true**2)
        dsc = 2*torch.sum(y_pred)*torch.sum(y_true)/(torch.sum(y_pred**2) + torch.sum(y_true**2))
        return 1. - dsc