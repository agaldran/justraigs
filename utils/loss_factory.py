import torch
import warnings

class CompoundLoss(torch.nn.Module):
    def __init__(self, loss1, loss2=None, alpha1=1., alpha2=0.):
        super(CompoundLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, y_pred, y_true):
        l1 = self.loss1(y_pred, y_true)
        if self.alpha2 == 0 or self.loss2 is None:
            return self.alpha1*l1
        l2 = self.loss2(y_pred, y_true)
        return self.alpha1*l1 + self.alpha2 * l2

class BCELoss(torch.nn.Module):
    def __init__(self, bce_ls=0.):
        super(BCELoss, self).__init__()
        self.bce_ls = bce_ls
        self.loss = torch.nn.BCEWithLogitsLoss()
    def forward(self, y_pred, y_true):
        if self.bce_ls>0:
            y_true[y_true == 1] = 1 - self.bce_ls
            y_true[y_true == 0] = self.bce_ls
        return self.loss(y_pred, y_true)

class BrierLoss(torch.nn.Module):
    def __init__(self):
        super(BrierLoss, self).__init__()
        self.loss = torch.nn.MSELoss()
    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)

class F1Loss(torch.nn.Module):
    def __init__(self, smooth=1., include_background=False):
        super(F1Loss, self).__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, y_pred, y_true):
        y_pred = y_pred.sigmoid()

        prec = (torch.sum(torch.multiply(y_pred, y_true)) + self.smooth)/(torch.sum(y_pred) + self.smooth)
        sens = (torch.sum(torch.multiply(y_pred, y_true)) + self.smooth)/(torch.sum(y_true) + self.smooth)
        return 1. - 2.0 * (prec*sens)/(prec+sens)


def get_loss(loss1, loss2=None, alpha1=1., alpha2=0., bce_ls=0.):
    if loss1 == loss2 and alpha2 != 0.:
        warnings.warn('using same loss twice, you sure?')
    loss_dict = dict()
    loss_dict['bce'] = BCELoss(bce_ls)
    loss_dict['brier'] = BrierLoss()
    loss_dict['f1'] = F1Loss()
    loss_dict['bcef1'] = CompoundLoss(BCELoss(), F1Loss(), alpha1=1., alpha2=1.)
    loss_dict[None] = None

    loss_fn = CompoundLoss(loss_dict[loss1], loss_dict[loss2], alpha1, alpha2)

    return loss_fn
