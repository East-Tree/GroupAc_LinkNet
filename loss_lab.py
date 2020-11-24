import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class Focalloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, device0=None, weight=None, attenuate=2.0):
        """
            this is a multi focal loss function base on F.nll_loss
            :param input: [N,p]
            :param target: [N]  ground truth
            :param weight:[p]
            :return:
            """
        if device0 is None:
            device1 = torch.device('cpu')
        else:
            device1 = device0
        input_soft = F.softmax(input, dim=1)
        input_logsoft = F.log_softmax(input, dim=1)
        batch = target.size()[0]
        target_mask = target.reshape(-1, 1)
        input_soft = torch.gather(input_soft, 1, target_mask)
        input_logsoft = torch.gather(input_logsoft, 1, target_mask)
        if weight is None:
            weight_tensor = torch.tensor([1] * batch, device=device1)
        else:
            weight_tensor = weight.repeat(batch, 1).to(device=device1)
            weight_tensor = torch.gather(weight_tensor, 1, target_mask)
        weight_tensor = weight_tensor.reshape(-1, 1)
        focal_weight = weight_tensor * torch.pow(1.0 - input_soft, attenuate)
        # print('focal loss coeff:' + str(focal_weight))
        loss = (-1) * focal_weight * input_logsoft
        loss = torch.mean(loss, dim=0)


        return loss, focal_weight

