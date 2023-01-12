# -*- coding: utf-8 -*-
import torch as tr
import torch.nn as nn
import torch.nn.functional as F


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * tr.log(input_ + epsilon)
    entropy = tr.sum(entropy, dim=1)
    return entropy


class CELabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CELabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)

        # with mixup, the raw label is already one-hot form
        # targets = tr.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss


class source_inconsistency_loss(nn.Module):
    """
    source models inconsistency loss
    """

    def __init__(self, th_max=0.1):
        super(source_inconsistency_loss, self).__init__()
        self.th_max = th_max

    def forward(self, prob):  # [4, 8, 4]
        si_std = tr.std(prob, dim=1).mean(dim=1).mean(dim=0)
        return si_std


class BatchEntropyLoss(nn.Module):
    """
    Batch-entropy loss
    """

    def __init__(self):
        super(BatchEntropyLoss, self).__init__()

    def forward(self, prob):  # prob: [4, 8, 4]
        batch_entropy = F.softmax(prob, dim=2).mean(dim=0)
        batch_entropy = batch_entropy * (-batch_entropy.log())
        batch_entropy = -batch_entropy.sum(dim=1)
        loss = batch_entropy.mean()
        return loss, batch_entropy


class InstanceEntropyLoss(nn.Module):
    """
    Instance-entropy loss.
    """

    def __init__(self):
        super(InstanceEntropyLoss, self).__init__()

    def forward(self, prob):
        instance_entropy = F.softmax(prob, dim=2) * F.log_softmax(prob, dim=2)
        instance_entropy = -1.0 * instance_entropy.sum(dim=2)
        instance_entropy = instance_entropy.mean(dim=0)
        loss = instance_entropy.mean()
        return loss, instance_entropy
