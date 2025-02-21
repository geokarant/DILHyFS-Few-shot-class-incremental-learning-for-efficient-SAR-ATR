from __future__ import print_function

import torch
import torch.nn as nn

import math
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
""" 
Focal Loss for Dense Object Detection in PyTorch by Tsung-Yi Lin et. al. (2016)
link: https://github.com/clcarwin/focal_loss_pytorch 
"""
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
""" 
A Discriminative Feature Learning Approach for Deep Face Recognition by Wen et al. (2016).
link: https://github.com/KaiyangZhou/pytorch-center-loss
"""

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, device='cuda', lambda_factor=1.0):
        """
        Center Loss implementation.
        Args:
            num_classes (int): Number of classes.
            feature_dim (int): Dimension of the feature space.
            device (str): Device to store the class centers ('cpu' or 'cuda').
            lambda_factor (float): Weighting factor for the Center Loss.
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_factor = lambda_factor
        self.device = device

    def forward(self, features, labels):
        """
        Compute the Center Loss.
        Args:
            features (torch.Tensor): Feature representations of the samples (batch_size, feature_dim).
            labels (torch.Tensor): Ground truth class labels for the samples (batch_size).
        Returns:
            torch.Tensor: The computed center loss.
        """
        batch_size = features.size(0)

        # Initialize centers dynamically based on the batch
        centers = torch.zeros(self.num_classes, self.feature_dim).to(self.device)
        counts = torch.zeros(self.num_classes).to(self.device)

        # Accumulate features and counts for each class
        for feature, label in zip(features, labels):
            centers[label] += feature
            counts[label] += 1

        # Avoid division by zero
        counts = torch.where(counts == 0, torch.ones_like(counts), counts)

        # Compute the mean for each class
        centers = centers / counts.unsqueeze(1)

        # Get the centers corresponding to each label in the batch
        centers_batch = centers[labels]

        # Compute the squared L2 distance between features and their respective centers
        loss = torch.sum((features - centers_batch) ** 2) / 2.0 / batch_size
        return loss