import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.Parameters import Parameters

'''
  _                         __                      _    _                    
 | |                       / _|                    | |  (_)                   
 | |      ___   ___  ___  | |_  _   _  _ __    ___ | |_  _   ___   _ __   ___ 
 | |     / _ \ / __|/ __| |  _|| | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
 | |____| (_) |\__ \\__ \ | |  | |_| || | | || (__ | |_ | || (_) || | | |\__ \
 |______|\___/ |___/|___/ |_|   \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
                                                                              
'''

class EmbeddingLoss(nn.Module):

    def __init__(self):

        super(EmbeddingLoss, self).__init__()

        self.p = Parameters()

    def forward(self, feature, ground_truth_instance):

        real_batch_size = feature.size(0)
        
        F_SIM_i = feature.view(real_batch_size, self.p.feature_size, 1, self.p.grid_y * self.p.grid_x)
        F_SIM_i = F_SIM_i.expand(real_batch_size, self.p.feature_size, self.p.grid_y * self.p.grid_x, self.p.grid_y * self.p.grid_x).detach()
        
        F_SIM_j = feature.view(real_batch_size, self.p.feature_size, self.p.grid_y * self.p.grid_x, 1)
        F_SIM_j = F_SIM_j.expand(real_batch_size, self.p.feature_size, self.p.grid_y * self.p.grid_x, self.p.grid_y * self.p.grid_x)
        
        Norm = (F_SIM_i - F_SIM_j) ** 2
        Norm = torch.norm(Norm, dim=1).view(real_batch_size, 1, self.p.grid_y * self.p.grid_x, self.p.grid_y * self.p.grid_x)

        # Calculate same instance loss (Loss_Instance_1)
        Loss_Instance_1 = torch.sum(Norm[ground_truth_instance == 1]) / torch.sum(ground_truth_instance == 1)

        # Calculate different instance loss (Loss_Instance_0)

        Loss_Instance_0 = torch.max(torch.zeros_like(Norm[ground_truth_instance == 2]), self.p.K1 - Norm[ground_truth_instance == 2])

        Loss_Instance_0 = torch.sum(Loss_Instance_0) / torch.sum(ground_truth_instance == 2)

        # for one class and no class

        if math.isnan(Loss_Instance_0):

            Loss_Instance_0 = 0

        if math.isnan(Loss_Instance_1):

            Loss_Instance_1 = 0

        Loss = Loss_Instance_1 + Loss_Instance_0

        return Loss


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Apply softmax along the class dimension (dim=1)
        inputs = F.softmax(inputs, dim=1)

        # Flatten label and prediction tensors
        inputs_flat = inputs.view(inputs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # Calculate Dice loss
        intersection = (inputs_flat * targets_flat).sum(dim=1)
        dice_loss = 1 - (2 * intersection + smooth) / (inputs_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth)
        dice_loss = dice_loss.mean()  # Average over the batch

        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets.unsqueeze(1), reduction='mean')

        # Combine Dice and BCE losses
        combined_loss = dice_loss + bce_loss

        return combined_loss



class BoundaryLoss1(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """
    def __init__(self, theta0=5, theta=11):
        super(BoundaryLoss1, self).__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-batch
        """  
        pred = torch.softmax(pred, dim=1)
        one_hot_gt = self.one_hot(gt, pred.size(1))

        gt_b = self.boundary_map(one_hot_gt, self.theta0)
        pred_b = self.boundary_map(1 - pred, self.theta0)

        gt_b_ext = self.boundary_map(gt_b, self.theta)
        pred_b_ext = self.boundary_map(1 - pred_b, self.theta)

        P, R, BF1 = self.compute_metrics(pred_b, gt_b_ext, pred_b_ext, gt_b)

        loss = torch.mean(1 - BF1)

        return loss

    def boundary_map(self, mask, kernel_size):
        boundary = F.max_pool2d(1 - mask, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        boundary -= 1 - mask
        return boundary

    def compute_metrics(self, pred_b, gt_b_ext, pred_b_ext, gt_b):
        eps = 1e-7
        P = (torch.sum(pred_b * gt_b_ext, dim=2) + eps) / (torch.sum(pred_b, dim=2) + eps)
        R = (torch.sum(pred_b_ext * gt_b, dim=2) + eps) / (torch.sum(gt_b, dim=2) + eps)
        BF1 = (2 * P * R + eps) / (P + R + eps)
        return P, R, BF1

    def one_hot(self, tensor, num_classes):
        one_hot = torch.zeros(tensor.size(0), num_classes, *tensor.size()[1:]).to(tensor.device)
        one_hot.scatter_(1, tensor.unsqueeze(1), 1)
        return one_hot


class ComputeLoss(nn.Module) :

    def __init__(self):
        super(ComputeLoss, self).__init__()
        self.P = Parameters()
        self.Segmentation_Confidence_Loss = DiceBCELoss()
        self.Feature_Embedding_Loss = EmbeddingLoss()
        
    def forward(self, predictions , targets):

        Pred_Confidence , Pred_EmbeddingFeatureArea , Pred_EmbeddingFeatureLane = predictions

        GroundTruth_Confidence , GroundTruth_EmbeddingFeatureArea , GroundTruth_EmbeddingFeatureLane = targets



        SegLoss = self.Segmentation_Confidence_Loss(Pred_Confidence,GroundTruth_Confidence)

        AreaFeatureLoss = self.Feature_Embedding_Loss(Pred_EmbeddingFeatureArea,GroundTruth_EmbeddingFeatureArea)

        LaneFeatureLoss = self.Feature_Embedding_Loss(Pred_EmbeddingFeatureLane,GroundTruth_EmbeddingFeatureLane)


        TotalLoss = self.P.Alpha1 * SegLoss + self.P.Alpha2 * AreaFeatureLoss + self.P.Alpha3 * LaneFeatureLoss

        return TotalLoss
        
        

