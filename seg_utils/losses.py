import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List
import math
import os 
import sys
import numpy as np
# Add the project root to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from seg_utils.Parameters import Parameters

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
        self.BCE = nn.BCELoss(weight, size_average)
    
    def forward(self, inputs, targets):
        # Cast inputs and targets to the same data type (e.g., torch.float32)
        inputs = inputs.to('cuda',torch.float32)
        targets = targets.to('cuda',torch.float32)
        
        # Flatten the inputs and targets to be of shape (batch_size, class_size, -1)
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)
        
        # Compute Binary Cross-Entropy Loss
        bce_loss = self.BCE(inputs, targets)
        
        # Compute Dice Coefficient
        intersection = torch.sum(inputs * targets, dim=(0, 2))
        union = torch.sum(inputs, dim=(0, 2)) + torch.sum(targets, dim=(0, 2))
        dice_coeff = (2.0 * intersection + 1e-5) / (union + 1e-5)
        
        # Compute Dice Loss
        dice_loss = 1.0 - dice_coeff.mean()
        
        # Combine BCE and Dice Loss (you can adjust the weighting factor)
        combined_loss = bce_loss + dice_loss
        
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

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def activation(x):
    return 1e-7 + (1 - 2 * 1e-7) * (0.5 + torch.arctan(x)/torch.tensor(np.pi))


def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score

def calculate_iou(prediction, target):

    intersection = torch.logical_and(prediction, target).sum()
    union = torch.logical_or(prediction, target).sum()
    if target.sum() == 0 :
        if prediction.sum() == 0:
            return 1
    if union == 0 :
        return 0 
    iou = float(intersection) / float(union)
    return iou

def is_parallel(model):
    print(type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class ComputeLoss:
    sort_obj_iou = False
    # Compute losses
    def __init__(self, autobalance=False):

        self.P = Parameters()
        self.Segmentation_Confidence_Loss = nn.BCEWithLogitsLoss(reduction="mean")#DiceBCELoss()
        self.Feature_Embedding_Loss = EmbeddingLoss()
        
        device = "cuda"  # get model device
        h = Parameters()  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(h.cls_pw, device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(h.obj_pw, device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps= 0.0)  # positive, negative BCE targets

        # Focal loss
        g = h.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        self.anchors = torch.tensor(h.anchors).to(device)
        self.na = 3 # number of anchors 
        self.nc = len(h.class_mapping) # number of classes
        self.nl = len(self.anchors)  # number of detection layers
        self.device = device
        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance



    def __call__(self, predictions, targetss):  # predictions, targets
        p = predictions[2]
        targets = targetss[2]
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp.box
        lobj *= self.hyp.obj
        lcls *= self.hyp.clss
        bs = tobj.shape[0]  # batch size

        Pred_Confidence , Pred_EmbeddingFeatureArea = predictions[0],predictions[1]

        GroundTruth_Confidence , GroundTruth_EmbeddingFeatureArea = targetss[0] , targetss[1]
        GroundTruth_Confidence = GroundTruth_Confidence.to(torch.float32)

        SegLoss = self.Segmentation_Confidence_Loss(Pred_Confidence,GroundTruth_Confidence)
        
        dice = dice_loss(activation(Pred_Confidence),GroundTruth_Confidence,multiclass=True)
        AreaFeatureLoss = self.Feature_Embedding_Loss(Pred_EmbeddingFeatureArea,GroundTruth_EmbeddingFeatureArea)
        ObjLoss = (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

        TotalLoss =  self.P.Alpha1*SegLoss + self.P.Alpha2 * dice + self.P.Alpha3 * AreaFeatureLoss  + (lbox + lobj + lcls) * bs

        return TotalLoss, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

# class ComputeLoss(nn.Module) :

#     def __init__(self):
#         super(ComputeLoss, self).__init__()
#         self.P = Parameters()
#         self.Segmentation_Confidence_Loss = nn.BCEWithLogitsLoss(reduction="mean")#DiceBCELoss()
#         self.Feature_Embedding_Loss = EmbeddingLoss()
        
#     def forward(self, model , predictions , targets):
#         lcls, lbox, lobj = torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda")
#         tcls, tbox, indices, anchors = build_targets(predictions[2], targets[2], model)  # targets

#         Pred_Confidence , Pred_EmbeddingFeatureArea = predictions[0],predictions[1]

#         GroundTruth_Confidence , GroundTruth_EmbeddingFeatureArea = targets
#         GroundTruth_Confidence = GroundTruth_Confidence.to(torch.float32)

#         SegLoss = self.Segmentation_Confidence_Loss(Pred_Confidence,GroundTruth_Confidence)
        
#         dice = dice_loss(activation(Pred_Confidence),GroundTruth_Confidence,multiclass=True)
#         AreaFeatureLoss = self.Feature_Embedding_Loss(Pred_EmbeddingFeatureArea,GroundTruth_EmbeddingFeatureArea)

#         TotalLoss =  self.P.Alpha1*SegLoss + self.P.Alpha2 * dice + self.P.Alpha3 * AreaFeatureLoss

#         return TotalLoss

