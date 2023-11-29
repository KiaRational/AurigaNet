    
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
import os
import sys
import matplotlib.pyplot as plt
import time
from torchvision.ops import nms
import matplotlib.patches as patches
import random
from collections import Counter

# Add the project root to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import specific modules
from   Models.MultiNet import MultiNet
import seg_utils.Parameters as Parameters


def iou_width_height(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box1: List or array-like, representing [x1, y1, x2, y2] of the first box.
    - box2: List or array-like, representing [x1, y1, x2, y2] of the second box.

    Returns:
    - iou: Intersection over Union (IoU) value.
    """

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_intersection = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))

    # Calculate area of intersection rectangle
    area_intersection = x_intersection * y_intersection

    # Calculate area of each bounding box
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate area of union between the boxes
    area_union = area_box1 + area_box2 - area_intersection

    # Calculate IoU
    iou = 0 if area_union == 0 else area_intersection / area_union

    return iou

def non_max_suppression(batch_bboxes, iou_threshold, threshold, max_detections=300, tolist=True):

    """new_bboxes = []
    for box in bboxes:
        if box[1] > threshold:
            box[3] = box[0] + box[3]
            box[2] = box[2] + box[4]
            new_bboxes.append(box)"""

    bboxes_after_nms = []
    for boxes in batch_bboxes:
        boxes = torch.masked_select(boxes, boxes[..., 1:2] > threshold).reshape(-1, 6)

        # from xywh to x1y1x2y2

        boxes[..., 2:3] = boxes[..., 2:3] - (boxes[..., 4:5] / 2)
        boxes[..., 3:4] = boxes[..., 3:4] - (boxes[..., 5:] / 2)
        boxes[..., 5:6] = boxes[..., 5:6] + boxes[..., 3:4]
        boxes[..., 4:5] = boxes[..., 4:5] + boxes[..., 2:3]

        indices = nms(boxes=boxes[..., 2:] + boxes[..., 0:1], scores=boxes[..., 1], iou_threshold=iou_threshold)
        boxes = boxes[indices]

        # sorts boxes by objectness score but it's already done internally by torch metrics's nms
        # _, si = torch.sort(boxes[:, 1], dim=0, descending=True)
        # boxes = boxes[si, :]

        if boxes.shape[0] > max_detections:
            boxes = boxes[:max_detections, :]

        bboxes_after_nms.append(
            boxes.tolist() if tolist else boxes
        )

    return bboxes_after_nms if tolist else torch.cat(bboxes_after_nms, dim=0)


def cells_to_bboxes(predictions, anchors, strides, is_pred=False, to_list=True):
    num_out_layers = len(predictions)
    grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
    anchor_grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
    strides = torch.tensor(strides).to("cuda")
    anchors = anchors.to("cuda")
    all_bboxes = []
    for i in range(num_out_layers):
        bs, naxs, ny, nx, _ = predictions[i].shape
        stride = strides[i]
        grid[i], anchor_grid[i] = make_grids(anchors, naxs, ny=ny, nx=nx, stride=stride, i=i)
        if is_pred:
            # formula here: https://github.com/ultralytics/yolov5/issues/471
            #xy, wh, conf = predictions[i].sigmoid().split((2, 2, 80 + 1), 4)
            layer_prediction = predictions[i].sigmoid()
            obj = layer_prediction[..., 4:5]
            xy = (2 * (layer_prediction[..., 0:2]) + grid[i].to("cuda") - 0.5) * stride
            wh = ((2*layer_prediction[..., 2:4])**2) * anchor_grid[i]
            best_class = torch.argmax(layer_prediction[..., 5:], dim=-1).unsqueeze(-1)

        else:
            predictions[i] = predictions[i].to(config.DEVICE, non_blocking=True)
            obj = predictions[i][..., 4:5]
            xy = (predictions[i][..., 0:2] + grid[i]) * stride
            wh = predictions[i][..., 2:4] * stride
            best_class = predictions[i][..., 5:6]

        scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).reshape(bs, -1, 6)

        all_bboxes.append(scale_bboxes)

    return torch.cat(all_bboxes, dim=1).tolist() if to_list else torch.cat(all_bboxes, dim=1)

def make_grids(anchors, naxs, stride, nx=20, ny=20, i=0):

    x_grid = torch.arange(nx)
    x_grid = x_grid.repeat(ny).reshape(ny, nx)

    y_grid = torch.arange(ny).unsqueeze(0)
    y_grid = y_grid.T.repeat(1, nx).reshape(ny, nx)

    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    xy_grid = xy_grid.expand(1, naxs, ny, nx, 2)
    anchor_grid = (anchors[i]*stride).reshape((1, naxs, 1, 1, 2)).expand(1, naxs, ny, nx, 2)

    return xy_grid, anchor_grid

def plot_image(image, boxes, labels):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = labels
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        bbox = box[2:]

        # FOR MY_NMS attempts, also rect = patches.Rectangle box[2] becomes box[2] - box[0] and box[3] - box[1]
        upper_left_x = max(bbox[0], 0)
        upper_left_x = min(upper_left_x, im.shape[1])
        lower_left_y = max(bbox[1], 0)
        lower_left_y = min(lower_left_y, im.shape[0])

        """upper_left_x = max(box[0] - box[2] / 2, 0)
        upper_left_x = min(upper_left_x, im.shape[1])
        lower_left_y = max(box[1] - box[3] / 2, 0)
        lower_left_y = min(lower_left_y, im.shape[0])"""

        rect = patches.Rectangle(
            (upper_left_x, lower_left_y),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x,
            lower_left_y,
            s=f"{class_labels[int(class_pred)]}: {box[1]:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
    # plt.show()
    return im


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2



def mAP(label, pred, class_num=10):
    # Convert label and pred to numpy arrays
    pred =pred.cpu().numpy()
    label_boxes = label[..., 2:] * 640
    label_boxess = np.zeros_like(label_boxes.numpy())
    label_boxess[:, 0] = label_boxes[..., 0] - label_boxes[..., 2] / 2
    label_boxess[:, 1] = label_boxes[..., 1] - label_boxes[..., 3] / 2
    label_boxess[:, 2] = label_boxes[..., 0] + label_boxes[..., 2] / 2
    label_boxess[:, 3] = label_boxes[..., 1] + label_boxes[..., 3] / 2
    
    label_classes = label[..., 1]
    pred_classes = pred[:, 0]
    pred_boxes = pred[:, 2:]
    pred_scores = pred[:, 1]
    matches = []
    np.set_printoptions(suppress=True)

    # Initialize TPFP50 and TPFP75 arrays
    TPFP50 = np.zeros((class_num, 2))  # |FP|TP|
    TPFP75 = np.zeros((class_num, 2))

    # Count occurrences of predicted classes
    counter = Counter(pred_classes)
    pred_counts = list(counter.items())

    for pred_count in pred_counts:
        TPFP50[int(pred_count[0]), 0] = int(pred_count[1])
        TPFP75[int(pred_count[0]), 0] = int(pred_count[1])


    for i, pred_box in enumerate(pred_boxes):
        ious = [iou_width_height(torch.tensor(pred_box), torch.tensor(label_box))
                for label_box in label_boxess]

        index = np.array(ious).argmax()

        if ious[index] >= 0.5:
            if pred_classes[i] == label_classes[index]:
                TPFP50[int(pred_classes[i]), 1] += 1  # count as TP
                TPFP50[int(pred_classes[i]), 0] -= 1
                if ious[index] < 0.75:
                    matches.append((int(pred_classes[i]), i, index, ious[index]))
        else:
            if pred_classes[i] == label_classes[index]:
                TPFP50[int(pred_classes[i]), 0] += 1  # count as FP

        if ious[index] >= 0.75:
            if pred_classes[i] == label_classes[index]:
                TPFP75[int(pred_classes[i]), 1] += 1
                TPFP75[int(pred_classes[i]), 0] -= 1
                matches.append((int(pred_classes[i]), i, index, ious[index]))
        else:
            if pred_classes[i] == label_classes[index]:
                TPFP75[int(pred_classes[i]), 0] += 1  # count as FP
    return TPFP50, TPFP75

