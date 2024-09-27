    
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
from rich.console import Console
from rich.table import Table
from skimage.morphology import binary_dilation, disk

# Add the project root to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import specific modules
from   Models.MultiNet import MultiNet
import seg_utils.Parameters as Parameters


BOUND_PIXELS = [1, 2, 5]

def activation(x):
    return 1e-7 + (1 - 2 * 1e-7) * (0.5 + torch.arctan(x)/torch.tensor(np.pi))

def eval_lane_per_threshold(gt_mask, pd_mask, bound_pix):
    batch_size = gt_mask.shape[0]
    f_scores = []

    for i in range(batch_size):
        gt_dil = binary_dilation(gt_mask[i], disk(bound_pix))
        pd_dil = binary_dilation(pd_mask[i], disk(bound_pix))

        gt_match = gt_mask[i] * pd_dil
        pd_match = pd_mask[i] * gt_dil

        n_gt = np.sum(gt_mask[i])
        n_pd = np.sum(pd_mask[i])

        if n_pd == 0 and n_gt > 0:
            precision = 1
            recall = 0
        elif n_pd > 0 and n_gt == 0:
            precision = 0
            recall = 1
        elif n_pd == 0 and n_gt == 0:
            precision = 1
            recall = 1
        else:
            precision = np.sum(pd_match) / float(n_pd)
            recall = np.sum(gt_match) / float(n_gt)

        if precision + recall == 0:
            f_score = 0.0
        else:
            f_score = 2.0 * precision * recall / (precision + recall)

        f_scores.append(f_score)

    return np.mean(f_scores)
def eval_lane_per_image(gt_mask, pd_mask):
    f_scores = [eval_lane_per_threshold(gt_mask, pd_mask, bp) for bp in BOUND_PIXELS]
    return np.mean(f_scores)

def print_metrics_in_rich_table(metrics_dicts, titles):
    console = Console()

    for metrics_dict, title in zip(metrics_dicts, titles):
        table = Table(title=title)
        columns = list(metrics_dict.keys())
        for column in columns:
            table.add_column(column, justify="center")
        

        values = [str(float(metrics_dict[column])) for column in columns]
        table.add_row(*values)
        
        console.print(table)
def normalize_to_uint8(array):
    # If the array contains boolean values, set min_val to 0 and max_val to 1
    if np.issubdtype(array.dtype, np.bool_):
        min_val = 0
        max_val = 1
    else:
        # Find the minimum and maximum values in the array, ignoring NaNs
        min_val = np.nanmin(array)
        max_val = np.nanmax(array)
    
    # Scale the array values to the range [0, 255]
    scaled_array = (array - min_val) * (255 / (max_val - min_val+ 1e-10))
    
    # Clip values to ensure they are within the range [0, 255]
    scaled_array = np.clip(scaled_array, 0, 255)
    
    # Convert the scaled array to uint8
    uint8_array = scaled_array.astype(np.uint8)
    
    return uint8_array

def create_subplots(images, titles, index , epoch , save_path):
    """
    Creates a subplot image with the specified images and titles using OpenCV.
    
    :param images: A list of 7 images to display in the subplots.
    :param titles: A list of 7 titles corresponding to the subplots.
    :return: An OpenCV image of the subplots.
    """
    assert len(images) == 8, "There should be exactly 7 images."
    assert len(titles) == 8, "There should be exactly 7 titles."

    # Define the size of each image and the border size
    img_height, img_width = 160, 160
    border_size = 10
    title_height = 30

    # Define the size of the overall subplot image
    subplot_height = (img_height + title_height + border_size) * 2 + border_size
    subplot_width = (img_width + border_size) * 4 + border_size

    # Create a blank canvas for the subplot image
    subplot_img = np.ones((subplot_height, subplot_width, 3), dtype=np.uint8) * 255

    # Positions for each subplot in the grid
    positions = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3) 
    ]

    # Add each image to the subplot canvas
    for idx, (image, title) in enumerate(zip(images, titles)):
        row, col = positions[idx]
        y1 = row * (img_height + title_height + border_size) + title_height + border_size
        y2 = y1 + img_height
        x1 = col * (img_width + border_size) + border_size
        x2 = x1 + img_width
        
        # Check if the image is grayscale and convert it to three channels if needed
        
        if title == 'embedding':
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

        else:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize the image to fit the grid cell
        resized_image = cv2.resize(image, (img_width, img_height))
        subplot_img[y1:y2, x1:x2] = resized_image
        
        # Add the title above the image
        title_y = row * (img_height + title_height + border_size) + border_size
        cv2.putText(subplot_img, title, (x1 + 5, title_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(f"{save_path}/{epoch}_result_{index}.png",subplot_img)

    return subplot_img

def iou_width_height_anch(gt_box, anchors, strided_anchors=True, stride=[8, 16, 32]):
    """
    Parameters:
        gt_box (tensor): width and height of the ground truth box
        anchors (tensor): lists of anchors containing width and height
        strided_anchors (bool): if the anchors are divided by the stride or not
    Returns:
        tensor: Intersection over union between the gt_box and each of the n-anchors
    """
    # boxes 1 (gt_box): shape (2,)
    # boxes 2 (anchors): shape (9,2)
    # intersection shape: (9,)
    anchors /= 640
    if strided_anchors:

        anchors = anchors.reshape(9, 2) * torch.tensor(stride).repeat(6, 1).T.reshape(9, 2)

    intersection = torch.min(gt_box[..., 0], anchors[..., 0]) * torch.min(
        gt_box[..., 1], anchors[..., 1]
    )
    union = (
        gt_box[..., 0] * gt_box[..., 1] + anchors[..., 0] * anchors[..., 1] - intersection
    )
    # intersection/union shape (9,)
    return intersection / union


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
        bs,naxs, ny, nx, _ = predictions[i].shape
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
            predictions[i] = predictions[i].to("cuda", non_blocking=True)
            obj = predictions[i][..., 4:5]
            xy = (predictions[i][..., 0:2] + grid[i].to("cuda")) * stride
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
def plot_image(image, boxes, labels , class_dict):
    """Plots predicted bounding boxes on the image"""
    cmap = cv2.applyColorMap(np.arange(0, 256, 256 // len(labels), dtype=np.uint8), cv2.COLORMAP_JET)
    class_labels = labels
    colors = [tuple(map(int, cmap[i, 0, ::-1])) for i in range(len(labels))]
    im = np.array(image)

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = int(box[0])
        bbox = box[2:]
        confidence = box[1]
        x1, y1, x2, y2 = map(int, box[2:])
        # Calculate upper-left corner of the bounding box
        upper_left_x = max(x1, 0)
        upper_left_y = max(y1, 0)
        
        # Calculate lower-right corner of the bounding box
        lower_right_x = min(x2, im.shape[1])
        lower_right_y = min(y2, im.shape[0])
        color = colors[class_pred]
        cv2.rectangle(im, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), color, 2)

        # Put class label and confidence score
        label = f"{labels[class_pred]}: {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(im, (upper_left_x, upper_left_y - text_height - baseline), (upper_left_x + text_width, upper_left_y), color, -1)
        cv2.putText(im, label, (upper_left_x, upper_left_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[..., 0].clamp_(0, img_shape[1])  # x1
    boxes[..., 1].clamp_(0, img_shape[0])  # y1
    boxes[..., 2].clamp_(0, img_shape[1])  # x2
    boxes[..., 3].clamp_(0, img_shape[0])  # y2
    
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



def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint", GIoU=False, SIOU=False, eps=1e-7):
    """
    Calculates intersection over union (IoU) with optional GIoU and SIOU.
    
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): "midpoint" (x, y, w, h) or "corners" (x1, y1, x2, y2)
        GIoU (bool): If True, compute GIoU (Generalized IoU)
        SIOU (bool): If True, compute SIOU (Scaled IoU)
        eps (float): Small value for numerical stability

    Returns:
        tensor: IoU, GIoU, or SIOU for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    else:
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    inter = (torch.min(box1_x2, box2_x2) - torch.max(box1_x1, box2_x1)).clamp(0) * \
            (torch.min(box1_y2, box2_y2) - torch.max(box1_y1, box2_y1)).clamp(0)

    w1, h1 = box1_x2 - box1_x1, box1_y2 - box1_y1
    w2, h2 = box2_x2 - box2_x1, box2_y2 - box2_y1
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU:
        cw = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)  # convex width
        ch = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)  # convex height
        c_area = cw * ch + eps  # convex area
        giou = iou - (c_area - union) / c_area  # GIoU
        return giou

    if SIOU:
        cw = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)  # convex width
        ch = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)  # convex height
        s_cw = (box2_x1 + box2_x2 - box1_x1 - box1_x2) * 0.5
        s_ch = (box2_y1 + box2_y2 - box1_y1 - box1_y2) * 0.5
        sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2) + eps
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = np.sqrt(2) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)

        angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - np.pi / 4), 2)

        rho_x = (s_cw / (cw + eps)) ** 2
        rho_y = (s_ch / (ch + eps)) ** 2
        gamma = 2 - angle_cost
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)

        siou = (iou + 0.5 * (distance_cost + shape_cost))
        return siou

    return iou

def mAP(label, pred, class_num=10):
    print(label.shape, pred.shape)
    correct_class = 0
    correct_obj = 0
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
    CONFMTX = np.zeros((class_num+1, class_num+1)) 
    CONFMTX_Totals = np.zeros((class_num+1, class_num+1))
    # Count occurrences of predicted classes
    counter = Counter(pred_classes)
    pred_counts = list(counter.items())
    for pred_count in pred_counts:
        TPFP50[int(pred_count[0]), 0] = int(pred_count[1])
        TPFP75[int(pred_count[0]), 0] = int(pred_count[1])

    tot_class , tot_obj = len(label_classes) , len(label_boxes)

    for i , box in enumerate(label_classes):
        CONFMTX_Totals[int(label_classes[i]),:-1] += 1

    for i, pred_box in enumerate(pred_boxes):
        ious = [iou_width_height(torch.tensor(pred_box), torch.tensor(label_box))
                for label_box in label_boxess]

        index = np.array(ious).argmax()

        if ious[index] >= 0.5:
            correct_obj += 1
            if pred_classes[i] == label_classes[index]:
                correct_class += 1
                CONFMTX[int(pred_classes[i]),int(label_classes[index])] += 1
                TPFP50[int(pred_classes[i]), 1] += 1  # count as TP
                TPFP50[int(pred_classes[i]), 0] -= 1
                if ious[index] < 0.75:
                    matches.append((int(pred_classes[i]), i, index, ious[index]))
            else:
                CONFMTX[int(pred_classes[i]),int(label_classes[index])] += 1
        
        else:
            if pred_classes[i] == label_classes[index]:
                TPFP50[int(pred_classes[i]), 0] += 1  # count as FP
                CONFMTX[int(pred_classes[i]),int(label_classes[index])] += 1
            else:
                CONFMTX[int(pred_classes[i]),int(label_classes[index])] += 1
        
        if ious[index] >= 0.75:
            if pred_classes[i] == label_classes[index]:
                TPFP75[int(pred_classes[i]), 1] += 1
                TPFP75[int(pred_classes[i]), 0] -= 1
                matches.append((int(pred_classes[i]), i, index, ious[index]))
        
        else:
            if pred_classes[i] == label_classes[index]:
                TPFP75[int(pred_classes[i]), 0] += 1  # count as FP

    class_acc = (correct_class / (tot_class + 1e-16))
    obj_acc   = (correct_obj / (tot_obj + 1e-16))
    CONFMTX = CONFMTX/(CONFMTX_Totals + 1e-16)
    print(matches)
    return TPFP50 , TPFP75 , class_acc , obj_acc , CONFMTX

def plot_confusion_matrix(conf_matrix, classes, save_path=None):

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues , vmin= 0)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes)+1)

    plt.xticks(tick_marks, classes+["no object"], rotation=45)
    plt.yticks(tick_marks, classes+ ["not detected"])
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    for i in range(len(classes)+1):
        for j in range(len(classes)+1):
            plt.text(j, i, str(conf_matrix[i, j])[:5], ha='center', va='center', color='red', fontsize=10)

    if save_path:
        plt.savefig(save_path+"confmtx.png")
        plt.close()
        # print(f"Confusion matrix plot saved at {save_path}")



def check_det_accuracy(y,out,conf_threshold):
    tot_class_preds, correct_class = 0, 0
    tot_obj, correct_obj = 0, 0

    for i in range(3):
        obj = y[i][..., 4] == 1  # in paper this is Iobj_i

        correct_class += torch.sum(
            torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
        )
        tot_class_preds += torch.sum(obj)

        obj_preds = torch.sigmoid(out[i][..., 0]) > conf_threshold
        correct_obj += torch.sum(obj_preds[obj] == y[i][..., 4][obj])
        tot_obj += torch.sum(obj)

    class_accuracy = (correct_class / (tot_class_preds + 1e-16))
    obj_accuracy = (correct_obj / (tot_obj + 1e-16))

    return class_accuracy,obj_accuracy


def ball_kernel(Z, X, kappa, metric='cosine'):
    """ Computes pairwise ball kernel (without normalizing constant)
        (note this is kernel as defined in non-parametric statistics, not a kernel as in RKHS)

        @param Z: a [n x d] torch.FloatTensor of NORMALIZED datapoints - the seeds
        @param X: a [m x d] torch.FloatTensor of NORMALIZED datapoints - the points

        @return: a [n x m] torch.FloatTensor of pairwise ball kernel computations,
                 without normalizing constant
    """
    if metric == 'euclidean':
        distance = Z.unsqueeze(1) - X.unsqueeze(0)
        distance = torch.norm(distance, dim=2)
        kernel = torch.exp(-kappa * torch.pow(distance, 2))
    elif metric == 'cosine':
        kernel = torch.exp(kappa * torch.mm(Z, X.t()))
    return kernel


def get_label_mode(array):
    """ Computes the mode of elements in an array.
        Ties don't matter. Ties are broken by the smallest value (np.argmax defaults)

        @param array: a numpy array
    """
    labels, counts = np.unique(array, return_counts=True)
    mode = labels[np.argmax(counts)].item()
    return mode


def connected_components(Z, epsilon, metric='cosine'):
    """
        For the connected components, we simply perform a nearest neighbor search in order:
            for each point, find the points that are up to epsilon away (in cosine distance)
            these points are labeled in the same cluster.

        @param Z: a [n x d] torch.FloatTensor of NORMALIZED datapoints

        @return: a [n] torch.LongTensor of cluster labels
    """
    n, d = Z.shape

    K = 0
    cluster_labels = torch.ones(n, dtype=torch.long) * -1
    for i in range(n):
        if cluster_labels[i] == -1:

            if metric == 'euclidean':
                distances = Z.unsqueeze(1) - Z[i:i + 1].unsqueeze(0)  # a are points, b are seeds
                distances = torch.norm(distances, dim=2)
            elif metric == 'cosine':
                distances = 0.5 * (1 - torch.mm(Z, Z[i:i+1].t()))
            component_seeds = distances[:, 0] <= epsilon

            # If at least one component already has a label, then use the mode of the label
            if torch.unique(cluster_labels[component_seeds]).shape[0] > 1:
                temp = cluster_labels[component_seeds].numpy()
                temp = temp[temp != -1]
                label = torch.tensor(get_label_mode(temp))
            else:
                label = torch.tensor(K)
                K += 1  # Increment number of clusters

            cluster_labels[component_seeds] = label

    return cluster_labels


def seed_hill_climbing_ball(X, Z, kappa, max_iters=10, metric='cosine'):
    """ Runs mean shift hill climbing algorithm on the seeds.
        The seeds climb the distribution given by the KDE of X

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
        @param dist_threshold: parameter for the ball kernel
    """
    n, d = X.shape
    m = Z.shape[0]

    for _iter in range(max_iters):

        # Create a new object for Z
        new_Z = Z.clone()

        W = ball_kernel(Z, X, kappa, metric=metric)

        # use this allocated weight to compute the new center
        new_Z = torch.mm(W, X)  # Shape: [n x d]

        # Normalize the update
        if metric == 'euclidean':
            summed_weights = W.sum(dim=1)
            summed_weights = summed_weights.unsqueeze(1)
            summed_weights = torch.clamp(summed_weights, min=1.0)
            Z = new_Z / summed_weights
        elif metric == 'cosine':
            Z = F.normalize(new_Z, p=2, dim=1)

    return Z


def mean_shift_with_seeds(X, Z, kappa, max_iters=10, metric='cosine'):
    """ Runs mean-shift

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
        @param dist_threshold: parameter for the von Mises-Fisher distribution
    """

    Z = seed_hill_climbing_ball(X, Z, kappa, max_iters=max_iters, metric=metric)

    # Connected components
    cluster_labels = connected_components(Z, 2 * 0.02, metric=metric)  # Set epsilon = 0.1 = 2*alpha

    return cluster_labels, Z


def select_smart_seeds(X, num_seeds, return_selected_indices=False, init_seeds=None, num_init_seeds=None, metric='cosine'):
    """ Selects seeds that are as far away as possible

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param num_seeds: number of seeds to pick
        @param init_seeds: a [num_seeds x d] vector of initial seeds
        @param num_init_seeds: the number of seeds already chosen.
                               the first num_init_seeds rows of init_seeds have been chosen already

        @return: a [num_seeds x d] matrix of seeds
                 a [n x num_seeds] matrix of distances
    """
    n, d = X.shape
    selected_indices = -1 * torch.ones(num_seeds, dtype=torch.long)

    # Initialize seeds matrix
    if init_seeds is None:
        seeds = torch.empty((num_seeds, d), device=X.device)
        num_chosen_seeds = 0
    else:
        seeds = init_seeds
        num_chosen_seeds = num_init_seeds

    # Keep track of distances
    distances = torch.empty((n, num_seeds), device=X.device)

    if num_chosen_seeds == 0:  # Select first seed if need to
        selected_seed_index = np.random.randint(0, n)
        selected_indices[0] = selected_seed_index
        selected_seed = X[selected_seed_index, :]
        seeds[0, :] = selected_seed
        if metric == 'euclidean':
            distances[:, 0] = torch.norm(X - selected_seed.unsqueeze(0), dim=1)
        elif metric == 'cosine':
            distances[:, 0] = 0.5 * (1 - torch.mm(X, selected_seed.unsqueeze(1))[:,0])  
        num_chosen_seeds += 1
    else:  # Calculate distance to each already chosen seed
        for i in range(num_chosen_seeds):
            if metric == 'euclidean':
                distances[:, i] = torch.norm(X - seeds[i:i+1, :], dim=1)
            elif metric == 'cosine':
                distances[:, i] = 0.5 * (1 - torch.mm(X, seeds[i:i+1, :].t())[:, 0])

    # Select rest of seeds
    for i in range(num_chosen_seeds, num_seeds):
        # Find the point that has the furthest distance from the nearest seed
        distance_to_nearest_seed = torch.min(distances[:, :i], dim=1)[0]  # Shape: [n]
        selected_seed_index = torch.argmax(distance_to_nearest_seed)
        selected_indices[i] = selected_seed_index
        selected_seed = torch.index_select(X, 0, selected_seed_index)[0, :]
        seeds[i, :] = selected_seed

        # Calculate distance to this selected seed
        if metric == 'euclidean':
            distances[:, i] = torch.norm(X - selected_seed.unsqueeze(0), dim=1)
        elif metric == 'cosine':
            distances[:, i] = 0.5 * (1 - torch.mm(X, selected_seed.unsqueeze(1))[:,0])

    return_tuple = (seeds,)
    if return_selected_indices:
        return_tuple += (selected_indices,)
    return return_tuple


def mean_shift_smart_init(X, kappa, num_seeds=100, max_iters=10, metric='cosine'):
    """ Runs mean shift with carefully selected seeds

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param dist_threshold: parameter for the von Mises-Fisher distribution
        @param num_seeds: number of seeds used for mean shift clustering

        @return: a [n] array of cluster labels
    """

    n, d = X.shape
    seeds, selected_indices = select_smart_seeds(X, num_seeds, return_selected_indices=True, metric=metric)
    seed_cluster_labels, updated_seeds = mean_shift_with_seeds(X, seeds, kappa, max_iters=max_iters, metric=metric)

    # Get distances to updated seeds
    if metric == 'euclidean':
        distances = X.unsqueeze(1) - updated_seeds.unsqueeze(0)  # a are points, b are seeds
        distances = torch.norm(distances, dim=2)
    elif metric == 'cosine':
        distances = 0.5 * (1 - torch.mm(X, updated_seeds.t())) # Shape: [n x num_seeds]

    # Get clusters by assigning point to closest seed
    closest_seed_indices = torch.argmin(distances, dim=1)  # Shape: [n]
    cluster_labels = seed_cluster_labels[closest_seed_indices]

    # assign zero to the largest cluster
    num = len(torch.unique(seed_cluster_labels))
    count = torch.zeros(num, dtype=torch.long)
    for i in range(num):
        count[i] = (cluster_labels == i).sum()
    label_max = torch.argmax(count)
    if label_max != 0:
        index1 = cluster_labels == 0
        index2 = cluster_labels == label_max
        cluster_labels[index1] = label_max
        cluster_labels[index2] = 0

    return cluster_labels, selected_indices
    
def clustering_features(features, num_seeds=100):
    metric = 'euclidean'
    height = features.shape[2]
    width = features.shape[3]
    out_label = torch.zeros((features.shape[0], height, width))

    # mean shift clustering
    kappa = 20
    selected_pixels = []

    for j in range(features.shape[0]):
        X = torch.from_numpy(features[j]).view(features.shape[1], -1)
        X = torch.transpose(X, 0, 1)
        cluster_labels, selected_indices = mean_shift_smart_init(X, kappa=kappa, num_seeds=num_seeds, max_iters=10, metric=metric)
        out_label[j] = cluster_labels.view(height, width)
        selected_pixels.append(selected_indices)
    return out_label, selected_pixels

def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='precision-recall_curve.png', names=[]):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)  # sorted index from big to small
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes, each number just showed up once
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros((unique_classes.shape[0], 1000)), np.zeros((unique_classes.shape[0], 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and (j == 0):
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    i = r.mean(0).argmax()

    if plot:
        plot_pr_curve(px, py, ap, save_dir, names)

    return p[:, i], r[:, i], ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[..., 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = general.box_iou(labels[:, 1:], detections[..., :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
            else:
                self.matrix[gc, self.nc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[self.nc, dc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FN'] if labels else "auto",
                       yticklabels=names + ['background FP'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
