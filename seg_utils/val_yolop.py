import os
import sys
import torch
import math
import time
import numpy as np
from  torch import nn
from  tqdm import  tqdm
import torch.nn.functional as F
import argparse
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from PIL import Image
from functools import partial
from multiprocessing import Pool
from skimage.morphology import binary_dilation, disk
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, jaccard_score
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import onnxruntime as ort
import torchvision

# from torch.utils.tensorboard import SummaryWriter

# Add the project root to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import specific modules
from Dataloader.Dataset import LabelGenerator, DataLoaderX , InfiniteDataLoader
from Dataloader.Preprocess import cluster_to_embedding_feature
# from   Models.unet_model import UNet as MultiNet
from Models.MultiNet import MultiNet
from   seg_utils.Parameters import Parameters
import seg_utils.losses as Loss
from   seg_utils.accuracy import Accuracy
import seg_utils.utils as utils
from seg_utils.utils import non_max_suppression,cells_to_bboxes,make_grids,plot_image,xywh2xyxy,scale_boxes, mAP 

BOUND_PIXELS = [1, 2, 5]
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
        return (box[2] - box[0]) * (box[3] - box[1]) #(x2-x1)*(y2-y1)

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression_s(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
    
def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


def activation(x):
    return 1e-7 + (1 - 2 * 1e-7) * (0.5 + torch.arctan(x)/torch.tensor(np.pi))
class Evaluator:
    def __init__(self, device, save_path, parameters, build_targets):
        self.device = device
        self.save_path = save_path
        self.parameters = parameters
        self.build_targets = build_targets
        self.activation = activation #torch.sigmoid  # Activation function for output layers

    @torch.inference_mode()
    def evaluate(self, net, dataloader, epoch):

        num_val_batches = len(dataloader)
        self.indices = np.sort(np.random.choice(num_val_batches, 6, replace=False))

        drivable_metrics = {
            "iou_sum": 0,
            "conf_matrix": np.zeros((2, 2), dtype=np.int64),
            "pixel_accuracy": 0,
            "dice_score": 0
        }
        lane_metrics = {
            "f_score_sum": 0,
            "iou_sum": 0,
            "conf_matrix": np.zeros((2, 2), dtype=np.int64),
            "pixel_accuracy": 0
        }

        preds, trues, stats = [], [], []
        ap, ap_class = [], []
        mAP50, mAP75 = 0, 0
        class_accuracy, obj_accuracy = 0, 0
        conf_threshold, nms_iou_thresh = 0.5, 0.3
        anchors = torch.tensor(self.parameters.anchors).float().view(3, -1, 2) / torch.tensor(self.parameters.strides).repeat(6, 1).T.reshape(3, 3, 2)
        iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        total = 0
        true = 0
        
        ort_session = ort.InferenceSession("./artifacts/yolop-640-640.onnx",providers=['CUDAExecutionProvider'])


        with torch.no_grad():
            for index, (images, confidence_mask, cluster_masks, object_annotations) in enumerate(tqdm(dataloader)):
                images = images.to(self.device, dtype=torch.float32, memory_format=torch.channels_last)
                confidence_mask = confidence_mask.to(self.device, dtype=torch.long)
                cluster_masks = cluster_masks.to(self.device, dtype=torch.long)

                # img_rgb = images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to NCHW format for ONNX
                # img_rgb = img_rgb[:, :, :, ::-1]  # Convert to RGB

                # # Resize and normalize images
                # canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb[0], (640, 640))
                # img = canvas.copy().astype(np.float32)  # (3, 640, 640) RGB
                # img /= 255.0
                # img[:, :, 0] -= 0.485
                # img[:, :, 1] -= 0.456
                # img[:, :, 2] -= 0.406
                # img[:, :, 0] /= 0.229
                # img[:, :, 1] /= 0.224
                # img[:, :, 2] /= 0.225

                # img = img.transpose(2, 0, 1)
                # img = np.expand_dims(img, 0)  # (1, 3, 640, 640)

                # Inference
                det_out, da_seg_out, ll_seg_out = ort_session.run(
                    ['det_out', 'drive_area_seg', 'lane_line_seg'],
                    input_feed={"images": images.cpu().numpy()}
                )

                # Convert outputs to tensors
                # print(da_seg_out[0,1,:,:].shape)
                da_seg_mask = da_seg_out[0,1,:,:]#np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
                ll_seg_mask = ll_seg_out[0,1,:,:]#np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)

                # print(da_seg_mask.shape)
                # print(ll_seg_mask.shape)
                # det_out = torch.from_numpy(det_out).float()
                # da_seg_out = torch.from_numpy(da_seg_out).float()
                # ll_seg_out = torch.from_numpy(ll_seg_out).float()

                # Process predictions and calculate metrics
                # (similar processing as in the original code)

                # Convert drivable area and lane line outputs to masks
                pred_drivable =da_seg_mask# self.activation(da_seg_out[0])
                pred_lane =ll_seg_mask# self.activation(ll_seg_out[0])
                gt_drivable = (confidence_mask[:, 0, :, :]).cpu().numpy()
                gt_lane = (confidence_mask[:, 1, :, :]).cpu().numpy()
                pred_drivable = (pred_drivable > 0.5)
                pred_lane = (pred_lane > 0.5)
                # Drivable area metrics
                drivable_metrics["iou_sum"] += jaccard_score(gt_drivable.flatten(), pred_drivable.flatten(),zero_division=1)
                drivable_metrics["conf_matrix"] += confusion_matrix(gt_drivable.flatten(), pred_drivable.flatten(), labels=[0, 1])
                drivable_metrics["pixel_accuracy"] += accuracy_score(gt_drivable.flatten(), pred_drivable.flatten())
                drivable_metrics["dice_score"] += f1_score(gt_drivable.flatten(), pred_drivable.flatten(),zero_division=1.0)
                # Lane metrics
                lane_metrics["f_score_sum"] += utils.eval_lane_per_image(gt_lane, np.expand_dims(pred_lane,0))
                lane_metrics["iou_sum"] += jaccard_score(gt_lane.flatten(), pred_lane.flatten(),zero_division=1)
                lane_metrics["conf_matrix"] += confusion_matrix(gt_lane.flatten(), pred_lane.flatten(), labels=[0, 1])
                lane_metrics["pixel_accuracy"] += accuracy_score(gt_lane.flatten(), pred_lane.flatten())

                # Object detection metrics
                # boxes = non_max_suppression(det_out,iou_threshold=nms_iou_thresh,threshold=conf_threshold,tolist=False,max_detections=300)[0].cpu().numpy().astype(np.float32)
                # if boxes.shape[0] != 0:
                #     boxes[:, 0] -= dw
                #     boxes[:, 1] -= dh
                #     boxes[:, 2] -= dw
                #     boxes[:, 3] -= dh
                #     boxes[:, :4] /= r

                labels = self.build_targets(det_out, object_annotations, pred_size=images.shape[2:4], train=False)
                # class_acc, obj_acc = utils.check_det_accuracy(det_out, labels, conf_threshold)
                # pred_boxes = non_max_suppression(det_out, iou_threshold=nms_iou_thresh, threshold=conf_threshold, tolist=False, max_detections=300).to(self.device)
                true_boxes = cells_to_bboxes(labels, anchors, strides=self.parameters.strides, is_pred=False, to_list=False)
                true_boxes = non_max_suppression(true_boxes, iou_threshold=nms_iou_thresh, threshold=conf_threshold, tolist=False, max_detections=300).to(self.device)
                det_out = torch.from_numpy(det_out).float().to(self.device)
                pred_boxes = non_max_suppression_s(det_out)[0] # [n,6] [x1,y1,x2,y2,conf,cls]
                # boxes = boxes.cpu().numpy().astype(np.float32)
                # if boxes.shape[0] == 0:
                #     print("no bounding boxes detected.")
                #     return

                # # scale coords to original size.
                # boxes[:, 0] -= dw
                # boxes[:, 1] -= dh
                # boxes[:, 2] -= dw
                # boxes[:, 3] -= dh
                # boxes[:, :4] /= r

                # utils.clip_coords(pred_boxes, images.shape[2:])
                iou = utils.box_iou(pred_boxes[:, 0:4], true_boxes[:, 2:])
                label_classes = true_boxes[..., 0]
                pred_classes = pred_boxes[:, 5]
                pred_scores = pred_boxes[:, 4]
                if len(true_boxes):
                    if len(pred_boxes):
                        correct = torch.zeros(pred_boxes.shape[0], niou, dtype=torch.bool, device=self.device)
                        for i, pred_box in enumerate(pred_boxes[:, 0:4]):
                            ious = [utils.iou_width_height(pred_box, label_box.cpu().numpy())
                                    for label_box in true_boxes[:, 2:]]
                            index = int(torch.tensor(ious).argmax())
                            if ious[index] >= iouv[0] and pred_classes[i] == label_classes[index]:
                                correct[i] = ious[index] > iouv
                                true += 1
                        stats.append((correct.cpu(), pred_scores.cpu(), pred_classes.cpu(), label_classes.cpu()))
                    else:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), label_classes.cpu()))
                else:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), label_classes.cpu()))

                # if index in self.indices:
                #     titles = [
                #         "lane gt", "drivable gt", "det gt", "image",
                #         "lane pd", "drivable pd", "det gt", "embedding"
                #     ]

                #     img = images[0].permute(1, 2, 0).cpu().numpy()
                #     if len(true_boxes) > 0:
                #         obj_gt = plot_image(img * 255, true_boxes, self.parameters.bdd, self.parameters.class_mapping)
                #     else:
                #         obj_gt = img
                #     if len(pred_boxes) > 0:
                #         obj_pd = plot_image(img * 255, pred_boxes, self.parameters.bdd, self.parameters.class_mapping)
                #     else:
                #         obj_pd = img
                #     embedding = np.sum(pred_fe[0, :, :, :].cpu().numpy(), axis=0, keepdims=True)[0]
                #     subplots = [gt_lane[0], gt_drivable[0], obj_gt, img, pred_lane[0], pred_drivable[0], obj_pd, embedding]
                #     create_subplots([utils.normalize_to_uint8(subplot) for subplot in subplots], titles, index, epoch, self.save_path)

        # Drivable area results
        num_batches = max(num_val_batches, 1)
        drivable_results = {
            "frequency_iou": drivable_metrics["iou_sum"] / num_batches,
            "mean_iou": drivable_metrics["iou_sum"] / num_batches,
            "pixel_accuracy": drivable_metrics["pixel_accuracy"] / num_batches,
            "dice_score": drivable_metrics["dice_score"] / num_batches
        }

        # Lane results
        lane_results = {
            "f_score": lane_metrics["f_score_sum"] / num_batches,
            "mean_iou": lane_metrics["iou_sum"] / num_batches,
            "pixel_accuracy": lane_metrics["pixel_accuracy"] / num_batches
        }

        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = utils.ap_per_class(*stats)
            ap50, ap70, ap75, ap = ap[:, 0], ap[:, 4], ap[:, 5], ap.mean(1)
            mp, mr, map50, map70, map75, map = p.mean(), r.mean(), ap50.mean(), ap70.mean(), ap75.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=1)
        else:
            nt = torch.zeros(1)
            mp, mr, map50, map70, map75, map = 0,0,0,0,0,0
        mAP_results = {
            "mAP50": map50,
            "mAP70": map70,
            "mAP75": map75,
            "map": map,
            "precision": mp,
            "recall": mr
        }

        return drivable_results, lane_results, mAP_results


def train(args,P):
    save_path = os.path.join(os.pardir,'artifacts/')
    # save_path = "/ghasemzadeh-akramifard/code/artifacts/"
    os.makedirs(save_path, exist_ok=True)
    min_loss = 100
    max_val = 0
    
    # Initialize dataset and dataloader

    val_dataset = LabelGenerator(
        P.val_data_path,
        P.val_label_path,
        save_path,
        image_size=(640, 640),
        normalize=True,
        class_mapping=P.class_mapping,
        train=False,
        transform=False,
    )
    val_dataloader = DataLoaderX(
        val_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=LabelGenerator.collate_fn,
    )

    target_builder = Loss.TargetBuilder(device = args.device)
    accuracy_fn = Accuracy()

    evaluator = Evaluator(
                        device = args.device,
                        save_path = save_path,
                        parameters = P,
                        build_targets = target_builder,
                    )

    drivable_score, lane_score, obj_scores = evaluator.evaluate(
                                                                net = None ,
                                                                dataloader = val_dataloader,
                                                                epoch = 0
                                                                )

    utils.print_metrics_in_rich_table([drivable_score, lane_score, obj_scores], ["Drivable Area Results", "Lane Results", "Object Detection Results"])

    return results, model

if __name__ == "__main__":
    # mp.set_start_method('spawn')
             

    parser = argparse.ArgumentParser(description="Your description here.")
    # parser.add_argument("--data_path", type=str, help="Path to data.")
    # parser.add_argument("--label_path", type=str, help="Path to labels.")
    parser.add_argument("--pre_trained", type=str, default= "/home/kia/Multi-Task-Network/artifacts/best.pt",help="Path to pre trained model.")
    parser.add_argument("--save_path", type=str, default="/home/kia/Multi-Task-Network/" ,help="Path to save results.")
    parser.add_argument("--device",type=str ,default="cuda", help="choose your training device cuda or cpu")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--shuffle", action="store_true",default=False, help="Shuffle the data.")
    parser.add_argument("--optimizer_type" , type=str , default = "sgd" , help="Optimizer Type between adam and sgd")
    parser.add_argument("--version", action="store_true",default="1", help="version of your training")
    args = parser.parse_args()
    P = Parameters()
    # writer_path =  os.path.join(args.save_path,"/runs/1/")
    # os.makedirs(writer_path, exist_ok=True)
    # writer = SummaryWriter(writer_path)
    results , model = train(args,P)