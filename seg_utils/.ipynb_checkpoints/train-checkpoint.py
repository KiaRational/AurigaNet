import os
import sys
import torch
import math
import time
import requests
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
# Define the register_nan_hooks function
def register_nan_hooks(model):
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            for idx, out in enumerate(output):
                if torch.isnan(out).any():
                    print(f"NaN detected in output of layer: {module} (output {idx})")
                    raise RuntimeError(f"NaN detected in output of layer: {module} (output {idx})")
        else:
            if torch.isnan(output).any():
                print(f"NaN detected in output of layer: {module}")
                raise RuntimeError(f"NaN detected in output of layer: {module}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Module):
            module.register_forward_hook(forward_hook)
def train_step(model, dataloader, loss_fn, accuracy_fn, optimizer, grad_scaler, device, epoch, num_warmup, Parameters, lf, arguments, last_opt_step, accumulate, SegOnly=False, grad_clip_freq=5):

    total_loss         = 0
    total_cls_loss     = 0
    total_iou_drivable = 0
    total_iou_lane     = 0
    total_bbx_loss     = 0
    total_obj_loss     = 0
    total_det_loss     = 0 
    total_steps = len(dataloader)
    nbs = total_steps
    lf = lambda x: ((1 + math.cos(x * math.pi / Parameters.epoch_number)) / 2) * (1 - Parameters.lrf) + Parameters.lrf  # cosine

    for i, (images, confidence_mask, cluster_masks, object_annotations) in enumerate(tqdm(dataloader)):
        ni = i + nbs * epoch  # number integrated batches (since train start)

        if ni < num_warmup:
            # Warm up
            xi = [0, num_warmup]
            accumulate = max(10, np.interp(ni, xi, [1, nbs / arguments.batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [Parameters.warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [Parameters.warmup_momentum, Parameters.momentum])

        if SegOnly:
            targets = [confidence_mask.to(device), cluster_masks.to(device)]
        else:
            targets = [confidence_mask.to(device), cluster_masks.to(device), object_annotations]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device):
            inputs = images.to(device)
            outputs = model(inputs)
            loss, obj_loss = loss_fn(outputs, targets, pred_size=images.shape[2:4])
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"NaN loss encountered at epoch {epoch}, step {i} , loss_obj{obj_loss[0],obj_loss[1],obj_loss[2]} , seg{obj_loss[3]} , embedding loss{obj_loss[4]}")
                continue
            
            total_loss += loss.item()

        # Backpropagation
        grad_scaler.scale(loss).backward()
        
        # Gradient clipping at intervals
        if (i + 1) % grad_clip_freq == 0:
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        grad_scaler.step(optimizer)
        grad_scaler.update()

        total_det_loss += obj_loss[0] + obj_loss[1] + obj_loss[2]
        total_cls_loss += obj_loss[2]
        total_bbx_loss += obj_loss[0]
        total_obj_loss += obj_loss[1]

    avg_total_loss = total_loss / len(dataloader)
    avg_det_loss = total_det_loss / len(dataloader)
    avg_bbx_loss = total_bbx_loss / len(dataloader)
    avg_obj_loss = total_obj_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)

    return model, avg_total_loss, avg_det_loss, avg_bbx_loss, avg_obj_loss, avg_cls_loss, optimizer, last_opt_step, accumulate
# def train_step(model, dataloader, loss_fn, accuracy_fn, optimizer, grad_scaler, device, epoch, num_warmup, Parameters, lf, arguments, last_opt_step, accumulate, SegOnly=False, grad_clip_freq=10):

#     total_loss         = 0
#     total_cls_loss     = 0
#     total_iou_drivable = 0
#     total_iou_lane     = 0
#     total_bbx_loss     = 0
#     total_obj_loss     = 0
#     total_det_loss     = 0 
#     total_steps = len(dataloader)
#     nbs = total_steps
#     lf = lambda x: ((1 + math.cos(x * math.pi / Parameters.epoch_number)) / 2) * (1 - Parameters.lrf) + Parameters.lrf  # cosine

#     for i, (images, confidence_mask, cluster_masks, object_annotations) in enumerate(dataloader):
#         ni = i + nbs * epoch  # number integrated batches (since train start)

#         if ni < num_warmup:
#             # Warm up
#             xi = [0, num_warmup]
#             accumulate = max(10, np.interp(ni, xi, [1, nbs / arguments.batch_size]).round())
#             for j, x in enumerate(optimizer.param_groups):
#                 x['lr'] = np.interp(ni, xi, [Parameters.warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
#                 if 'momentum' in x:
#                     x['momentum'] = np.interp(ni, xi, [Parameters.warmup_momentum, Parameters.momentum])

#         if SegOnly:
#             targets = [confidence_mask.to(device), cluster_masks.to(device)]
#         else:
#             targets = [confidence_mask.to(device), cluster_masks.to(device), object_annotations]

#         optimizer.zero_grad(set_to_none=True)

#         try:
#             with torch.amp.autocast(device):
#                 inputs = images.to(device)
#                 outputs = model(inputs)
#                 loss, obj_loss = loss_fn(outputs, targets, pred_size=images.shape[2:4])

#                 # Check for NaN in loss
#                 if torch.isnan(loss):
#                     print(f"NaN loss encountered at epoch {epoch}, step {i} , loss_obj{obj_loss[0],obj_loss[1],obj_loss[2]} , seg{obj_loss[3] , obj_loss[4]} , embedding loss{obj_loss[5]}")
#                     continue
                
#                 total_loss += loss.item()

#             # Backpropagation
#             grad_scaler.scale(loss).backward()

#             # Gradient clipping at intervals
#             if (i + 1) % grad_clip_freq == 0:
#                 grad_scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             grad_scaler.step(optimizer)
#             grad_scaler.update()

#             total_det_loss += obj_loss[0] + obj_loss[1] + obj_loss[2]
#             total_cls_loss += obj_loss[2]
#             total_bbx_loss += obj_loss[0]
#             total_obj_loss += obj_loss[1]

#         except RuntimeError as e:
#             print(f"RuntimeError at epoch {epoch}, step {i}: {str(e)}")
#             continue

#     avg_total_loss = total_loss / len(dataloader)
#     avg_det_loss = total_det_loss / len(dataloader)
#     avg_bbx_loss = total_bbx_loss / len(dataloader)
#     avg_obj_loss = total_obj_loss / len(dataloader)
#     avg_cls_loss = total_cls_loss / len(dataloader)

#     return model, avg_total_loss, avg_det_loss, avg_bbx_loss, avg_obj_loss, avg_cls_loss, optimizer, last_opt_step, accumulate

def activation(x):
    return 1e-7 + (1 - 2 * 1e-7) * (0.5 + torch.arctan(x)/torch.tensor(np.pi))

class Evaluator:
    def __init__(self, device, save_path, parameters, build_targets):
        self.device = device
        self.save_path = save_path
        self.parameters = parameters
        self.build_targets = build_targets
        self.activation = activation
        self.metric = MeanAveragePrecision(box_format="xyxy")

    @torch.inference_mode()
    def evaluate(self, net, dataloader, epoch):
        self.net = net
        self.net.eval()
        num_val_batches = len(dataloader)
        self.indices = np.sort(np.random.choice(num_val_batches, epoch*3,replace=False))

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

        preds, trues ,stats= [], [] ,[]
        ap, ap_class = [] , []
        mAP50, mAP75 = 0, 0
        class_accuracy, obj_accuracy = 0, 0
        conf_threshold, nms_iou_thresh = 0.5, 0.3
        anchors = torch.tensor(self.parameters.anchors).float().view(3, -1, 2) / torch.tensor(self.parameters.strides).repeat(6, 1).T.reshape(3, 3, 2)
        iouv = torch.linspace(0.5,0.95,10).to(self.device)     #iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        with torch.no_grad():
            for index , (images, confidence_mask, cluster_masks, object_annotations) in enumerate(tqdm(dataloader)):
                images = images.to(self.device, dtype=torch.float32, memory_format=torch.channels_last)
                confidence_mask = confidence_mask.to(self.device, dtype=torch.long)
                cluster_masks = cluster_masks.to(self.device, dtype=torch.long)
                t1 = time.time()
                predictions = self.net(images)
                t2=time.time()
                print(t2-t1)
                pred_conf_drivable = self.activation(predictions[0][:, 0, :, :])
                pred_conf_lane = self.activation(predictions[0][:, 1, :, :])
                pred_fe = predictions[1]
                gt_drivable = (confidence_mask[:, 0, :, :]).cpu().numpy()
                gt_lane = (confidence_mask[:, 1, :, :]).cpu().numpy()
                pred_drivable = (pred_conf_drivable > 0.5).cpu().numpy()
                pred_lane = (pred_conf_lane > 0.5).cpu().numpy()

                # Drivable area metrics
                drivable_metrics["iou_sum"] += jaccard_score(gt_drivable.flatten(), pred_drivable.flatten(),zero_division=1)
                drivable_metrics["conf_matrix"] += confusion_matrix(gt_drivable.flatten(), pred_drivable.flatten(), labels=[0, 1])
                drivable_metrics["pixel_accuracy"] += accuracy_score(gt_drivable.flatten(), pred_drivable.flatten())
                drivable_metrics["dice_score"] += f1_score(gt_drivable.flatten(), pred_drivable.flatten(),zero_division=1)

                # Lane metrics
                lane_metrics["f_score_sum"] += utils.eval_lane_per_image(gt_lane, pred_lane)
                lane_metrics["iou_sum"] += jaccard_score(gt_lane.flatten(), pred_lane.flatten(),zero_division=1)
                lane_metrics["conf_matrix"] += confusion_matrix(gt_lane.flatten(), pred_lane.flatten(), labels=[0, 1])
                lane_metrics["pixel_accuracy"] += accuracy_score(gt_lane.flatten(), pred_lane.flatten())

                # Object detection metrics
                labels = self.build_targets(predictions, object_annotations, pred_size=images.shape[2:4], train=False)
                class_acc, obj_acc = utils.check_det_accuracy(predictions[2][0], labels, conf_threshold)
                true_boxes = cells_to_bboxes(labels, anchors, strides=self.parameters.strides, is_pred=False, to_list=False)
                pred_boxes = non_max_suppression(predictions[2][1], iou_threshold=nms_iou_thresh, threshold=conf_threshold, tolist=False, max_detections=300)
                # pred_boxes[...,2:] = pred_boxes[...,2:]+15
                true_boxes = non_max_suppression(true_boxes, iou_threshold=0.99, threshold=conf_threshold, tolist=False, max_detections=300)
                utils.clip_coords(pred_boxes, images.shape[2:])
                label_classes = true_boxes[..., 0]
                pred_classes = pred_boxes[:, 0]
                pred_scores = pred_boxes[:, 1]
                if len(true_boxes):
                    if len(pred_boxes):
                        correct = torch.zeros(pred_boxes.shape[0], niou, dtype=torch.bool, device=self.device)
                        for i, pred_box in enumerate(pred_boxes[:, 2:]):
                            ious = [utils.iou_width_height(pred_box, label_box.cpu().numpy())
                                    for label_box in true_boxes[:,2:]]
                            index = int(torch.tensor(ious).argmax())
    
                            if ious[index] >= iouv[0] and pred_classes[i] == label_classes[index]:
                                correct[i] = ious[index] > iouv
                                # matches.append((int(pred_classes[i]), i, index, ious[index]))
                        stats.append((correct.cpu(), pred_scores.cpu(), pred_classes.cpu(), label_classes.cpu()))
                    else:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), label_classes.cpu()))
                else:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), label_classes.cpu()))
                if index in self.indices:
                    titles = [
                    "lane gt", "drivable gt", "det gt", "image",
                    "lane pd", "drivable pd", "det gt", "embedding"
                    ]

                    img = images[0].permute(1, 2, 0).cpu().numpy()
                    if len(true_boxes)>0:
                        obj_gt= plot_image(img*255, true_boxes, self.parameters.bdd, self.parameters.class_mapping)
                    else:
                        obj_gt = img
                    if len(pred_boxes)>0:
                        obj_pd= plot_image(img*255, pred_boxes, self.parameters.bdd, self.parameters.class_mapping)
                    else:
                        obj_pd = img
                    embedding =np.sum(pred_fe[0,:,:,:].cpu().numpy(), axis=0, keepdims=True)[0]
                    subplots = [gt_lane[0],gt_drivable[0],obj_gt,img,pred_lane[0],pred_drivable[0],obj_pd,embedding]
                    utils.create_subplots([utils.normalize_to_uint8(subplot) for subplot in subplots],titles,index,epoch,self.save_path)  

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
            mp, mr, map50, map70, map75, map = 0 ,0 ,0 ,0 ,0 ,0
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
    # save_path = os.path.join(os.pardir,'artifacts/')
    save_path = "./artifacts/"
    url = 'https://raw.githubusercontent.com/KiaRational/text/main/epoch.txt'
    # os.makedirs(save_path, exist_ok=True)
    min_loss = 100
    max_val = 0
    # Create a dictionary to store results
    results = {
        "train_loss": [],
        'drivable_frequency_iou': [],
        'drivable_mean_iou': [],
        'drivable_pixel_accuracy': [],
        'drivable_dice_score': [],
        'lane_f_score': [],
        'lane_mean_iou': [],
        'lane_pixel_accuracy': [],
        'mAP_mAP50':[],
        'mAP_mAP75': [],
        "map": [],
        "precision": [],
        "recall": []
        }
    last_opt_step = -1
    nd = torch.cuda.device_count()  # number of CUDA devices
    args.num_workers = min([os.cpu_count() // max(nd, 1), args.batch_size if args.batch_size > 1 else 0, args.num_workers]) 
    # Initialize train dataset and dataloader
    
    train_dataset = LabelGenerator(
        P.train_data_path,
        P.train_label_path,
        save_path,
        image_size=(640, 640),
        normalize=True,
        class_mapping=P.class_mapping,
        train=True,
        transform=True
    )

    train_dataloader = InfiniteDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=LabelGenerator.collate_fn,
    )

    # Initialize validation dataset and dataloader
    val_dataset = LabelGenerator(
        P.val_data_path,
        P.val_label_path,
        save_path,
        image_size=(640, 640),
        normalize=True,
        class_mapping=P.class_mapping,
        train=False,
        transform = False
    )
    val_dataloader = DataLoaderX(
        val_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=LabelGenerator.collate_fn,
    )
    print(f"train dataloader size is :{len(train_dataloader)}")
    print(f"val dataloader size is :{len(val_dataloader)}")

    # first_batch = next(iter(val_dataloader))
    # test_loader = [first_batch] * 25

    # Initialize model
    model = MultiNet().to(args.device)

    if args.pre_trained:
        print("model loaded")
        model.load_state_dict(torch.load(args.pre_trained, map_location="cuda"),strict=False)

    # register_nan_hooks(model)
    if args.optimizer_type == 'adam':

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),  lr=P.lr0_adam , betas=(P.momentum, 0.999))
        lf = lambda x: ((1 + math.cos(x * math.pi / P.epoch_number)) / 2) * \
                (1 - P.lrf) + P.lrf  # cosine

    elif args.optimizer_type == 'sgd':

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=P.lr0_sgd , momentum=P.momentum )
        lf = lambda x: ((-0.9/(1+math.exp(-0.9*(x-2))))+ 1)

    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    loss_fn = Loss.ComputeLoss(device = args.device)
    target_builder = Loss.TargetBuilder(device = args.device)
    accuracy_fn = Accuracy()
    scaler =  torch.cuda.amp.GradScaler(enabled=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)    # Training loop
    accumulate = None
    num_warmup = max(round(P.warmup_epochs * len(train_dataloader)), 100)
    evaluator = Evaluator(
                        device = args.device,
                        save_path = save_path,
                        parameters = P,
                        build_targets = target_builder,
                    )
    for epoch in range(P.epoch_number):
       # model.train()  # Set model in training mode
       # t1 = time.time()
       # model, train_loss, obj_total, box, obj, clss, optimizer ,last_opt_step , accumulate = train_step(
       #     model=model,
       #     dataloader=train_dataloader ,
       #     loss_fn=loss_fn,
       #     accuracy_fn=accuracy_fn,
       #     optimizer=optimizer,
       #     grad_scaler=scaler,
       #     device=args.device,
       #     epoch=epoch,
       #     num_warmup=num_warmup,
       #     Parameters=P,
       #     lf=lf,
       #     arguments= args,
       #     last_opt_step=last_opt_step,
       #     accumulate=accumulate
       #  )
       # t2 = time.time()
       # print(
       #     f"Epoch: {epoch + 1} | "
       #     f"train_loss: {train_loss:.4f} | "
       #     f"obj_total loss: {obj_total:.4f} |"
       #     f"box loss : {box:.4f}| "
       #     f"obj loss : {obj:.4f}| "
       #     f"class loss : {clss:.4f}| "
       #     f"time : {(t2-t1)/60}"
       # )

       # scheduler.step()
        if ((epoch+1) % 1 == 0):
            t3 = time.time()
            drivable_score, lane_score, obj_scores = evaluator.evaluate(
                                                                        net = model ,
                                                                        dataloader = val_dataloader,
                                                                        epoch = epoch
                                                                        )
            t4 = time.time()
            print(
                f"Validation: {epoch + 1} | "
                f"time : {(t4-t3)/60}"
            )
            utils.print_metrics_in_rich_table([drivable_score, lane_score, obj_scores], ["Drivable Area Results", "Lane Results", "Object Detection Results"])
            
            results["drivable_frequency_iou"].append(drivable_score["frequency_iou"])
            results["drivable_mean_iou"].append(drivable_score["mean_iou"])
            results["drivable_pixel_accuracy"].append(drivable_score["pixel_accuracy"])
            results["drivable_dice_score"].append(drivable_score["dice_score"])
            results["lane_f_score"].append(lane_score["f_score"])
            results["lane_mean_iou"].append(lane_score["mean_iou"])
            results["lane_pixel_accuracy"].append(lane_score["pixel_accuracy"])
            results["mAP_mAP50"].append(obj_scores["mAP50"])
            results["mAP_mAP75"].append(obj_scores["mAP75"])
            results["map"].append(obj_scores["map"])
            results["precision"].append(obj_scores["precision"])
            results["recall"].append(obj_scores["recall"])
        
            val_score = (drivable_score["mean_iou"] + drivable_score["pixel_accuracy"] + lane_score["pixel_accuracy"] + lane_score["mean_iou"] + obj_scores["mAP50"])/5
            # if train_loss < min_loss and val_score > max_val:
        if train_loss < min_loss:

        # Save model
            model.eval()
            torch.save(model.state_dict(), os.path.join(save_path, f"best.pt"))
            model.train()
            min_loss = train_loss
            # max_val = val_score

        # Update results dictionary
        results["train_loss"].append(train_loss)

        torch.save(model.state_dict(), os.path.join(save_path, f"last.pt"))

        for key, values in results.items():
            plt.figure(figsize=(8, 6))
            plt.plot(values, marker='o', linestyle='-', color='b')
            plt.title(key)
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.grid(True)
            plt.savefig(f'{save_path}/{key}_plot.png')  # Save the figure with a unique name based on the key
            plt.clf()
            plt.close()
        try:
            end_epoch_manual = int(requests.get(url).content)
            print(end_epoch_manual)
            if epoch >= end_epoch_manual :
                break
        except Exception as e:
            print(e)

    return results, model

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    print('''
      _______        _       
     |__   __|      (_)      
        | |_ __ __ _ _ _ __  
        | | '__/ _` | | '_ \ 
        | | | | (_| | | | | |
        |_|_|  \__,_|_|_| |_|

            '''    )
    parser = argparse.ArgumentParser(description="Your description here.")
    # parser.add_argument("--data_path", type=str, help="Path to data.")
    # parser.add_argument("--label_path", type=str, help="Path to labels.")
    parser.add_argument("--pre_trained", type=str, default= "./artifacts/train_2/best.pt",help="Path to pre trained model.")
    parser.add_argument("--save_path", type=str, default="./" ,help="Path to save results.")
    parser.add_argument("--device",type=str ,default="cuda", help="choose your training device cuda or cpu")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for dataloader.")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size.")
    parser.add_argument("--shuffle", action="store_true",default=False, help="Shuffle the data.")
    parser.add_argument("--optimizer_type" , type=str , default = "adam" , help="Optimizer Type between adam and sgd")
    parser.add_argument("--version", action="store_true",default="1", help="version of your training")
    args = parser.parse_args()
    P = Parameters()
    # writer_path =  os.path.join(args.save_path,"/runs/1/")
    # os.makedirs(writer_path, exist_ok=True)
    # writer = SummaryWriter(writer_path)
    results , model = train(args,P)
