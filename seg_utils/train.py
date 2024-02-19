import os
import sys
import torch
import math
import numpy as np
from  torch import nn
from  tqdm import  tqdm
import torch.nn.functional as F
import argparse
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# from torch.utils.tensorboard import SummaryWriter

# Add the project root to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import specific modules
from Dataloader.Dataset import LabelGenerator, DataLoaderX
from Dataloader.Preprocess import cluster_to_embedding_feature
# from   Models.unet_model import UNet as MultiNet
from Models.MultiNet import MultiNet
from   seg_utils.Parameters import Parameters
import seg_utils.losses as Loss
from   seg_utils.accuracy import Accuracy
import seg_utils.utils as utils
from seg_utils.utils import non_max_suppression,cells_to_bboxes,make_grids,plot_image,xywh2xyxy,scale_boxes, mAP , plot_confusion_matrix

def train_step(model,dataloader,loss_fn,accuracy_fn,optimizer,grad_scaler,device,epoch,num_warmup,Parameters,lf,arguments,SegOnly=False):

    total_loss = 0
    total_cls_loss  = 0
    total_iou_drivable = 0
    total_iou_lane = 0
    total_bbx_loss = 0
    total_obj_loss = 0
    total_det_loss = 0 
    total_steps = len(dataloader)
    nbs = total_steps
    lf = lambda x: ((1 + math.cos(x * math.pi / Parameters.epoch_number)) / 2) * \
        (1 - Parameters.lrf) + Parameters.lrf  # cosine
    for i, (images, confidence_mask, cluster_masks , object_annotations ) in enumerate(tqdm(dataloader)):
        # instance_drivable = cluster_to_embedding_feature(cluster_masks,P.embedding_grid_size,device)

        ni = i + nbs * epoch  # number integrated batches (since train start)
        # print(cluster_masks.size())

        if ni < num_warmup:
            # warm up
            xi = [0, num_warmup]
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [Parameters.warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [Parameters.warmup_momentum, Parameters.momentum])

        if SegOnly:
            targets = [confidence_mask.to(device),cluster_masks.to(device)]
        else:
            targets = [confidence_mask.to(device),cluster_masks.to(device),object_annotations]
        with torch.autocast(device):

            inputs = images.to(device)
            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss , obj_loss = loss_fn(outputs, targets , pred_size=images.shape[2:4])
            
            total_loss += loss.item()

        # Backpropagation
        # compute gradient and do update step
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        total_det_loss += obj_loss[0] + obj_loss[1] + obj_loss[2]
        total_cls_loss += obj_loss[2]
        total_bbx_loss += obj_loss[0]
        total_obj_loss += obj_loss[1]
        # total_loss = 10
    return model , (total_loss / len(dataloader)) , total_det_loss / len(dataloader) , total_bbx_loss / len(dataloader),  total_obj_loss / len(dataloader), total_cls_loss / len(dataloader) , optimizer

def activation(x):
    return 1e-7 + (1 - 2 * 1e-7) * (0.5 + torch.arctan(x)/torch.tensor(np.pi))

@torch.inference_mode()
def evaluate(net, dataloader, device , save_path , Parameters , loss_fn):
    single_cls = False
    net.eval()
    TPFP50 = np.zeros((len(Parameters.bdd),2))
    TPFP75 = np.zeros((len(Parameters.bdd),2))
    num_val_batches = len(dataloader)
    mAP50 = 0
    mAP75 = 0
    preds = []
    trues = []
    iou = 0
    iou_lane = 0
    iou_area = 0
    tot_class_preds, correct_class = 0, 0
    tot_obj, correct_obj = 0, 0
    class_accuracy = 0
    obj_accuracy = 0
    conf_threshold = 0.3
    nms_iou_thresh = 0.5
    ultra = False
    metric = MeanAveragePrecision(box_format = "cxcywh")

    if ultra :
        anchors = torch.tensor(Parameters.anchors)
    else:
        anchors = torch.tensor(Parameters.anchors).float().view(3, -1, 2) / torch.tensor(Parameters.strides).repeat(6, 1).T.reshape(3, 3, 2)
    
    # iterate over the validation set
    with torch.autocast(device):

        for i, (images, confidence_mask, cluster_masks , object_annotations) in enumerate(tqdm(dataloader)):
            
            # move images and labels to correct device and type
            image = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            confidence_mask = confidence_mask.to(device=device, dtype=torch.long)
            # predict the mask
            # instance_drivable = cluster_to_embedding_feature(cluster_masks,Parameters.embedding_grid_size,device)

            targ = [confidence_mask.to(device),cluster_masks.to(device),object_annotations]

            with torch.no_grad():
                predictions = net(image)

            loss , obj_loss , labels = loss_fn(predictions, targ , pred_size=images.shape[2:4],train=False)

            class_acc , obj_acc = utils.check_det_accuracy(predictions[2],labels,conf_threshold)
            pred_boxes = cells_to_bboxes(predictions[2], anchors, strides=Parameters.strides, is_pred=True, to_list=False)

            true_boxes = cells_to_bboxes(labels, anchors, strides=Parameters.strides, is_pred=False, to_list=False)
            
            pred_boxes = non_max_suppression(pred_boxes, iou_threshold=nms_iou_thresh, threshold=conf_threshold,
                                            tolist=False, max_detections=300)

            true_boxes = non_max_suppression(true_boxes, iou_threshold=nms_iou_thresh,threshold=conf_threshold,
                                             tolist=False, max_detections=300)
            

            preds.append(
                dict(
                    boxes=pred_boxes[..., 2:],
                    scores=pred_boxes[..., 1],
                    labels=pred_boxes[..., 0].int(),
                )
            )

            trues.append(
                dict(
                    boxes=true_boxes[..., 2:],
                    labels=true_boxes[..., 0].int(),
                )
            )

            obj_accuracy += obj_acc
            class_accuracy += class_acc
            Pred_Confidence , Pred_EmbeddingFeatureArea = predictions[0] , predictions[1]
            # Pred_Confidence = predictions
            Pred_Confidence_Drivable = activation(Pred_Confidence[:,0,:,:])
            Pred_Confidence_Lane = activation(Pred_Confidence[:,1,:,:])
            # Calculate IoU for Drivable Area
            iou_area += Loss.calculate_iou(Pred_Confidence_Drivable > 0.3, confidence_mask[:, 0, :, :])
            # Calculate IoU for Lane
            iou_lane += Loss.calculate_iou(Pred_Confidence_Lane > 0.3, confidence_mask[:, 1, :, :])
            if len(preds)>0:
                im = plot_image(image[0].permute(1, 2, 0).to("cpu"), pred_boxes.to("cpu"),Parameters.bdd)
                metric.update(preds, trues)
                metrics = metric.compute()     
                mAP50 += metrics["map_50"]
                mAP75 += metrics["map_75"]
                preds = []
                trues = []

        # plot_confusion_matrix(confusion_matrix,Parameters.bdd,save_path)
        output = net(image)
        output=activation(output[0][0])         
        drivable_area_mask = output[0,:,:].cpu().numpy()
        lane_mask = output[1,:,:].cpu().numpy()
        plt.clf()
        plt.imshow(drivable_area_mask)
        plt.savefig(save_path+"saved_area.png")
        plt.imshow(lane_mask)
        plt.savefig(save_path+"saved_lane.png")
        net.train()
    return ((iou_area + iou_lane)/2) / max(num_val_batches, 1) , iou_area / max(num_val_batches, 1) , iou_lane / max(num_val_batches, 1) , mAP50/max(num_val_batches, 1) , mAP75/max(num_val_batches, 1) , class_accuracy / max(num_val_batches,1) , obj_accuracy / max(num_val_batches,1)

def train(args,P):
    save_path = os.path.join(os.pardir,'artifacts/')
    os.makedirs(save_path, exist_ok=True)
    min_loss = 100
    # Create a dictionary to store results
    results = {
        "train_loss": [],
        "train_lane_iou_acc": [],
        "train_drivable_iou_acc": [],
    }

    # Initialize train dataset and dataloader
    train_dataset = LabelGenerator(
        P.train_data_path,
        P.train_label_path,
        save_path,
        image_size=(640, 640),
        normalize=True,
        class_mapping=P.class_mapping,
        train=True,
    )
    train_dataloader = DataLoaderX(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
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
    )
    val_dataloader = DataLoaderX(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=args.num_workers,
        collate_fn=LabelGenerator.collate_fn,
    )
    print(f"train dataloader size is :{len(train_dataloader)}")
    print(f"val dataloader size is :{len(val_dataloader)}")

    first_batch = next(iter(val_dataloader))
    test_loader = [first_batch] * 25

    # Initialize model
    model = MultiNet().to(args.device)

    if args.pre_trained:
        model.load_state_dict(torch.load(args.pre_trained, map_location="cuda"))

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
    accuracy_fn = Accuracy()
    scaler =  torch.cuda.amp.GradScaler(enabled=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)    # Training loop

    num_warmup = max(round(P.warmup_epochs * len(train_dataloader)), 100)

    for epoch in range(P.epoch_number):
        model.train()  # Set model in training mode

        model, train_loss, obj_total, box, obj, clss, optimizer = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            optimizer=optimizer,
            grad_scaler=scaler,
            device=args.device,
            epoch=epoch,
            num_warmup=num_warmup,
            Parameters=P,
            lf=lf,
            arguments= args
        )
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"obj_total loss: {obj_total:.4f} |"
            f"box loss : {box:.4f}| "
            f"obj loss : {obj:.4f}| "
            f"class loss : {clss:.4f}| "
        )

        scheduler.step()

        val_score, drivable_score, lane_score, map50, map75, class_acc, obj_acc = evaluate(
                                                                                    net = model ,
                                                                                    dataloader = train_dataloader,
                                                                                    device = args.device,
                                                                                    save_path = save_path,
                                                                                    Parameters = P,
                                                                                    loss_fn = loss_fn
                                                                                )

        print(
            f"Validation: {epoch + 1} | "
            f"total: {val_score:.4f} | "
            f"drivable acc: {drivable_score:.4f} | "
            f"lane acc: {lane_score:.4f} | "
            f"class acc: {class_acc:.4f} | "
            f"obj acc: {obj_acc:.4f} | "
            f"map50 acc: {map50:.4f} | "
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_lane_iou_acc"].append(lane_score)
        results["train_drivable_iou_acc"].append(drivable_score)
        if train_loss < min_loss:
            # Save model
            torch.save(
                model.state_dict(), os.path.join(save_path, f"{epoch}_{train_loss}_AurigaNet.pth")
            )
            
            min_loss = train_loss

        plt.clf()
        # Plot and save loss curves
        plt.plot(results["train_loss"], label="train_loss")
        plt.legend()
        plt.savefig(os.path.join(save_path, "loss.png"))
        plt.clf()
        plt.plot(results["train_lane_iou_acc"], label="train_lane_iou_acc")
        plt.plot(results["train_drivable_iou_acc"], label="train_drivable_iou_acc")    print("Training complete")
        plt.legend()
        plt.savefig(os.path.join(save_path, "acc.png"))
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
    parser.add_argument("--pre_trained", type=str, default= "",help="Path to pre trained model.")
    parser.add_argument("--save_path", type=str, default="/home/kia/Multi-Task-Network/" ,help="Path to save results.")
    parser.add_argument("--device",type=str ,default="cuda", help="choose your training device cuda or cpu")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--shuffle", action="store_true",default=False, help="Shuffle the data.")
    parser.add_argument("--optimizer_type" , type=str , default = "adam" , help="Optimizer Type between adam and sgd")
    parser.add_argument("--version", action="store_true",default="1", help="version of your training")
    args = parser.parse_args()
    P = Parameters()
    # writer_path =  os.path.join(args.save_path,"/runs/1/")
    # os.makedirs(writer_path, exist_ok=True)
    # writer = SummaryWriter(writer_path)
    results , model = train(args,P)
