import os
import sys
import torch
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
from   Dataloader.Dataset import LabelGenerator, DataLoaderX
# from   Models.unet_model import UNet as MultiNet
from Models.MultiNet import MultiNet
from   seg_utils.Parameters import Parameters
import seg_utils.losses as Loss
from   seg_utils.accuracy import Accuracy
from seg_utils.utils import non_max_suppression,cells_to_bboxes,make_grids,plot_image,xywh2xyxy,scale_boxes, mAP


def train_step(model,dataloader,loss_fn,accuracy_fn,optimizer,grad_scaler,device,SegOnly=False):

    total_loss = 0
    total_cls_loss = 0
    total_iou_drivable =0
    total_iou_lane = 0
    total_bbx_loss = 0
    total_obj_loss = 0
    total_steps = len(dataloader)

    for i, (images, confidence_mask, instance_drivable , object_annotations ) in enumerate(tqdm(dataloader)):
        # Zero the gradients
        # print(object_annotations[..., 1])
        if SegOnly:
            targets = [confidence_mask.to(device),instance_drivable.to(device)]
        else:
            targets = [confidence_mask.to(device),instance_drivable.to(device),object_annotations.to(device)]
        with torch.autocast(device):

            inputs = images.to(device)
            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss , obj_loss = loss_fn(outputs, targets)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        total_loss += loss.item()
        total_obj_loss += obj_loss[0] + obj_loss[1] + obj_loss[2]
        total_cls_loss += obj_loss[2]
        total_bbx_loss += obj_loss[0]
        total_obj_loss += obj_loss[1]
        # total_loss = 10

    return model , (total_loss / len(dataloader)) , total_obj_loss / len(dataloader) , total_bbx_loss / len(dataloader),  total_obj_loss / len(dataloader), total_cls_loss / len(dataloader)

def activation(x):
    return 1e-7 + (1 - 2 * 1e-7) * (0.5 + torch.arctan(x)/torch.tensor(np.pi))

@torch.inference_mode()

def evaluate(net, dataloader, device):
    single_cls = False
    net.eval()
    TPFP50 = np.zeros((10,2))
    TPFP75 = np.zeros((10,2))
    num_val_batches = len(dataloader)
    iou = 0
    iou_lane = 0
    iou_area = 0
    tot_class_preds, correct_class = 0, 0
    tot_obj, correct_obj = 0, 0
    class_accuracy = 0
    obj_accuracy = 0
    conf_threshold = 0.4
    nms_iou_thresh = 0.6
    jdict, stats, ap, ap_class = [], [], [], []
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    anchors = torch.tensor([
                [[10, 13], [16, 30], [33, 23]],  # P3/8
                [[30, 61], [62, 45], [59, 119]],  # P4/16
                [[116, 90], [156, 198], [373, 326]] # P5/32
            ])
    # iterate over the validation set
    with torch.autocast(device):

        for i, (images, confidence_mask, instance_drivable , object_annotations) in enumerate(tqdm(dataloader)):
            
            # move images and labels to correct device and type
            image = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            confidence_mask = confidence_mask.to(device=device, dtype=torch.long)
            # predict the mask
            with torch.no_grad():
                predictions = net(image)

            
            pred_boxes = cells_to_bboxes(predictions[2], anchors, strides=[8, 16, 32], is_pred=True, to_list=False)


            pred_boxes = non_max_suppression(pred_boxes, iou_threshold=nms_iou_thresh, threshold=conf_threshold,
                                            tolist=False, max_detections=300)


            T50 , T75 = mAP(object_annotations , pred_boxes)
            TPFP50 += T50 
            TPFP75 += T75
            Pred_Confidence , Pred_EmbeddingFeatureArea = predictions[0] , predictions[1]
            # Pred_Confidence = predictions
            Pred_Confidence_Drivable = activation(Pred_Confidence[:,0,:,:])
            Pred_Confidence_Lane = activation(Pred_Confidence[:,1,:,:])
            # Calculate IoU for Drivable Area
            iou_area += Loss.calculate_iou(Pred_Confidence_Drivable > 0.5, confidence_mask[:, 0, :, :])
            # Calculate IoU for Lane
            iou_lane += Loss.calculate_iou(Pred_Confidence_Lane > 0.5, confidence_mask[:, 1, :, :])

    mAP50 = TPFP50[:,1]/(TPFP50[:,0]+TPFP50[:,1])
    mAP75 = TPFP75[:,1]/(TPFP75[:,0]+TPFP75[:,1])
    
    output = net(image)
    output=activation(output[0][0])            
    drivable_area_mask = output[0,:,:].cpu().numpy()
    lane_mask = output[1,:,:].cpu().numpy()
    plt.imshow(drivable_area_mask)
    plt.savefig("/home/kia/Multi-Task-Network/Saved/saved_area.png")
    plt.imshow(lane_mask)
    plt.savefig("/home/kia/Multi-Task-Network/Saved/saved_lane.png")

    net.train()
    return ((iou_area + iou_lane)/2) / max(num_val_batches, 1) , iou_area / max(num_val_batches, 1) , iou_lane / max(num_val_batches, 1) , np.nanmean(mAP50)/max(num_val_batches, 1) , np.nanmean(mAP75)/max(num_val_batches, 1) , class_accuracy / max(num_val_batches,1) , obj_accuracy / max(num_val_batches,1)

def train(args,P):
    # Initialize Parameters and set up paths and settings
    train_data_path = P.train_data_path
    train_label_path = P.train_label_path
    val_data_path = P.val_data_path
    val_label_path = P.val_label_path
    save_path = P.save_path
    class_mapping = P.class_mapping
    epochs = P.epoch_number
    num_workers = args.num_workers
    batch_size = args.batch_size
    shuffle = args.shuffle
    device = args.device
    save_path =  os.path.join("/home/kia/Multi-Task-Network/Saved/")
    pre_trained = args.pre_trained
    #os.makedirs(save_path, exist_ok=True)

    # Create a dictionary to store results
    results = {"train_loss": [],
        "train_lane_iou_acc": [],
        "train_drivable_iou_acc": [],
        "test_loss": [],
        "test_acc": [],
        "class_acc":[],
        "obj_acc":[],
        "map50":[],
        "map75":[]
    }
    
    # Initialize training dataset and dataloader
    # train_dataset = LabelGenerator(train_data_path, train_label_path, save_path, image_size=(640, 640), normalize=True, class_mapping=class_mapping)
    # train_dataloader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False, num_workers=num_workers)

    # Initialize validation dataset and dataloader
    val_dataset = LabelGenerator(val_data_path, val_label_path, save_path, image_size=(640, 640), normalize=True, class_mapping=class_mapping,train=False)
    val_dataloader = DataLoaderX(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers , collate_fn=LabelGenerator.collate_fn)
    first_batch = next(iter(val_dataloader))

    test_laoder = [first_batch] * 20

    # Initialize model
    # model = MultiNet(3,2).to(device)
    model = MultiNet().to(device)

    if pre_trained != "":
        model.load_state_dict(torch.load(pre_trained, map_location="cuda")) 

    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
    loss_fn = Loss.ComputeLoss()
    accuracy_fn = Accuracy()
    scaler =  torch.cuda.amp.GradScaler(enabled=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3 ,verbose=True)
    # Training loop
    for epoch in range(epochs):

        model.train()  # Set model in training mode

        model , train_loss , obj_total , box , obj , clss = train_step(model=model,
                                        dataloader=test_laoder,
                                        loss_fn=loss_fn,
                                        accuracy_fn = accuracy_fn,
                                        optimizer=optimizer,
                                        grad_scaler = scaler
                                        ,device = device)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"obj_total loss: {obj_total:.4f} |"
            f"box loss : {box:.4f}| "
            f"obj loss : {obj:.4f}| "
            f"class loss : {clss:.4f}| "

        )
        scheduler.step(train_loss)

        val_score , drivable_score , lane_score , map50 , map75 , class_acc , obj_acc = evaluate(model, test_laoder, device)

        print(
            f"Validation: {epoch+1} | "
            f"total: {val_score:.4f} | "
            f"drivable acc: {drivable_score:.4f} | "
            f"lane acc: {lane_score:.4f} | "
            f"class acc: {class_acc:.4f} | "
            f"obj acc: {obj_acc:.4f} | "
            f"map50 acc: {map50:.4f} | "
        ) 
        ## TensorBoard Summaries
        # writer.add_scalar('training loss ',train_loss , epoch+1)
        # Update results dictionary

        results["train_loss"].append(train_loss)
        results["train_lane_iou_acc"].append(lane_score)
        results["train_drivable_iou_acc"].append(drivable_score)

    torch.save(
        model.state_dict(),
        save_path+str(epoch)+'_'+str(train_loss)+'_'+'AurigaNet.pth'
    )

    print("Training complete")
    plt.cla()
    plt.plot(results["train_loss"],label="train_loss")
    plt.plot(results["train_lane_iou_acc"],label="train_lane_iou_acc")
    plt.plot(results["train_drivable_iou_acc"],label="train_drivable_iou_acc")

    plt.legend()

    plt.savefig("/home/kia/Multi-Task-Network/Saved/loss.png")
    # writer.close()
    return results , model

if __name__ == "__main__":
    mp.set_start_method('spawn')
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
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--shuffle", action="store_true",default=False, help="Shuffle the data.")
    parser.add_argument("--version", action="store_true",default="1", help="version of your training")
    args = parser.parse_args()
    P = Parameters()
    # writer_path =  os.path.join(args.save_path,"/runs/1/")
    # os.makedirs(writer_path, exist_ok=True)
    # writer = SummaryWriter(writer_path)
    results , model = train(args,P)