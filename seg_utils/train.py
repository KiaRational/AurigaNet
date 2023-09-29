import os
import sys
import torch
import numpy as np
from  torch import nn
from  tqdm import  tqdm
import torch.nn.functional as F
import argparse
import torch.multiprocessing as mp
# from torch.utils.tensorboard import SummaryWriter

# Add the project root to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import specific modules
from   Dataloader.Dataset import LabelGenerator, DataLoaderX
from   Models.MultiNet import MultiNet
from   seg_utils.Parameters import Parameters
import seg_utils.losses as Loss
from   seg_utils.accuracy import Accuracy


def train_step(model,dataloader,loss_fn,accuracy_fn,optimizer,grad_scaler,device):
    total_loss = 0
    total_f1_lane = 0
    total_iou_drivable =0
    total_iou_lane = 0
    total_f1_drivable = 0
    total_steps = len(dataloader)

    for i, (images, confidence_mask, instance_drivable  ) in enumerate(tqdm(dataloader)):
        # Zero the gradients
        targets = [confidence_mask.to(device),instance_drivable.to(device)]
        with torch.autocast(device):

            inputs = images.to(device)
            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = loss_fn(outputs, targets)

        # Backpropagation

        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        total_loss += loss.item()


    return model , (total_loss / len(dataloader))


@torch.inference_mode()
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    dice_score_lane = 0
    dice_score_area = 0
    # iterate over the validation set
    with torch.autocast(device):
        for i, (images, confidence_mask, instance_drivable  ) in enumerate(tqdm(dataloader)):
            
            # move images and labels to correct device and type
            image = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            confidence_mask = confidence_mask.to(device=device, dtype=torch.long)

            # predict the mask
            predictions = net(image)
            Pred_Confidence , Pred_EmbeddingFeatureArea = predictions
            Pred_Confidence_Drivable = Pred_Confidence[:,0,:,:]
            Pred_Confidence_Lane = Pred_Confidence[:,1,:,:]
            dice_score_lane += Loss.dice_coeff(F.sigmoid(Pred_Confidence_Lane),confidence_mask[:,1,:,:],reduce_batch_first=True)
            dice_score_area += Loss.dice_coeff(F.sigmoid(Pred_Confidence_Drivable),confidence_mask[:,0,:,:],reduce_batch_first=True)
            dice_score += Loss.multiclass_dice_coeff(F.sigmoid(Pred_Confidence), confidence_mask, reduce_batch_first=True)

    net.train()
    return dice_score / max(num_val_batches, 1) , dice_score_area / max(num_val_batches, 1) , dice_score_lane / max(num_val_batches, 1)

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
        "train_lane_f1_acc": [],
        "train_lane_iou_acc": [],
        "train_drivable_f1_acc": [],
        "train_drivable_iou_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Initialize training dataset and dataloader
    train_dataset = LabelGenerator(train_data_path, train_label_path, save_path, image_size=(640, 640), normalize=True, class_mapping=class_mapping)
    train_dataloader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False, num_workers=num_workers)

    # Initialize validation dataset and dataloader
    val_dataset = LabelGenerator(val_data_path, val_label_path, save_path, image_size=(640, 640), normalize=True, class_mapping=class_mapping,train=False)
    val_dataloader = DataLoaderX(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    first_batch = next(iter(val_dataloader))

    # test_laoder = [first_batch] * 20

    # Initialize model
    model = MultiNet().to(device)

    if pre_trained != "":

        model.load_state_dict(torch.load(pre_trained, map_location="cuda")) 

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    loss_fn = Loss.ComputeLoss()
    accuracy_fn = Accuracy()
    scaler =  torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # Training loop
    for epoch in range(epochs):


        model.train()  # Set model in training mode

        model , train_loss  = train_step(model=model,
                                        dataloader=train_datalaoder,
                                        loss_fn=loss_fn,
                                        accuracy_fn = accuracy_fn,
                                        optimizer=optimizer,
                                        grad_scaler = scaler
                                        ,device = device)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "

        )
        val_score , drivable_score , lane_score  = evaluate(model, val_dataloader, device)
        scheduler.step(val_score)

        print(
            f"Validation: {epoch+1} | "
            f"total: {val_score:.4f} | "
            f"drivable acc: {drivable_score:.4f} | "
            f"lane acc: {lane_score:.4f} | "
        )
        
        ## TensorBoard Summaries
        # writer.add_scalar('training loss ',train_loss , epoch+1)
        # writer.add_scalar('train drivable mIoU acc ',train_drivable_iou_acc ,  epoch+1)
        # writer.add_scalar('train drivable f1 acc',train_drivable_f1_acc ,  epoch+1)
        # writer.add_scalar('train lane f1 acc',train_lane_f1_acc ,  epoch+1)
        # writer.add_scalar('train lane mIoU acc ',train_lane_iou_acc ,  epoch+1)
        # Update results dictionary


        results["train_loss"].append(train_loss)
        if val_score>0.95:
            break

    torch.save(
        model.state_dict(),
        save_path+str(epoch)+'_'+str(train_loss)+'_'+'AurigaNet.pth'
    )

    print("Training complete")
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
