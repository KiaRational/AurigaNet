import os
import sys
import torch
import numpy as np
from  torch import nn
from  tqdm import  tqdm
import torch.nn.functional as F
import argparse

# Add the project root to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import specific modules
from PreProcess.Dataloader import LabelGenerator, DataLoaderX
from Models.MultiNet import MultiNet
from utils.Parameters import Parameters
import utils.losses as Loss

def train_step(model,dataloader,loss_fn,optimizer,device):

    for i, (images, confidence_mask, instance_drivable , instance_lane , objects_annotations ) in enumerate(dataloader):

        # Zero the gradients
        targets = [confidence_mask.to(device),instance_drivable.to(device),instance_lane.to(device)]
        inputs = images.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = loss_fn(outputs, targets)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(args):
    # Initialize Parameters and set up paths and settings
    P = Parameters()
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
    # Create a dictionary to store results
    results = {
        "train_loss": [],
        "val_loss": [],
    }

    # Initialize training dataset and dataloader
    train_dataset = LabelGenerator(train_data_path, train_label_path, save_path, image_size=(640, 640), normalize=True, class_mapping=class_mapping)
    train_dataloader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False, num_workers=num_workers)

    # Initialize validation dataset and dataloader
    val_dataset = LabelGenerator(val_data_path, val_label_path, save_path, image_size=(640, 640), normalize=True, class_mapping=class_mapping)
    val_dataloader = DataLoaderX(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)

    # Initialize model
    model = MultiNet().to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0008)
    loss_fn = Loss.ComputeLoss()

    # Training loop
    for epoch in tqdm(range(epochs)):
        model.train()  # Set model in training mode

        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer
                                ,device = device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)

    print("Training complete")

if __name__ == "__main__":
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
    # parser.add_argument("--save_path", type=str, help="Path to save results.")
    parser.add_argument("--device",type=str ,default="cuda", help="choose your training device cuda or cpu")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--shuffle", action="store_true",default=False, help="Shuffle the data.")
    args = parser.parse_args()
    train(args)