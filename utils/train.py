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

def train_step(model,dataloader,loss_fn,optimizer):

    for i, (images, cluster_masks, instance_masks , objects_annotations ) in enumerate(train_data_loader):

            continue


def train(args):

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

    results = {"train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    train_dataset = LabelGenerator(train_data_path, train_label_path, save_path ,image_size=(640, 640), normalize=True, class_mapping=class_mapping)
    train_dataloader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)

    val_dataset = LabelGenerator(val_data_path, val_label_path, save_path ,image_size=(640, 640), normalize=True, class_mapping=class_mapping)
    val_dataloader = DataLoaderX(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)

    model = MultiNet()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0008)

    loss_fn = Loss.ComputeLoss()

    model.train()

    for epoch in tqdm(range(epochs)):    
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer)
        

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            # f"train_acc: {train_acc:.4f} | "
            f"test_loss: {val_loss:.4f} | "
            # f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        # results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        # results["test_acc"].append(test_acc)
    # for i, (image, cluster_mask, instance_mask , objects_annotations , image_name) in enumerate(tqdm(train_data_loader)):
       
    #     continue

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
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--shuffle", action="store_true",default=False, help="Shuffle the data.")
    args = parser.parse_args()
    main(args)