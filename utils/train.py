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



def main(args):

    P = Parameters()
    train_data_path = P.train_data_path
    train_label_path = P.train_label_path
    val_data_path = P.val_data_path
    val_label_path = P.val_label_path  
    save_path = P.save_path
    class_mapping = P.class_mapping
    num_workers = args.num_workers
    batch_size = args.batch_size
    shuffle = args.shuffle

    train_dataset = LabelGenerator(train_data_path, train_label_path, save_path ,image_size=(640, 640), normalize=True, class_mapping=class_mapping)
    train_data_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)

    val_dataset = LabelGenerator(val_data_path, val_label_path, save_path ,image_size=(640, 640), normalize=True, class_mapping=class_mapping)
    val_data_loader = DataLoaderX(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)
   

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