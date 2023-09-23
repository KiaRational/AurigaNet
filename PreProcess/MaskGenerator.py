from tqdm import tqdm
import os
import sys 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from Dataloader import CustomDataLoader

from seg_utils.Parameters import Parameters


class LabelGenerator(Dataset):

    def __init__(self, data_path, label_path, save_dir ,image_size=(720, 1280), normalize=True, class_mapping=None,train=True):

        self.data_loader = CustomDataLoader(data_path, label_path, image_size, normalize, class_mapping)

        self.image_size = image_size
        if train:
            self.save_dir = save_dir + "/train/"
        else:
            self.save_dir = save_dir + "/val/"

        self.Area_save_dir = os.path.join(self.save_dir , "Area_Mask")
        self.Lane_save_dir = os.path.join(self.save_dir , "Lane_Masks")

        os.makedirs(self.Area_save_dir, exist_ok=True)
        os.makedirs(self.Lane_save_dir, exist_ok=True)

        self.check_disk_space(self.save_dir , self.data_loader.num_samples )


    def calculate_disk_space(self, num_samples):
        # Size calculation for image, cluster_mask, and instance_mask
        image_shape = (720, 1280, 3)  # Assuming 3 channels (e.g., RGB)

        # Calculate the size of the image data in bytes
        image_size = np.prod(image_shape) * np.dtype(np.uint8).itemsize 


        # Calculate the size for compressed JPEG images
        compression_ratio = 1.0 / 90.0  # Assume 10:1 compression ratio
        jpeg_size = image_size * compression_ratio

        total_size = num_samples * (jpeg_size * 2)

        return total_size

    def check_disk_space(self, save_dir, num_samples):

        required_space = self.calculate_disk_space(num_samples)
        available_space = shutil.disk_usage(save_dir).free
        if required_space > available_space:
            raise ValueError(f" ⚠️ Not enough disk space available for saving the data! you need {required_space//(10**6)}MB but have {available_space//(10**6)}MB free" )

    def __len__(self):

        return self.data_loader.num_samples

    
    def __getitem__(self, index):

        annotation = self.data_loader.annotations[index]

        image_name = annotation['name']

        drivable_clustered , drivable_cluster_index ,lane_clustered , lane_cluster_index , obj_annot = self.data_loader.create_masks(annotation,CreateMasks=True)

        cv2.imwrite(os.path.join(self.Area_save_dir,str(image_name[:-4])+".png"),(drivable_clustered*(255/(drivable_cluster_index))).astype(np.uint8))

        # cv2.imwrite(os.path.join(self.Lane_save_dir,str(image_name[:-4])+".png"),lane_clustered*(255//lane_cluster_index))

        return []


class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def main():
    intro  = '''
    _____        _                 _     _           _          _    _____                           _             
    |  __ \      | |               | |   | |         | |        | |  / ____|                         | |            
    | |  | | __ _| |_ __ _ ___  ___| |_  | |     __ _| |__   ___| | | |  __  ___ _ __   ___ _ __ __ _| |_ ___  _ __ 
    | |  | |/ _` | __/ _` / __|/ _ \ __| | |    / _` | '_ \ / _ \ | | | |_ |/ _ \ '_ \ / _ \ '__/ _` | __/ _ \| '__|
    | |__| | (_| | || (_| \__ \  __/ |_  | |___| (_| | |_) |  __/ | | |__| |  __/ | | |  __/ | | (_| | || (_) | |   
    |_____/ \__,_|\__\__,_|___/\___|\__| |______\__,_|_.__/ \___|_|  \_____|\___|_| |_|\___|_|  \__,_|\__\___/|_|   
                                                                                                                    
                                                                                
        '''

    print(intro)

    # Usage

    P = Parameters()
    train_data_path = P.train_data_path
    train_label_path = P.train_label_path
    val_data_path = P.val_data_path
    val_label_path = P.val_label_path
    save_path = P.save_path
    class_mapping = P.class_mapping
    epochs = P.epoch_number

    train_dataset = LabelGenerator(train_data_path, train_label_path, save_path ,image_size=(640, 640), normalize=True, class_mapping=class_mapping,train=True)
    val_dataset = LabelGenerator(val_data_path, val_label_path, save_path ,image_size=(640, 640), normalize=True, class_mapping=class_mapping,train=False)

    batch_size = 1

    train_data_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=1)
    val_data_loader = DataLoaderX(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=1)


    for i, (run) in enumerate(tqdm(train_data_loader)):
        
        continue
    print("Generating Train Mask Label Done")

    for i, (run) in enumerate(tqdm(val_data_loader)):
        
        continue

    print("Generating Validation Mask Label Done")

if __name__ == "__main__":

    main()
