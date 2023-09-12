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

from PreProcess.Dataloader.Preprocess import CustomDataLoader


from utils.Parameters import Parameters


class LabelGenerator(Dataset):

    def __init__(self, data_path, label_path, save_dir ,image_size=(720, 1280), normalize=True, class_mapping=None):

        self.data_loader = CustomDataLoader(data_path, label_path, image_size, normalize, class_mapping)

        self.image_size = image_size

        self.save_dir = save_dir
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

        cv2.imwrite(os.path.join(self.Area_save_dir,image_name),drivable_clustered*(255//drivable_cluster_index))

        cv2.imwrite(os.path.join(self.Lane_save_dir,image_name),lane_clustered*(255//lane_cluster_index))

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
    data_path = "/home/kia/BDD100K/bdd100k_images_100k_5/bdd100k/images/100k/train/"
    label_path = '/home/kia/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
    save_path = '/home/kia/BDD100K/Generated/'

    class_mapping = {
        'traffic_light': 0,
        'traffic_sign': 1,
        'car': 2
    }

    dataset = LabelGenerator(data_path, label_path, save_path ,image_size=(640, 640), normalize=True, class_mapping=class_mapping)
    batch_size = 1
    train_data_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=1)
    # Create subfolders if they don't exist
    save_dir = save_path
    image_save_dir = os.path.join(save_dir, "Area_Mask")
    mask_save_dir = os.path.join(save_dir, "Lane_Masks")

    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    for i, (run) in enumerate(tqdm(train_data_loader)):
        
        continue

if __name__ == "__main__":

    main()
