import torch
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
# import albumentations as A
import shutil
from .Preprocess import CustomDataLoader


class LabelGenerator(Dataset):
    def __init__(self, data_path, label_path, save_dir ,image_size=(720, 1280), normalize=True, class_mapping=None  , train=True ):

        self.data_loader = CustomDataLoader(data_path, label_path, image_size, normalize, class_mapping , train)

        self.image_size = image_size

        self.save_dir = save_dir

        self.check_disk_space(self.save_dir , self.data_loader.num_samples )

    def calculate_disk_space(self, num_samples):
        # Size calculation for image, cluster_mask, and instance_mask
        image_size = np.array((3, 640, 640)).nbytes  # Size of float32 data in bytes
        cluster_mask_size = np.array((2, 320, 320)).nbytes # Size of int32 data in bytes
        instance_mask_size = np.array((2, 6400, 6400)).nbytes # Size of int32 data in bytes

        total_size = num_samples * (image_size + cluster_mask_size + instance_mask_size)

        return total_size

    def check_disk_space(self, save_dir, num_samples):

        required_space = self.calculate_disk_space(num_samples)
        available_space = shutil.disk_usage(save_dir).free
        if required_space > available_space:
            raise ValueError(" ⚠️ Not enough disk space available for saving the data! ")

    def __len__(self):

        return self.data_loader.num_samples

    
    def __getitem__(self, index):

        annotation = self.data_loader.annotations[index]
        print(annotation['name'])
        image  , cluster_mask , instance_drivable  , objects_annotations = self.data_loader.process(annotation)

        confidence_mask = np.zeros_like(cluster_mask)

        confidence_mask[cluster_mask>0]=1
        
        image = image.transpose((2, 0, 1))

        return image, confidence_mask , instance_drivable  


class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

