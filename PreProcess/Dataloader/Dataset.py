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
from .PreProcess import CustomDataLoader

# only for test
def max_pooling_2d( input_array, pool_size=(2, 2),Type = "max"):
    
    input_height, input_width = input_array.shape
    output_height = input_height // pool_size[0]
    output_width = input_width // pool_size[1]

    # Reshape input_array to be divided into pool_size blocks
    reshaped_array = input_array[:output_height*pool_size[0], :output_width*pool_size[1]].reshape(
        output_height, pool_size[0], output_width, pool_size[1]
    )

    # Apply max pooling using numpy's max function along specified axes
    if Type == "max":
        pooled_array = reshaped_array.max(axis=(1, 3))
    if Type == "avg":
        pooled_array = reshaped_array.mean(axis=(1, 3)).astype(int)
    return pooled_array

class LabelGenerator(Dataset):
    def __init__(self, data_path, label_path, save_dir ,image_size=(720, 1280), normalize=True, class_mapping=None):

        self.data_loader = CustomDataLoader(data_path, label_path, image_size, normalize, class_mapping)

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

        image  , cluster_mask , instance_drivable , instance_lane , objects_annotations, image_name = self.data_loader.process(annotation)
        

        image = image.transpose((2, 0, 1))

        confidence_mask = np.zeros_like(cluster_mask)
        confidence_mask[cluster_mask>0]=1
        
        return image, confidence_mask , instance_drivable , instance_lane , objects_annotations 


class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# # Usage
# data_path = "/content/dataset/bdd100k/bdd100k/images/100k/val/"
# label_path = '/content/dataset/bdd100k_labels_release/bdd100k/labels/'
# class_mapping = {
#     'traffic_light': 0,
#     'traffic_sign': 1,
#     'car': 2
# }
# dataset = CustomDataLoader(data_path, label_path, image_size=(640,640), normalize=True, class_mapping=class_mapping)
# ano = dataset.annotations[10]
# print(ano)
# image , mask , ground = dataset.process(ano)
# d = mask[0]
# d = max_pooling_2d(d)
# d = max_pooling_2d(d)

# # d = max_pooling_2d(d)

# # aug = A.Compose({
# # A.Rotate(limit=(-40, 40),p=1),
# #                   })
# # transformed = aug(image=image, masks=mask)
# # image = transformed["image"]
# # mask = transformed["masks"]
# # maskss = cv2.resize(mask[1],(160,160), interpolation=cv2.INTER_NEAREST)
# # kernel = np.ones((3,3),np.uint8)
# # # Perform image dilation
# # maskss = cv2.dilate(maskss, kernel, iterations=1)
# # maskss = cv2.resize(maskss,(80,80), interpolation=cv2.INTER_NEAREST)
# # kernel = np.ones((2,2),np.uint8)
# # # Perform image dilation
# # maskss = cv2.dilate(maskss, kernel, iterations=1)
# # maskss = cv2.resize(maskss,(40,40), interpolation=cv2.INTER_NEAREST)
# l = np.zeros_like(d)
# l[d>0] = 1
# plt.imshow(d)
# plt.show()


# # batch_size = 8 # Choose your desired batch size
# # train_data_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=False,pin_memory=False, num_workers=2)

# # for i, (batch_images, masks) in enumerate(train_data_loader):
# #     try:
# #         print(batch_images.shape)
# #         print(masks.shape)
# #         print(i)
# #     except Exception as e:
# #         print("Error occurred:", e)

# #     # Clear previous data from axes
# #     if ax is not None:
# #         ax.clear()  # Clear any patches from the axis
