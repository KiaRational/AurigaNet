import torch
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
import os
import cv2
import numpy as np
import json
import time
import matplotlib.pyplot as plt
# import albumentations as A
import shutil
from .Preprocess import CustomDataLoader


class LabelGenerator(Dataset):
    def __init__(self, data_path, label_path, save_dir ,image_size=(720, 1280), normalize=True, class_mapping=None  , train=True , transform=True ):

        self.data_loader = CustomDataLoader(data_path, label_path, image_size, normalize, class_mapping , train , transform)

        self.image_size = image_size

        self.save_dir = save_dir

        self.check_disk_space(self.save_dir , self.data_loader.num_samples )

        self.ultra = False
        self.train = train

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
        image  , cluster_mask , drivable_clustered_mask_pooled  , objects_annotations = self.data_loader.process(annotation)

        confidence_mask = (cluster_mask>0).astype(int)

        image = image.transpose((2, 0, 1))
        image = torch.tensor(image)
        confidence_mask = torch.tensor(confidence_mask)
        embedding = torch.from_numpy(drivable_clustered_mask_pooled).unsqueeze(0)

        if not self.ultra or not self.train:

            return image, confidence_mask ,embedding , objects_annotations

        else:
            return image, [confidence_mask , embedding , objects_annotations]

    def collate_fn(batch):

        image, confidence_masks , drivable_clustered_masks_pooled , object_annotations = zip(*batch)

        return torch.stack(image, 0), torch.stack(confidence_masks,0) , torch.stack(drivable_clustered_masks_pooled,0) , object_annotations


    def collate_fn_ultra(batch):
        image, label= zip(*batch)
        label_det, label_seg, label_inst = [], [], []
        for i , (confidence_mask , drivable_clustered_mask_pooled , objects_annotations) in enumerate(label):
            if len(objects_annotations) > 0:
                objects_annotations[:, 0] = i  # add target image index for build_targets()
            label_det.append(objects_annotations)
            label_seg.append(confidence_mask)
            label_inst.append(drivable_clustered_mask_pooled)
        return torch.stack(image, 0), torch.stack(label_seg, 0), torch.stack(label_inst, 0), torch.cat(label_det, 0)

class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        """Initializes an InfiniteDataLoader that reuses workers with standard DataLoader syntax, augmenting with a
        repeating sampler.
        """
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler in the InfiniteDataLoader."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Yields batches of data indefinitely in a loop by resetting the sampler when exhausted."""
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        """Initializes a perpetual sampler wrapping a provided `Sampler` instance for endless data iteration."""
        self.sampler = sampler

    def __iter__(self):
        """Returns an infinite iterator over the dataset by repeatedly yielding from the given sampler."""
        while True:
            yield from iter(self.sampler)