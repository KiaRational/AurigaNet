import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import time
from copy import deepcopy

from .CreatMask import poly2ds_to_mask ,  ImageSize


class CustomDataLoader:
    def __init__(self, data_path, label_path, image_size=(720, 1280), normalize=True, class_mapping=None):
        self.data_path = data_path
        self.image_size = image_size
        self.normalize = normalize
        self.class_mapping = class_mapping
        self.label_path = label_path
        self.annotations = self.load_data()
        self.num_samples = len(self.annotations)

        # self.fig = plt.figure(facecolor="0")
        # self.fig.set_size_inches(1280 / self.fig.get_dpi(), 720 / self.fig.get_dpi())
        # self.ax = self.fig.add_axes([0, 0, 1, 1])

    def load_data(self):
        json_file_path = os.path.join(self.label_path, 'bdd100k_labels_images_train.json')
        with open(json_file_path, 'r') as f:
            annotations = json.load(f)

        return self.filter_invalid_images(annotations)

    def filter_invalid_images(self, annotations):
        valid_annotations = []
        for annotation in annotations:
            image_name = annotation['name']
            image_path = os.path.join(self.data_path, image_name)
            if os.path.exists(image_path):
                valid_annotations.append(annotation)
            else:
                continue
        return valid_annotations


    def process(self, annotation):
        # Initialize variables
        image = None
        combined_mask = None
        lane_mark_mask = None
        drivable_mask = None

        # Get image name and path from the annotation
        image_name = annotation['name']
        image_path = os.path.join(self.data_path, image_name)

        # Read and convert the image to RGB format
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create drivable and lane cluster masks
        drivable_clustered_mask , lane_clustered_mask = self.create_masks(annotation)#self.create_masks(self.fig, self.ax, annotation)

        # Resize and normalize the image
        image = self.resize_image(image)
        image = self.normalize_image(image)

        # Resize drivable and lane cluster masks
        lane_clustered_mask = self.resize_mask(lane_clustered_mask)
        drivable_clustered_mask = self.resize_mask(drivable_clustered_mask)

        # Combine drivable and lane cluster masks
        cluster_mask = np.stack([drivable_clustered_mask , lane_clustered_mask])

        # Apply max pooling twice to drivable and lane cluster masks
        drivable_clustered_mask_pooled = self.max_pooling_2d(self.max_pooling_2d(drivable_clustered_mask))
        lane_clustered_mask_pooled = self.max_pooling_2d(self.max_pooling_2d(lane_clustered_mask))

        # Convert cluster masks to instance masks using embedding feature
        instance_drivable = self.cluster_to_embedding_feature(drivable_clustered_mask_pooled,  size=80) 
        instance_lane     = self.cluster_to_embedding_feature(lane_clustered_mask_pooled   ,   size=80)

        # Combine drivable and lane instance masks
        instance_mask = np.stack([instance_drivable , instance_lane])

        # Return processed image, cluster masks, and instance masks
        return image, cluster_mask , instance_mask , image_name


    def create_masks(self, annotation):
        lane_clustered = np.zeros((720, 1280))
        drivable_clustered = np.zeros((720, 1280))
        lane_cluster_index = 0
        drivable_cluster_index = 0
        for label_info in annotation['labels']:

            category = label_info['category']

            if category == 'lane':# and label_info['attributes']['laneDirection'] == 'parallel':

                poly2d = label_info['poly2d']
                image_size = ImageSize(width=1280, height=720)
                lane_mask = poly2ds_to_mask(image_size, poly2d)  # Pass fig and ax here

                lane_cluster_index += 1
                lane_clustered += (lane_mask/255)*lane_cluster_index

            if category == 'drivable area':

                poly2d = label_info['poly2d']
                image_size = ImageSize(width=1280, height=720)
                drivable_area_mask = poly2ds_to_mask(image_size, poly2d)  # Pass fig and ax here
                drivable_cluster_index += 1
                drivable_clustered += (drivable_area_mask/255)*drivable_cluster_index

        return drivable_clustered ,lane_clustered

    def max_pooling_2d(self , input_array, pool_size=(2, 2)):

        input_height, input_width = input_array.shape
        output_height = input_height // pool_size[0]
        output_width = input_width // pool_size[1]

        # Reshape input_array to be divided into pool_size blocks
        reshaped_array = input_array[:output_height*pool_size[0], :output_width*pool_size[1]].reshape(
            output_height, pool_size[0], output_width, pool_size[1]
        )

        # Apply max pooling using numpy's max function along specified axes
        pooled_array = reshaped_array.max(axis=(1, 3))

        return pooled_array


    def cluster_to_embedding_feature(self, cluster_mask , size):
        
        temp = cluster_mask

        dimensions = size**2

        ground = np.zeros((1, dimensions , dimensions))

        for i in range(dimensions): # make feature embedding ground truth 

            temp = temp.flatten()
            gt_one = deepcopy(temp)

            if temp[i]>0:

                gt_one[temp==temp[i]] = 1   #same instance
                gt_one[temp!=temp[i]] = 2 #different instance, same class
                gt_one[temp==0] = 3 #different instance, different class
                ground[0][i] += gt_one

        return ground

    def resize_image(self, image):
        return cv2.resize(image, self.image_size)

    def resize_mask(self, mask):
        return cv2.resize(mask, (self.image_size[0]//2,self.image_size[1]//2), interpolation=cv2.INTER_NEAREST)

    def resize_obj_mask(self, obj_mask):
        return cv2.resize(obj_mask, (self.image_size[1] // 32, self.image_size[0] // 32), interpolation=cv2.INTER_NEAREST)

    def normalize_image(self, image):
        return image.astype(np.float32) / 255.0
