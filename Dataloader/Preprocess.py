import os
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A
import numpy as np
import json
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import torch
import sys

from .CreatMask import poly2ds_to_mask ,  ImageSize
from .Augmentation import RandomShadow

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from seg_utils.Parameters import Parameters


class CustomDataLoader:
    def __init__(self, data_path, label_path, image_size=(720, 1280), normalize=True, class_mapping=None , train=True , transform=False):
        self.data_path = data_path
        self.p = Parameters()
        self.image_size = image_size
        self.normalize = normalize
        self.class_mapping = class_mapping
        self.label_path = label_path
        self.annotations = self.load_data()
        self.num_samples = len(self.annotations)
        self.device = "cuda"
        self.train = train
        self.transform = transform

        if self.transform :
            # CoarseDropout is not defined for object => what does it do?
            self.transforms = A.Compose(transforms=[
                                                RandomShadow(prob=0.50),
                                                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                                                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                                                A.GaussNoise(var_limit=(0, 255), p=0.1),
                                                A.MotionBlur(blur_limit=17, p=0.1),
                                                A.ImageCompression(quality_lower=39, quality_upper=60, p=0.2),
                                                A.HorizontalFlip(p=0.2),
                                                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.075, rotate_limit=20, p=0.2,border_mode=cv2.BORDER_CONSTANT)
                                                ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))


    def load_data(self):
        json_file_path = os.path.join(self.label_path)
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
        
        if self.train:
            lane_mask_path = os.path.join(self.p.train_Lane_Mask_path,str(image_name[:-4])+".png")
            area_mask_path = os.path.join(self.p.train_Area_Mask_path,str(image_name[:-4])+".png")
        else:
            lane_mask_path = os.path.join(self.p.val_Lane_Mask_path,str(image_name[:-4])+".png")
            area_mask_path = os.path.join(self.p.val_Area_Mask_path,str(image_name[:-4])+".png")
            
        # Read and convert the image to RGB format
        image = cv2.imread(image_path)
        lane_clustered_mask = cv2.imread(lane_mask_path)
        drivable_clustered_mask =  cv2.imread(area_mask_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lane_clustered_mask = cv2.cvtColor(lane_clustered_mask, cv2.COLOR_BGR2GRAY)
        drivable_clustered_mask = cv2.cvtColor(drivable_clustered_mask, cv2.COLOR_BGR2GRAY)

        object_annotations = self.create_masks(annotation, False)
        bboxes = []
        class_ids = []
        for object_annot in object_annotations:
            
            bboxes.append(object_annot[2:6])
            class_ids.append(object_annot[1])

        if self.transform:
                
            masks = [drivable_clustered_mask, lane_clustered_mask]

            augmented = self.transforms(image        = image,
                                        masks        = masks,
                                        bboxes       = bboxes,
                                        category_ids = class_ids)

            image = augmented['image']

            drivable_clustered_mask = augmented['masks'][0]

            lane_clustered_mask = augmented['masks'][1]

            bboxes = augmented['bboxes']
            category_ids = augmented['category_ids']

            for i in range(len(bboxes)): # TODO: check reassign augmented formatted box and class id

                object_annotations[i,2:6] = torch.tensor(list(bboxes[i]))
                object_annotations[i,1]   = category_ids[i]

        lane_clustered_mask = self.clusterize(lane_clustered_mask)
        drivable_clustered_mask = self.clusterize(drivable_clustered_mask)


        # Resize and normalize the image
        image = self.resize_image(image)
        image = self.normalize_image(image)

        # Resize drivable and lane cluster masks
        lane_clustered_mask = self.resize_mask(lane_clustered_mask)
        drivable_clustered_mask = self.resize_mask(drivable_clustered_mask)

        cluster_mask = np.stack([drivable_clustered_mask , lane_clustered_mask]) 

        drivable_clustered_mask_pooled = self.max_pooling_2d(self.max_pooling_2d(self.max_pooling_2d(drivable_clustered_mask)))
        # Convert cluster masks to instance masks using embedding feature
        # instance_drivable = self.cluster_to_embedding_feature(drivable_clustered_mask_pooled,  size=40) 
        # instance_drivable = torch.zeros((1,1600,1600))

        # Return processed image, cluster masks, and instance masks

        
        return image  , cluster_mask , drivable_clustered_mask_pooled , object_annotations 

    def create_masks(self, annotation , CreateMasks = True ) :
        lane_clustered = np.zeros((720, 1280))
        drivable_clustered = np.zeros((720, 1280))
        lane_cluster_index = 1
        drivable_cluster_index = 1
        obj_annot = []
        for label_info in annotation['labels']:
            
            category = label_info['category']

            if category == 'lane':# and label_info['attributes']['laneDirection'] == 'parallel':
                
                if CreateMasks:

                    poly2d = label_info['poly2d']

                    image_size = ImageSize(width=1280, height=720)
                    lane_mask = poly2ds_to_mask(image_size, poly2d)
                    # Identify zero elements in lane_clustered
                    zero_elements = lane_clustered == 0

                    # Add (lane_mask/255)*lane_cluster_index to zero elements
                    lane_clustered[zero_elements] += (lane_mask[zero_elements]/255) * lane_cluster_index


            if category == 'drivable area':
                if CreateMasks:
                    poly2d = label_info['poly2d']
                    image_size = ImageSize(width=1280, height=720)
                    drivable_area_mask = poly2ds_to_mask(image_size, poly2d)  
                    drivable_clustered += (drivable_area_mask/255)*drivable_cluster_index
                    drivable_cluster_index += 1

            if category in self.class_mapping.keys():

                class_index = self.class_mapping[category]

                x1 = int(label_info['box2d']['x1'])/(1280/640)
                y1 = int(label_info['box2d']['y1'])/(720/640)
                x2 = int(label_info['box2d']['x2'])/(1280/640)
                y2 = int(label_info['box2d']['y2'])/(720/640)

                # Convert bounding box to YOLO format
                box_center_x = (x1 + x2) / 2.0
                box_center_y = (y1 + y2) / 2.0
                box_width = x2 - x1
                box_height = y2 - y1

                #filter boxes smaller than 5px 
                if box_width*box_height >= 5:

                    xc, yc, wb, hb = self.format_yolo(
                        [box_center_x, box_center_y, box_width, box_height])

                    obj_annot.append(np.array([0 , class_index, xc, yc, wb, hb]))

        if CreateMasks:

            return drivable_clustered , drivable_cluster_index ,lane_clustered , lane_cluster_index , obj_annot

        else:
            return torch.from_numpy(np.array(obj_annot))

    def format_yolo(self, box):

        xc, yc, wb, hb = box[0]/(640), box[1]/640, box[2]/640, box[3]/640
        
        return [xc, yc, wb, hb]

    def local_picks(self,array):
        last = 0
        picks = []
        for i,s in enumerate(array):
            if s - last > 15:
                picks.append(i)
            last = s
        return picks

    def clusterize(self,k):
        uniques = np.unique(k)
        picks = self.local_picks(uniques)
        if len(picks)>1:

            k = (np.ceil((k)/max(uniques[picks[0]],uniques[picks[1]-1]))).astype(np.uint8)

        elif len(picks)==1:
            
            k[k>0]=1

        else:
            k = np.zeros_like(k)

        return k

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

    # def cluster_to_embedding_feature(self,cluster_mask, size):

    #     dimensions = size ** 2

    #     cluster_mask = cluster_mask.flatten()
    #     ground = np.zeros((1, dimensions, dimensions), dtype=np.int32)

    #     # Create a 2D mask for the same instance condition
    #     same_instance_mask = (cluster_mask[:, None] == cluster_mask)

    #     # Create a 2D mask for different instance, same class condition
    #     diff_instance_same_class_mask = ~same_instance_mask & (cluster_mask[:, None] != 0)

    #     # Assign values based on conditions
    #     ground[0][same_instance_mask] = 1  # same instance
    #     ground[0][diff_instance_same_class_mask] = 2  # different instance, same class
    #     ground[0][cluster_mask == 0] = 3  # different instance, different class

    #     return ground
        
    def cluster_to_embedding_feature(self,cluster_mask, size):
        
        dimensions = size ** 2

        cluster_mask = torch.from_numpy(cluster_mask.flatten()).to('cuda')
        ground = torch.zeros((1, dimensions, dimensions), dtype=torch.int32, device='cuda')

        # Create a 2D mask for the same instance condition
        same_instance_mask = (cluster_mask.unsqueeze(1) == cluster_mask)

        # Create a 2D mask for different instance, same class condition
        diff_instance_same_class_mask = ~same_instance_mask & (cluster_mask.unsqueeze(1) != 0)

        # Assign values based on conditions
        ground[0][same_instance_mask] = 1  # same instance
        ground[0][diff_instance_same_class_mask] = 2  # different instance, same class
        ground[0][cluster_mask == 0] = 3  # different instance, different class

        return ground

    def resize_image(self, image):
        return cv2.resize(image, self.image_size)

    def resize_mask(self, mask):
        return cv2.resize(mask, (self.image_size[0]//2,self.image_size[1]//2), interpolation=cv2.INTER_NEAREST)

    def resize_obj_mask(self, obj_mask):
        return cv2.resize(obj_mask, (self.image_size[1] // 32, self.image_size[0] // 32), interpolation=cv2.INTER_NEAREST)

    def normalize_image(self, image):
        return image.astype(np.float32) / 255.0

def cluster_to_embedding_feature(cluster_mask, size , device):
    batch_size = cluster_mask.shape[0]
    dimensions = size ** 2
    ground = torch.zeros((batch_size, 1 , dimensions, dimensions), dtype=torch.int32, device=device)

    for batch in range(batch_size):
        cluster_mask_temp = cluster_mask[batch,:,:].flatten().to(device)
        # Create a 2D mask for the same instance condition
        same_instance_mask = (cluster_mask_temp.unsqueeze(1) == cluster_mask_temp)

        # Create a 2D mask for different instance, same class condition
        diff_instance_same_class_mask = ~same_instance_mask & (cluster_mask_temp.unsqueeze(1) != 0)

        # Assign values based on conditions
        ground[batch][0][same_instance_mask] = 1  # same instance
        ground[batch][0][diff_instance_same_class_mask] = 2  # different instance, same class
        ground[batch][0][cluster_mask_temp == 0] = 3  # different instance, different class

    return ground