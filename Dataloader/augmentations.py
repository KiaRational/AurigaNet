# Import the libraries we will use

import albumentations as A
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# from .Augmentation import RandomShadow


class Custom_Augment:
    def __init__(self):
        self.hflip_p       = 0.5
        self.vflip_p       = 0.5
        self.random_rain_p = 0.5

    def create_pipeline(self):
        # Create a pipeline of transformations

        augmentation_pipeline = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.3),
        albumentations.RandomRain(p=1.0)
        ])
        return augmentation_pipeline

    # def perform(self, ):


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    cv2.imshow("augmented", img)


# Create a function for transforming images using multiple transformations

augmentation_pipeline = A.Compose([
        # RandomShadow(prob=0.50),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.GaussNoise(var_limit=(0, 255), p=0.1),
            A.MotionBlur(blur_limit=17, p=0.1),
            # A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.1), # problem with object
            A.ImageCompression(quality_lower=39, quality_upper=60, p=0.2),
            A.HorizontalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.075, rotate_limit=20, p=0.2,border_mode=cv2.BORDER_CONSTANT)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# augmentation_pipeline.append([A.VerticalFlip(p=0.3), A.RandomRain(p=1.0)])

print(f"the pipeline for augmentation:  {augmentation_pipeline}")

def multiple_augment(aug_pipeline, image):
       image_array = np.array(image)
       augmented_img = aug_pipeline(image=image_array)['image']
       
       return augmented_img
    #    return Image.fromarray(augmented_img)


'''
test augmentations on a data
'''

if __name__ == "__main__":

    # cap = cv2.VideoCapture(0) # from webcam
    # sample_data_path = r"E:\everything_related_to_programming\everything_related_to_AI\paper\AurigaNet\model\AurigaNew\object_utils\sample_data\images\hamrahAval-1402-04-13-1.jpg"
    # img = cv2.imread(sample_data_path) # an image

    category_ids = [0]
    category_id_to_name = {0 : "no_vehicles"}

    while True:
        img = cv2.imread("./test_data/1.jpg")
        lane_mask = cv2.imread("./test_data/1_mask.jpg")
        drivable_mask = cv2.imread("./test_data/1_drivable_mask.jpg")

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lane_clustered_mask = cv2.cvtColor(lane_mask, cv2.COLOR_BGR2GRAY)
        drivable_clustered_mask = cv2.cvtColor(drivable_mask, cv2.COLOR_BGR2GRAY)

        bboxes = [[640, 220, 80, 100]] # bounding boxes for object

        masks = [drivable_clustered_mask, lane_clustered_mask]

        augmented = augmentation_pipeline(image=image, masks=masks, bboxes=bboxes, category_ids=category_ids)

        image = augmented['image']

        drivable_clustered_mask = augmented['masks'][0]

        lane_clustered_mask = augmented['masks'][1]

        bboxes = augmented['bboxes']
        category_ids = augmented['category_ids']

        visualize(
            image,
            bboxes,
            category_ids,
            category_id_to_name,
        )

        cv2.imshow("drivable", drivable_clustered_mask)
        cv2.imshow("lane", lane_clustered_mask)

        # # Apply multiple transformations to the image
        # augmented_image = multiple_augment(augmentation_pipeline, img)

        # cv2.imshow("aug out: ", augmented_image)

        k = cv2.waitKey(0) 
        if k == ord("q"):
            cv2.destroyAllWindows()
            break

    # while True:
    #     ret, frame = cap.read()

    #     k = cv2.waitKey(1)
    #     # Apply multiple transformations to the image
    #     augmented_image = multiple_augment(augmentation_pipeline, frame)

    #     if k == ord("q"):
    #         cap.close()
    #         cv2.destroyAllWindows()
    #         break

    #     cv2.imshow("aug out: ", augmented_image)

