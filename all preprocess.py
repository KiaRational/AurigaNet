
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import numpy.typing as npt
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel

DictStrAny = Dict[str, Any]  # type: ignore[misc]

NDArrayF64 = npt.NDArray[np.float64]
NDArrayI32 = npt.NDArray[np.int32]
NDArrayU8 = npt.NDArray[np.uint8]

class ImageSize(BaseModel):
    """Define image size in config."""

    width: int
    height: int
class Poly2D(BaseModel):
    """Polygon or polyline 2D."""

    vertices: List[Tuple[float, float]]
    types: str
    closed: bool

def poly_to_patch(
    vertices: List[Tuple[float, float]],
    types: str,
    color: Tuple[float, float, float],
    closed: bool,
) -> mpatches.PathPatch:
    """Draw polygons using the Bezier curve."""
    moves = {"L": Path.LINETO, "C": Path.CURVE4}
    points = list(vertices)
    codes = [moves[t] for t in types]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.LINETO)

    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=color if closed else "none",
        edgecolor=color,
        lw=0 if closed else 1,
        alpha=1,
        antialiased=False,
        snap=True,
    )
def poly2ds_to_mask(shape: ImageSize, poly2d: List[Poly2D]) -> NDArrayU8:
    """Converting Poly2D to mask."""
    fig = plt.figure(facecolor="0")
    fig.set_size_inches(
        shape.width / fig.get_dpi(), shape.height / fig.get_dpi()
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, shape.width)
    ax.set_ylim(0, shape.height)
    ax.set_facecolor((0, 0, 0, 0))
    ax.invert_yaxis()

    for poly in poly2d:
        ax.add_patch(
            poly_to_patch(
                poly["vertices"],
                poly["types"],
                color=(1, 1, 1),
                closed=poly["closed"],
            )
        )

    fig.canvas.draw()
    mask: NDArrayU8 = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
    mask = mask.reshape((shape.height, shape.width, -1))[..., 0]
    plt.close()
    return mask


import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt



class CustomDataLoader:
    def __init__(self, data_path, label_path, image_size=(720, 1280), normalize=True, class_mapping=None, batch_size=16):
        self.data_path = data_path
        self.image_size = image_size
        self.normalize = normalize
        self.class_mapping = class_mapping
        self.label_path = label_path
        self.batch_size = batch_size
        self.annotations = self.load_data()
        self.num_samples = len(self.annotations)

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


    def process_batch(self, batch_annotations):
        batch_images = []
        batch_drivable_masks = []
        batch_lane_masks = []

        for annotation in batch_annotations:
            image_name = annotation['name']
            image_path = os.path.join(self.data_path, image_name)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)

                drivable_mask, lane_mark_mask = self.create_masks(image, annotation)

                # Resize drivable_mask and lane_mark_mask to the desired image size
                drivable_mask = self.resize_mask(drivable_mask)
                lane_mark_mask = self.resize_mask(lane_mark_mask)

                # Append processed data to lists
                batch_images.append(image)
                batch_drivable_masks.append(drivable_mask[np.newaxis])  # Add a new axis to make it (1, H, W)
                batch_lane_masks.append(lane_mark_mask[np.newaxis])  # Add a new axis to make it (1, H, W)

            else:
              continue
            return batch_images, batch_drivable_masks, batch_lane_masks

    def create_masks(self, image, annotation):
        lane_masks = np.zeros_like(image[:,:,0])
        drivable_area_masks = np.zeros_like(image[:,:,0])


        for label_info in annotation['labels']:
            category = label_info['category']
            if category=='lane' and label_info['attributes']['laneDirection']=='parallel' :
              poly2d = label_info['poly2d']
              image_size = ImageSize(width=1280, height=720)
              lane_mask = poly2ds_to_mask(image_size, poly2d)
              lane_masks += lane_mask

            if category=='drivable area':
              poly2d = label_info['poly2d']
              image_size = ImageSize(width=1280, height=720)
              drivable_area_mask = poly2ds_to_mask(image_size, poly2d)
              drivable_area_masks+= drivable_area_mask
            # else:
            #     class_index = self.class_mapping[category]
            #     x1 = int(label_info['box2d']['x1'])
            #     y1 = int(label_info['box2d']['y1'])
            #     x2 = int(label_info['box2d']['x2'])
            #     y2 = int(label_info['box2d']['y2'])

            #     # Convert bounding box to YOLO format
            #     box_center_x = (x1 + x2) / 2.0
            #     box_center_y = (y1 + y2) / 2.0
            #     box_width = x2 - x1
            #     box_height = y2 - y1

            #     grid_cell_x = int(box_center_x / self.image_size[1] * (self.image_size[1] // 32))
            #     grid_cell_y = int(box_center_y / self.image_size[0] * (self.image_size[0] // 32))

            #     obj_mask[grid_cell_y, grid_cell_x, 0] = 1.0
            #     obj_mask[grid_cell_y, grid_cell_x, 1:5] = [box_center_x, box_center_y, box_width, box_height]
            #     obj_mask[grid_cell_y, grid_cell_x, 5 + class_index] = 1.0

        return drivable_area_masks, lane_masks

    def resize_image(self, image):
        return cv2.resize(image, self.image_size)

    def resize_mask(self, mask):
        return cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

    def resize_obj_mask(self, obj_mask):
        return cv2.resize(obj_mask, (self.image_size[1] // 32, self.image_size[0] // 32), interpolation=cv2.INTER_NEAREST)

    def normalize_image(self, image):
        return image.astype(np.float32) / 255.0


import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path, label_path, image_size=(720, 1280), normalize=True, class_mapping=None, batch_size=8):
        self.data_loader = CustomDataLoader(data_path, label_path, image_size, normalize, class_mapping, batch_size)
        self.batch_size = batch_size
        self.image_size =  image_size

    def __len__(self):
        return self.data_loader.num_samples // self.batch_size

    def __getitem__(self, index):

        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.data_loader.num_samples)
        batch_annotations = self.data_loader.annotations[start_idx:end_idx]

        batch_images, batch_drivable_masks, batch_lane_masks = self.data_loader.process_batch(batch_annotations)

        if len(batch_images) == 0:  # Handle empty batch
            batch_images = np.zeros((1, *self.image_size, 3), dtype=np.float32)
            batch_drivable_masks = np.zeros((1, *self.image_size), dtype=np.float32)
            batch_lane_masks = np.zeros((1, *self.image_size), dtype=np.float32)

        # Convert lists to numpy arrays and stack the masks
        batch_images = np.array(batch_images)
        batch_drivable_masks = np.stack(batch_drivable_masks, axis=0)
        batch_lane_masks = np.stack(batch_lane_masks, axis=0)
        batch_images = torch.tensor(batch_images.transpose(0, 3, 1, 2), dtype=torch.float32)
        batch_drivable_masks = torch.tensor(batch_drivable_masks, dtype=torch.float32)  # No need for unsqueeze
        batch_lane_masks = torch.tensor(batch_lane_masks, dtype=torch.float32)  # No need for unsqueeze

        return batch_images, batch_drivable_masks, batch_lane_masks

# Usage
data_path = "/content/dataset/bdd100k/bdd100k/images/100k/train/"
label_path = '/content/dataset/bdd100k_labels_release/bdd100k/labels/'
class_mapping = {
    'traffic_light': 0,
    'traffic_sign': 1,
    'car': 2
}
dataset = CustomDataset(data_path, label_path, image_size=(720, 1280), normalize=True, class_mapping=class_mapping)
batch_size = 8  # Choose your desired batch size
train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Example of how to iterate through the data loader during training
for i, (batch_images, drivable_masks, lane_masks) in enumerate(train_data_loader):
    # Here you can use the batch_images, drivable_masks, and lane_masks for training
    # Remember that the batch size is determined by the 'batch_size' parameter you set above
    # Perform your training process here
    try:
        print(batch_images.shape)
        print(drivable_masks.shape)
        print(lane_masks.shape)
        print(i * batch_size)  # Print the starting index of each batch
    except:
        continue
