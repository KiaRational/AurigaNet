import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
import os
import sys
import matplotlib.pyplot as plt
import time
from torchvision.ops import nms
import matplotlib.patches as patches
import matplotlib
from torchmetrics.detection import MeanAveragePrecision
import random
from collections import Counter
# Add the project root to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import specific modules
from   Models.MultiNet import MultiNet
from seg_utils.Parameters import Parameters
from Dataloader.Preprocess import CustomDataLoader
import seg_utils.utils as utils

matplotlib.use('GTK3Agg')


def activation(x):
    return 1e-7 + (1 - 2 * 1e-7) * (0.5 + torch.arctan(x)/torch.tensor(np.pi))

class SegmentationTester:
    def __init__(self, model_path, image_path,image_name, device='cuda'):
        self.device = device
        self.P = Parameters()
        # Load the model checkpoint and extract the state_dict
        self.model = MultiNet()  # Replace with your segmentation model class
        self.model.load_state_dict(torch.load(model_path, map_location="cuda")) 
        self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode
        self.anchors = torch.tensor([
                    [[10, 13], [16, 30], [33, 23]],  # P3/8
                    [[30, 61], [62, 45], [59, 119]],  # P4/16
                    [[116, 90], [156, 198], [373, 326]] # P5/32
                ])
        self.anchors = torch.tensor(self.anchors).float().view(3, -1, 2) / torch.tensor([8,16,32]).repeat(6, 1).T.reshape(3, 3, 2)
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.bdd = [
        'car' ,
        'rider' ,
        'pedestrian' ,
        'truck' ,
        'bus' ,
        'train' ,
        'motorcycle' ,
        'bicycle' ,
        'traffic light' ,
        'traffic sign' 
        ]

        self.image_path = image_path

        val_data_path = self.P.val_data_path
        val_label_path = self.P.val_label_path
        save_path = self.P.save_path
        class_mapping = self.P.class_mapping
        epochs = self.P.epoch_number
        self.val_dataset = CustomDataLoader(data_path = val_data_path, label_path = val_label_path, image_size=(640, 640), normalize=True, class_mapping=class_mapping,train=False)
        for i in range(len(self.val_dataset.annotations)):
            if image_name == self.val_dataset.annotations[i]['name']:
                _  , _ , _  , self.objects_annotations = self.val_dataset.process(self.val_dataset.annotations[i])
                print("_____________found_____________________")
    def local_picks(self,array):
        last = 0
        picks = []
        for i,s in enumerate(array):
            if s - last > 5:
                picks.append(i)
            last = s
        return picks

    def clusterize(self,k):
        uniques = np.unique(k)
        print(uniques)
        picks = self.local_picks(uniques)
        print(picks)
        if len(picks)>1:
            k = (np.ceil((k)/(max(uniques[picks[0]],uniques[picks[1]-1])))).astype(np.uint8)
            print(picks)
        elif len(picks)==1:
            
            k[k>0]=1

        else:
            k = np.zeros_like(k)

        return k
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (640,640))
        # Normalize pixel values to [0, 1]
        image =image.astype(np.float32) / 255.0
        image = self.transforms(image).unsqueeze(0).to(self.device)
        
        return image

    def non_max_suppression(self,batch_bboxes, iou_threshold, threshold, max_detections=300, tolist=True):

        """new_bboxes = []
        for box in bboxes:
            if box[1] > threshold:
                box[3] = box[0] + box[3]
                box[2] = box[2] + box[4]
                new_bboxes.append(box)"""

        bboxes_after_nms = []
        for boxes in batch_bboxes:
            boxes = torch.masked_select(boxes, boxes[..., 1:2] > threshold).reshape(-1, 6)

            # from xywh to x1y1x2y2

            boxes[..., 2:3] = boxes[..., 2:3] - (boxes[..., 4:5] / 2)
            boxes[..., 3:4] = boxes[..., 3:4] - (boxes[..., 5:] / 2)
            boxes[..., 5:6] = boxes[..., 5:6] + boxes[..., 3:4]
            boxes[..., 4:5] = boxes[..., 4:5] + boxes[..., 2:3]

            indices = nms(boxes=boxes[..., 2:] + boxes[..., 0:1], scores=boxes[..., 1], iou_threshold=iou_threshold)
            boxes = boxes[indices]

            # sorts boxes by objectness score but it's already done internally by torch metrics's nms
            # _, si = torch.sort(boxes[:, 1], dim=0, descending=True)
            # boxes = boxes[si, :]

            if boxes.shape[0] > max_detections:
                boxes = boxes[:max_detections, :]

            bboxes_after_nms.append(
                boxes.tolist() if tolist else boxes
            )

        return bboxes_after_nms if tolist else torch.cat(bboxes_after_nms, dim=0)


    def cells_to_bboxes(self,predictions, anchors, strides, is_pred=False, to_list=True):
        num_out_layers = len(predictions)
        grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
        anchor_grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
        strides = torch.tensor(strides).to("cuda")
        anchors = anchors.to("cuda")
        all_bboxes = []
        for i in range(num_out_layers):
            bs, naxs, ny, nx, _ = predictions[i].shape
            stride = strides[i]
            grid[i], anchor_grid[i] = self.make_grids(anchors, naxs, ny=ny, nx=nx, stride=stride, i=i)
            if is_pred:
                # formula here: https://github.com/ultralytics/yolov5/issues/471
                #xy, wh, conf = predictions[i].sigmoid().split((2, 2, 80 + 1), 4)
                layer_prediction = predictions[i].sigmoid()
                obj = layer_prediction[..., 4:5]
                xy = (2 * (layer_prediction[..., 0:2]) + grid[i].to("cuda") - 0.5) * stride
                wh = ((2*layer_prediction[..., 2:4])**2) * anchor_grid[i]
                best_class = torch.argmax(layer_prediction[..., 5:], dim=-1).unsqueeze(-1)

            else:
                predictions[i] = predictions[i].to(config.DEVICE, non_blocking=True)
                obj = predictions[i][..., 4:5]
                xy = (predictions[i][..., 0:2] + grid[i]) * stride
                wh = predictions[i][..., 2:4] * stride
                best_class = predictions[i][..., 5:6]

            scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).reshape(bs, -1, 6)

            all_bboxes.append(scale_bboxes)
        print(all_bboxes)
        return torch.cat(all_bboxes, dim=1).tolist() if to_list else torch.cat(all_bboxes, dim=1)

    def make_grids(self,anchors, naxs, stride, nx=20, ny=20, i=0):

        x_grid = torch.arange(nx)
        x_grid = x_grid.repeat(ny).reshape(ny, nx)

        y_grid = torch.arange(ny).unsqueeze(0)
        y_grid = y_grid.T.repeat(1, nx).reshape(ny, nx)

        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        xy_grid = xy_grid.expand(1, naxs, ny, nx, 2)
        anchor_grid = (anchors[i]*stride).reshape((1, naxs, 1, 1, 2)).expand(1, naxs, ny, nx, 2)

        return xy_grid, anchor_grid

    def plot_image(self,image, boxes, labels):
        """Plots predicted bounding boxes on the image"""
        cmap = plt.get_cmap("tab20b")
        class_labels = labels
        colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
        im = np.array(image)

        # Create figure and axes
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(im)

        # box[0] is x midpoint, box[2] is width
        # box[1] is y midpoint, box[3] is height

        # Create a Rectangle patch
        for box in boxes:
            assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
            class_pred = box[0]
            bbox = box[2:]

            # FOR MY_NMS attempts, also rect = patches.Rectangle box[2] becomes box[2] - box[0] and box[3] - box[1]
            upper_left_x = max(bbox[0], 0)
            upper_left_x = min(upper_left_x, im.shape[1])
            lower_left_y = max(bbox[1], 0)
            lower_left_y = min(lower_left_y, im.shape[0])

            """upper_left_x = max(box[0] - box[2] / 2, 0)
            upper_left_x = min(upper_left_x, im.shape[1])
            lower_left_y = max(box[1] - box[3] / 2, 0)
            lower_left_y = min(lower_left_y, im.shape[0])"""

            rect = patches.Rectangle(
                (upper_left_x, lower_left_y),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=2,
                edgecolor=colors[int(class_pred)],
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.text(
                upper_left_x,
                lower_left_y,
                s=f"{class_labels[int(class_pred)]}: {box[1]:.2f}",
                color="white",
                verticalalignment="top",
                bbox={"color": colors[int(class_pred)], "pad": 0},
            )
        # plt.show()
        return im
    def max_pooling_2d(self , input_array, pool_size=(2, 2)):

        input_height, input_width = input_array.shape
        output_height = input_height // pool_size[0]
        output_width = input_width // pool_size[1]

        # Reshape input_array to be divided into pool_size blocks
        reshaped_array = input_array[:output_height*pool_size[0], :output_width*pool_size[1]].reshape(
            output_height, pool_size[0], output_width, pool_size[1]
        )

        # Apply max pooling using numpy's max function along specified axes
        pooled_array = reshaped_array.mean(axis=(1, 3))

        return pooled_array
    def one_hot_encode(self,arr):
        unique_values = np.unique(arr)  # Find unique values in the input array
        num_classes = len(unique_values)  # Number of unique values = number of classes

        # Create an empty one-hot encoded array
        one_hot = np.zeros((arr.shape[0], arr.shape[1], num_classes), dtype=int)

        # Fill in the one-hot array based on the unique values
        for i, value in enumerate(unique_values):
            one_hot[:, :, i] = (arr == value)

        return one_hot

    def mAP(self, label, pred, class_num=10):

        # Convert label and pred to numpy arrays
        pred = np.array(pred)
        label_boxes = label[..., 2:] * 640
        label_boxess = np.zeros_like(label_boxes.numpy())
        label_boxess[:, 0] = label_boxes[..., 0] - label_boxes[..., 2] / 2
        label_boxess[:, 1] = label_boxes[..., 1] - label_boxes[..., 3] / 2
        label_boxess[:, 2] = label_boxes[..., 0] + label_boxes[..., 2] / 2
        label_boxess[:, 3] = label_boxes[..., 1] + label_boxes[..., 3] / 2
        label_classes = label[..., 1]
        pred_classes = pred[0, :, 0]
        pred_boxes = pred[0, :, 2:]
        pred_scores = pred[0, :, 1]
        matches = []
        np.set_printoptions(suppress=True)

        # Initialize TPFP50 and TPFP75 arrays
        TPFP50 = np.zeros((class_num, 2))  # |FP|TP|
        TPFP75 = np.zeros((class_num, 2))

        # Count occurrences of predicted classes
        counter = Counter(pred_classes)
        pred_counts = list(counter.items())

        for pred_count in pred_counts:
            TPFP50[int(pred_count[0]), 0] = int(pred_count[1])
            TPFP75[int(pred_count[0]), 0] = int(pred_count[1])


        for i, pred_box in enumerate(pred_boxes):
            ious = [utils.iou_width_height(torch.tensor(pred_box), torch.tensor(label_box))
                    for label_box in label_boxess]

            index = np.array(ious).argmax()

            if ious[index] >= 0.5:
                if pred_classes[i] == label_classes[index]:
                    TPFP50[int(pred_classes[i]), 1] += 1  # count as TP
                    TPFP50[int(pred_classes[i]), 0] -= 1
                    if ious[index] < 0.75:
                        matches.append((int(pred_classes[i]), i, index, ious[index]))
            else:
                if pred_classes[i] == label_classes[index]:
                    TPFP50[int(pred_classes[i]), 0] += 1  # count as FP

            if ious[index] >= 0.75:
                if pred_classes[i] == label_classes[index]:
                    TPFP75[int(pred_classes[i]), 1] += 1
                    TPFP75[int(pred_classes[i]), 0] -= 1
                    matches.append((int(pred_classes[i]), i, index, ious[index]))
            else:
                if pred_classes[i] == label_classes[index]:
                    TPFP75[int(pred_classes[i]), 0] += 1  # count as FP
        return TPFP50, TPFP75




    def test(self):

        with torch.no_grad():
            image = self.preprocess_image(self.image_path)
            # image = image.transpose((2, 0, 1))
            print(image.shape)
            output = self.model(image)
            
            out = output[1]
            obj_out = output[2]
            output=activation(output[0][0])
        # Convert the outrain_data_path = P.train_data_path

        S = [8, 16, 32]
        boxes = self.cells_to_bboxes(obj_out, self.anchors, S, to_list=False, is_pred=True)
        boxes = self.non_max_suppression(boxes, iou_threshold=0.5, threshold=.5, max_detections=300)
        # self.mAP(self.objects_annotations,boxes)
        im = self.plot_image(image[0].permute(1, 2, 0).to("cpu"), boxes[0],self.bdd)
        output_array = output
        out = out[:,:,:,:].cpu().numpy()
        result = np.sum(out**2, axis=1, keepdims=True)
        drivable_area_mask = output_array[0,:,:].cpu().numpy()
        lane_mask = output_array[1,:,:].cpu().numpy()
        imgg = image[0].permute(1, 2, 0).to("cpu")

        img = result[0,0,:,:]
        
        drivable_area_mask = cv2.resize((drivable_area_mask>0.5).astype(np.float32),(640,640),cv2.INTER_LINEAR)
        lane_mask = cv2.resize((lane_mask>0.5).astype(np.float32),(640,640),cv2.INTER_LINEAR)
        eroded = cv2.erode(drivable_area_mask.astype(np.float32), None, iterations=5)

        # green_mask = np.zeros_like(imgg)
        # green_mask[:, :] = (0, 255, 0)  # Green color
        red_mask = np.zeros_like(imgg)
        red_mask[:, :] = (255, 0, 0)  # Red color

        # overlay_area = cv2.bitwise_and(green_mask, green_mask, mask=eroded.astype(np.uint8))
        overlay_lane = cv2.bitwise_and(red_mask, red_mask, mask=lane_mask.astype(np.uint8))

        # result = cv2.addWeighted(imgg.numpy(), 1, overlay_area, 1, 0)
        result = cv2.addWeighted(imgg.numpy(), 1, overlay_lane, 1, 0)
        
        # eroded = cv2.erode(drivable_area_mask, None, iterations=3)

        img = ((((cv2.resize(drivable_area_mask,(40,40),cv2.INTER_NEAREST))>0.5).astype(np.float32))*img).astype(np.uint8)
        # plt.imshow(img)
        # plt.show()
        img = self.clusterize(img)

        Feature_masks=self.one_hot_encode(img)

        for i in range(Feature_masks.shape[2]):
            if i == 0:
                continue
            green_mask = np.zeros_like(imgg)
            green_mask[:, :] = (i*5/255, (i-2)*90/255, (i-1)*50/255)  # Green color
            mask = ((cv2.resize((Feature_masks[:,:,i]).astype(np.float32),(640,640),cv2.INTER_LINEAR))>0.5)*eroded

            overlay_area = cv2.bitwise_and(green_mask, green_mask, mask=mask.astype(np.uint8))
            result = cv2.addWeighted(result, 1, overlay_area, 1, 0)

        plt.imshow(result)
        plt.show()
        # cv2.imwrite("/home/kia/Multi-Task-Network/Saved/output_network.png",result)
        # plt.savefig("/home/kia/Multi-Task-Network/Saved/output_network_as.png")
        # print(result)
        return output_array

# Example usage:
if __name__ == "__main__":

    model_path = '/home/kia/Multi-Task-Network/Saved/14_0.23993450045585632_AurigaNet.pth'
    image_path = '/home/kia/BDD100K/bdd100k_images_100k_5/bdd100k/images/100k/val/b3e137e5-fe7ef981.jpg'
    image_name = 'b3e137e5-fe7ef981.jpg'
    
    tester = SegmentationTester(model_path, image_path, image_name)
    result = tester.test()
    print(result.shape)  # Print the shape of the segmentation mask


