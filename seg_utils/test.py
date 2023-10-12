import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
import os
import sys
import matplotlib.pyplot as plt
# Add the project root to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import specific modules
from   Models.MultiNet import MultiNet
def activation(x):
    return 1e-7 + (1 - 2 * 1e-7) * (0.5 + torch.arctan(x)/torch.tensor(np.pi))

class SegmentationTester:
    def __init__(self, model_path, image_path, device='cuda'):
        self.device = device

        # Load the model checkpoint and extract the state_dict
        self.model = MultiNet()  # Replace with your segmentation model class
        self.model.load_state_dict(torch.load(model_path, map_location="cuda")) 
        self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        self.image_path = image_path

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (640,640))
        # Normalize pixel values to [0, 1]
        image =image.astype(np.float32) / 255.0
        image = self.transforms(image).unsqueeze(0).to(self.device)

        return image

    def test(self):
        with torch.no_grad():
            image = self.preprocess_image(self.image_path)
            # image = image.transpose((2, 0, 1))
            print(image.shape)
            output = self.model(image)
            output=activation(output[0][0])
        # Convert the output to a NumPy array
        output_array = output

        drivable_area_mask = output_array[0,:,:].cpu().numpy()
        lane_mask = output_array[1,:,:].cpu().numpy()
        # lane_mask[lane_mask<0.1]=0
        # drivable_area_mask[drivable_area_mask>0.3]=1
        plt.imshow(drivable_area_mask)
        plt.show()
        return output_array

# Example usage:
if __name__ == "__main__":

    model_path = '/home/kia/Multi-Task-Network/Saved/29_0.021674420218914747_AurigaNet.pth'
    image_path = '/home/kia/BDD100K/bdd100k_images_100k_5/bdd100k/images/100k/val/bd57e60e-187054e3.jpg'

    tester = SegmentationTester(model_path, image_path)
    result = tester.test()
    print(result.shape)  # Print the shape of the segmentation mask


