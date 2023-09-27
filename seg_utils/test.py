import torch
import numpy as np
import cv2
from torchvision import transforms
import os
import sys
# Add the project root to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import specific modules
from   Models.MultiNet import MultiNet

class SegmentationTester:
    def __init__(self, model_path, image_path, device='cuda'):
        self.device = device

        # Load the model checkpoint and extract the state_dict
        self.model = MultiNet()  # Replace with your segmentation model class
        self.model= load_state_dict(torch.load(model_path, map_location="cuda")) 

        self.model.eval()  # Set the model to evaluation mode

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.image_path = image_path

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = self.transforms(image).unsqueeze(0).to(self.device)
        
        # Normalize pixel values to [0, 1]
        image = image / 255.0

        return image

    def test(self):
        with torch.no_grad():
            image = self.preprocess_image(self.image_path)
            output = self.model(image)

        # Convert the output to a NumPy array
        output_array = output.cpu().numpy()

        return output_array

# Example usage:
if __name__ == "__main__":
    model_path = '/home/kia/Multi-Task-Network/Weights/49_0.5514169096946716_AurigaNet.pth'
    image_path = '/home/kia/BDD100K/bdd100k_images_100k_5/bdd100k/images/100k/train/0004974f-05e1c285.jpg'

    tester = SegmentationTester(model_path, image_path)
    result = tester.test()
    print(result.shape)  # Print the shape of the segmentation mask
