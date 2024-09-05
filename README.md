# Multi-Task DNN for Autonomous Vehicles

The cutting-edge Multi-Task Deep Neural Network (DNN) developed for Autonomous Vehicles revolutionizes how self-driving cars understand their environment. This unified framework seamlessly integrates various tasks such as lane detection, object recognition, and drivable area detection into a single neural network.

![Network Architecture](https://github.com/KiaRational/Multi-Task-Network/blob/main/frontend/architecture.svg)

## Key Features

- **Lane Detection**: Accurately detects lanes in challenging driving conditions.
- **Object Recognition**: Identifies and classifies objects like vehicles, pedestrians, and traffic signs.
- **Drivable Area Detection**: Segments the drivable area for autonomous vehicle navigation.

## Model Performance

### Drivable Area Segmentation

AurigaNet outperforms state-of-the-art models in drivable area segmentation, achieving the highest IoU and accuracy:

| **Model**               | **Input Size** | **IoU** | **Accuracy** |
|-------------------------|----------------|---------|--------------|
| FCN (R-50-D8)           | 769×769        | 74.8%   | 90.9%        |
| PSPNet (R-50-D8)        | 769×769        | 83.5%   | 94.9%        |
| YOLOP                   | 640×640        | 84.5%   | 97.3%        |
| HybridNets              | 640×640        | 83.4%   | 92.0%        |
| Swin Transformer (T)    | 512×1024       | 81.8%   | 94.8%        |
| **AurigaNet**    | 640×640        | **85.2%**   | **97.7%**    |

### Lane Detection

AurigaNet demonstrates significant improvements in lane detection with a 30% higher IoU compared to existing models:

| **Model**      | **IoU**  | **Accuracy** |
|----------------|----------|--------------|
| ENet           | 14.64%   | 34.12%       |
| SCNN           | 15.84%   | 35.79%       |
| ENet-SAD       | 16.02%   | 36.56%       |
| YOLOP          | 26.20%   | 70.50%       |
| HybridNets     | 31.60%   | 85.4%        |
| **AurigaNet**  | **60.80%** | **98.77%**   |

### Traffic Object Detection

AurigaNet achieves a higher mAP score compared to baseline models, excelling in traffic object detection:

| **Model**      | **mAP@0.5:0.95**  | **Recall** |
|----------------|-------------------|------------|
| YOLOv5s        | 42.5%             | 62.5%      |
| YOLOP          | 43.1%             | 89.2%      |
| HybridNets     | 44.7%             | **92.8%**  |
| **AurigaNet**  | **47.6%**         | 75.9%      |

### Time Comparison

AurigaNet offers a balance between model complexity and inference speed, providing competitive real-time performance:

| **Model**      | **Param (M)** | **FPS (RTX 4080)** | **Inference Time (Jetson Nano)** |
|----------------|---------------|-------------------|-----------------------------------|
| YOLOP          | 7.90          | 362               | n/a                               |
| HybridNets     | 12.83         | 13.94             | n/a                               |
| **AurigaNet**  | 9.09          | 217               | n/a                               |

## Demo Test

To give you a real-time look at how AurigaNet works, here’s a demo GIF showcasing lane detection, object recognition, and drivable area segmentation in action:

![Demo Test GIF](https://github.com/KiaRational/Multi-Task-Network/blob/main/frontend/demo1_input.gif)
![input Test GIF](https://github.com/KiaRational/Multi-Task-Network/blob/main/frontend/demo1_auriganet.gif)

