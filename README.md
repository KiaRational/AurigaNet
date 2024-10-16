# 🚗 AurigaNet 🚗

[![State-of-the-Art](https://img.shields.io/badge/State--of--the--Art-%F0%9F%94%A5-green)](https://paperswithcode.com/sota/multi-task-learning-on-bdd100k)
[![Paper](https://img.shields.io/badge/Read%20the%20Paper-%F0%9F%93%96-blue)](https://arxiv.org/abs/12345678)  

[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.7+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.8+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)

**AurigaNet: A Multi-Task Real-Time Path Aggregation Network for Enhanced Urban Driving Perception and Drivable Paths Estimation with Discriminative Features**

🔗 **Reference our paper**:  
If you use AurigaNet in your research, please consider citing our paper:

```
@article{auriganet2024,
  title={AurigaNet: A Multi-Task Real-Time Path Aggregation Network for Enhanced Urban Driving Perception and Drivable Paths Estimation with Discriminative Features},
  author={Kiarash Ghasemzadeh, Sedigheh Dehghani},
  journal={},
  year={2024}
}
```

---

The cutting-edge **AurigaNet** developed for **Autonomous Vehicles** revolutionizes the way self-driving cars understand the environment. 🌍 This unified framework seamlessly integrates various tasks such as **lane detection**, **object recognition**, and **drivable area detection** into a single network. 🛣️🚘

![Network Architecture](https://github.com/KiaRational/Multi-Task-Network/blob/main/frontend/architecture.svg)

---

## ✨ Key Features

- **Lane Detection**: 🚦 Accurately detects lanes in challenging driving conditions.
- **Object Recognition**: 🚗 Recognizes and classifies cars, trucks, and buses.
- **Drivable Area Detection**: 📍 Segments the drivable area for safe navigation.
- **Drivable Area Instance Segmentation**: ⛍ Segments the instances of drivable area for path detection.
---

## 📊 Model Performance

### 🛣️ Drivable Area Segmentation

AurigaNet sets the **state-of-the-art** in drivable area segmentation, achieving the highest IoU and accuracy:

| **Model**               | **Input Size** | **IoU** | **Accuracy** |
|-------------------------|----------------|---------|--------------|
| FCN (R-50-D8)           | 769×769        | 74.8%   | 90.9%        |
| PSPNet (R-50-D8)        | 769×769        | 83.5%   | 94.9%        |
| YOLOP                   | 640×640        | 84.5%   | 97.3%        |
| HybridNets              | 640×640        | 83.4%   | 92.0%        |
| Swin Transformer (T)    | 512×1024       | 81.8%   | 94.8%        |
| **AurigaNet (Ours)**    | 640×640        | **85.2%**   | **97.7%**    |

### 🛤️ Lane Detection

AurigaNet demonstrates significant improvements in lane detection with a **30% higher IoU** compared to existing models:

| **Model**      | **IoU**  | **Accuracy** |
|----------------|----------|--------------|
| ENet           | 14.64%   | 34.12%       |
| SCNN           | 15.84%   | 35.79%       |
| ENet-SAD       | 16.02%   | 36.56%       |
| YOLOP          | 26.20%   | 70.50%       |
| HybridNets     | 31.60%   | 85.4%        |
| **AurigaNet**  | **60.80%** | **98.77%**   |

### 🚦 Traffic Object Detection

AurigaNet excels in traffic object detection, outperforming other models:

| **Model**      | **mAP@0.5:0.95**  | **Recall** |
|----------------|-------------------|------------|
| YOLOv5s        | 42.5%             | 62.5%      |
| YOLOP          | 43.1%             | 89.2%      |
| HybridNets     | 44.7%             | **92.8%**  |
| **AurigaNet**  | **47.6%**         | 75.9%      |

### ⏱️ Time Comparison

AurigaNet balances model complexity with inference speed, providing competitive real-time performance:

| **Model**      | **Param (M)** | **FPS (RTX 4080)** | **Inference Time (Jetson Nano)** |
|----------------|---------------|-------------------|-----------------------------------|
| YOLOP          | 7.90          | 362               | 4.002                          |
| HybridNets     | 12.83         | 139               | 1.986                               |
| **AurigaNet**  | 9.09          | 217               | 5.077                               |

---

## 🖼️ Results Compare

![Figure 8](https://github.com/user-attachments/assets/a5bf5be1-e240-432f-9ae9-1f4ca29388ff)

## 🎥 Demo Test

Here’s a demo GIF showcasing **AurigaNet** in action, with lane detection, object recognition, and drivable area segmentation:

![Demo Test GIF](https://github.com/KiaRational/Multi-Task-Network/blob/main/frontend/demo_input.gif)
![input Test GIF](https://github.com/KiaRational/Multi-Task-Network/blob/main/frontend/demo1_auriganet.gif)

---

## 🏗️ How to Train the Model

### Prerequisites
- Python 3.8+
- PyTorch 1.7+
- CUDA 11.0+ (for GPU training)

### Steps to Train
1. Clone the repository:
   ```bash
   git clone https://github.com/KiaRational/AurigaNet.git
   cd AurigaNet
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the **BDD100K dataset**:
   - [BDD100K Images (10k)](https://drive.google.com/file/d/1WMU_ey0wpuUKxf_PeC-yzaZXu_6VTe-w/view?usp=drive_link)
   - [BDD100K Labels](https://drive.google.com/file/d/1JIEcsZouh1N11dSpqrR9NCBn1aUJDE6_/view?usp=drive_link)
   - [BDD100K Segmentation Labels](https://drive.google.com/file/d/1h4kYSH-BJEFJPNARsD_-KbOFjCtikh5r/view?usp=drive_link)

4. Download the preprocessed dataset and create the necessary data splits and modify seg_utils/parameters.py file:

5. Train the model:
   ```bash
   python seg_utils/train.py
   ```

---

## 🗂️ Repository Structure

```bash
.
├── all preprocess.py
├── artifacts
├── Dataloader
│   ├── Augmentation.py
│   ├── CreatMask.py
│   ├── Dataset.py
│   ├── __init__.py
│   ├── new.png
│   ├── Preprocess.py
├── frontend
│   ├── architecture.svg
│   ├── AurigaNet_arc-1.png
│   ├── AurigaNet_arc-1.svg
│   ├── demo1_auriganet.gif
│   ├── demo1.gif
│   ├── demo_input.gif
│   ├── Netarc_1_1.svg
├── __init__.py
├── Models
│   ├── auriganet.onnx
│   ├── BackBone.py
│   ├── glam.py
│   ├── global_channel_attention.py
│   ├── global_spatial_attention.py
│   ├── __init__.py
│   ├── local_channel_attention.py
│   ├── local_spatial_attention.py
│   ├── MultiNet.py
│   ├── Neck.py
│   ├── ObjHead.py
│   ├── ObjNeck.py
│   ├── SegHead.py
│   ├── SegHead_v1.py
│   ├── SegNeck.py
│   └── utils.py
├── PreProcess
│   └── MaskGenerator.py
├── README.md
├── requirements.txt
├── seg_utils
│   ├── accuracy.py
│   ├── __init__.py
│   ├── losses.py
│   ├── Parameters.py
│   ├── test.py
│   ├── train.py
│   ├── utils.py
│   └── val_yolop.py
└── Toolkits
    └── __init__.py

```

--- 

## 📥 Dataset Links

- [BDD100K Images (10k)](https://drive.google.com/file/d/1WMU_ey0wpuUKxf_PeC-yzaZXu_6VTe-w/view?usp=drive_link)
- [BDD100K Labels](https://drive.google.com/file/d/1JIEcsZouh1N11dSpqrR9NCBn1aUJDE6_/view?usp=drive_link)
- [BDD100K Segmentation Labels](https://drive.google.com/file/d/1h4kYSH-BJEFJPNARsD_-KbOFjCtikh5r/view?usp=drive_link)
