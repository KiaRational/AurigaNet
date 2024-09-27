# 🚗 AurigaNet 🚗

[![State-of-the-Art](https://img.shields.io/badge/State--of--the--Art-%F0%9F%94%A5-green)](https://paperswithcode.com/sota/multi-task-learning-on-bdd100k)
[![Paper](https://img.shields.io/badge/Read%20the%20Paper-%F0%9F%93%96-blue)](https://arxiv.org/abs/12345678)  
**AurigaNet: A Multi-Task Real-Time Path Aggregation Network for Enhanced Urban Driving Perception and Path Estimation with Discriminative Features and Deformable Convolutions**

🔗 **Reference our paper**:  
If you use AurigaNet in your research, please consider citing our paper:

```
@article{auriganet2023,
  title={AurigaNet: A Multi-Task Real-Time Path Aggregation Network for Enhanced Urban Driving Perception and Path Estimation with Discriminative Features and Deformable Convolutions},
  author={Kiarash Ghasemzadeh, Sedigheh Dehghani},
  journal={},
  year={2023}
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
   - [BDD100K Images (10k)](https://bdd-data.berkeley.edu/download.html)
   - [BDD100K Labels](https://bdd-data.berkeley.edu/download.html)

4. Download the preprocessed dataset and create the necessary data splits:

5. Train the model:
   ```bash
   python seg_utils/train.py
   ```

---

## 🗂️ Repository Structure

```bash
code
├── Dataloader
│   ├── Preprocess.py
│   ├── Augmentation.py
│   ├── CreatMask.py
│   ├── __init__.py
│   ├── .ipynb_checkpoints
│   │   ├── __init__-checkpoint.py
│   │   ├── Preprocess-checkpoint.py
│   │   └── Dataset-checkpoint.py
│   └── Dataset.py
├── Generated.zip
├── main.py
├── Models
│   ├── MultiNet.py
│   ├── global_spatial_attention.py
│   ├── SegHead.py
│   ├── ObjNeck.py
│   ├── SegNeck.py
│   ├── __init__.py
│   ├── ObjHead.py
│   ├── .ipynb_checkpoints
│   │   ├── ObjHead-checkpoint.py
│   │   └── MultiNet-checkpoint.py
│   ├── BackBone.py
│   ├── utils.py
│   ├── local_channel_attention.py
│   ├── global_channel_attention.py
│   ├── glam.py
│   ├── SegHead_v1.py
│   ├── Neck.py
│   └── local_spatial_attention.py
├── frontend
│   ├── Netarc_1_1.svg
│   └── Netarc_1.svg
├── Toolkits
│   └── __init__.py
├── tree.py
├── artifacts
│   ├── train1.zip
│   ├── train_1
│   │   ├── last.pt
│   │   └── best.pt
│   ├── last.pt
│   ├── best.pt
│   └── train_2
│       ├── .ipynb_checkpoints
│       ├── last.pt
│       └── best.pt
├── seg_utils
│   ├── losses.py
│   ├── accuracy.py
│   ├── __init__.py
│   ├── val_yolop.py
│   ├── Parameters.py
│   ├── utils.py
│   ├── train.py
│   ├── test.py
├── PreProcess
│   └── MaskGenerator.py
├── requirements.txt


```

--- 

## 📥 Dataset Links

- [BDD100K Dataset](https://bdd-data.berkeley.edu/download.html)
- [BDD100K Annotations](https://bdd-data.berkeley.edu/download.html)
