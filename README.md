# TSNet-Project
Official implementation of the paper: "TSNet: A Multi-modal Deep Learning Framework for Subtyping Appendicitis"
# TSNet: A Multi-modal Deep Learning Framework for Subtyping Appendicitis

This repository contains the official PyTorch implementation of the paper:
**"TSNet: A Multi-modal Deep Learning Framework for Subtyping Appendicitis: Integrating Ultrasound Images, Handcrafted Features, and Clinical Data"**

## 📄 Abstract
Accurate preoperative differentiation of appendicitis subtypes—particularly the clinically challenging Acute Exacerbation of Chronic Appendicitis (AEC)—remains a significant diagnostic dilemma. To address this, we propose **TSNet**, a multi-modal Two-Stream fusion framework integrating ultrasound images (ResNet-50 + SAM) and clinical features (MLP).

## 🛠️ Requirements
The code is implemented in Python 3.11 using PyTorch.
* torch
* torchvision
* scikit-learn
* numpy
* pandas

## 🚀 Usage
1. **Data Preparation**: Due to privacy restrictions, the dataset is not provided. Users should prepare their own ultrasound images and tabular data.
2. **Training**: Run `train.py` to train the model.
   ```bash
   python train.py
