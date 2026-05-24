import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
from ultralytics import YOLO
# from models.yolo import Model  # 以ultralytics/YOLOv3改写版为例
model = YOLO("yolov3.yaml")  # 加载YOLOv3结构

# 方式一：直接统计参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params/1e6:.2f}M")
print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
