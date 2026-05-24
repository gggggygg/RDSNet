
# ckpt_id = [802, 724, 519, 548, 547, 545, 556, 553, 551, 639, 86, 638, 498, 657, 497, 659, 671, 552, 550, 549, 557, 555, 554, 546, 541, 562]
# ckpt_id.sort()
# print(ckpt_id)


import numpy
print(numpy.__version__)

assert 0

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import time
import torch.nn as nn
from ultralytics.nn.modules.block import rConv, rConv_fix, rConv_dcnv4, dConv2, r_customConv_test


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.pre_x = None
        self.c2 = c2
        self.s = s

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
        # y = torch.zeros(x.shape[0], self.c2, x.shape[2]//self.s, x.shape[2]//self.s, device='cuda', requires_grad=True)
        # return self.act(self.bn(y))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

from dcn_v2 import dcn_v2_conv, DCNv2, DCN
class dConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

if __name__ == '__main__':
    #测试函数
    def benchmark_conv(input_size, out_channel = 64, stride = 1, device='cuda:0', num_iter=2000):
        in_channel = input_size[1]
        print(f"Benchmarking Conv and rConv with input size: {input_size}, in_channel: {in_channel}, out_channel: {out_channel}, stride: {stride}")

        model = Conv(in_channel, out_channel, k=3,s = stride).to(device)
        model.eval()
        x = torch.zeros(input_size).to(device)

        # 预热
        for _ in range(10):
            _ = model(x)

        # 正式计时
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iter):
            _ = model(x)
        torch.cuda.synchronize()
        end = time.time()

        avg_time = (end - start) / num_iter * 1000  # ms
        print(f"Conv Input: {input_size}, Avg Inference Time: {avg_time:.3f} ms")


        # model = rConv(in_channel, 64, k=3, s = stride, in_h = input_size[-2], in_w = input_size[-1], batch_size=input_size[0]).to(device)
        model = r_customConv_test(in_channel, out_channel, k=3, s=stride, in_h = input_size[-2], in_w = input_size[-1], batch_size=input_size[0]).to(device)
        # model = rConv_fix(in_channel, 64, kernel_size=3).to(device)
        # model = rConv_dcnv4(in_channel, 64, k=3, in_h=input_size[-2], in_w=input_size[-1], batch_size=input_size[0]).to(device)
        # model = dConv2(in_channel, 64, k=3, in_h=input_size[-2], in_w=input_size[-1], batch_size=input_size[0]).to(device)
        model.eval()
        x = torch.zeros(input_size).to(device)

        # 预热
        for _ in range(10):
            _ = model(x)

        # 正式计时
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iter):
            _ = model(x)
        torch.cuda.synchronize()
        end = time.time()

        avg_time = (end - start) / num_iter * 1000  # ms
        print(f"rConv Input: {input_size}, Avg Inference Time: {avg_time:.3f} ms")


    # 运行测试
    for size in [1024]:
        benchmark_conv((1, 3, size, size), out_channel=64, stride = 1) #0.514 ms 1.236 ms

    for size in [512]:
        benchmark_conv((1, 64, size, size), out_channel=128, stride=1) #0.495 ms 1.399 ms

    for size in [256]:
        benchmark_conv((1, 128, size, size), out_channel=128, stride=1) #0.416 ms 2.516 ms

    for size in [256]:
        benchmark_conv((1, 128, size, size), out_channel=256, stride=2) #0.221 ms  0.630 ms

    for size in [128]:
        benchmark_conv((1, 256, size, size), out_channel=256, stride=1) #0.246 ms 1.285 ms

    for size in [128]:
        benchmark_conv((1, 256, size, size), out_channel=512, stride=2)# 0.178 ms  0.349 ms

    for size in [64]:
        benchmark_conv((1, 512, size, size), out_channel=512, stride=1)#0.171 ms 0.645 ms

    for size in [64]:
        benchmark_conv((1, 512, size, size), out_channel=512, stride=2)#0.167 ms 0.277 ms

    for size in [32]:
        benchmark_conv((1, 512, size, size), out_channel=512, stride=1)#0.158 ms 0.251 ms

#试一下load ckpt


# import cv2
# image_path = '/ric_nas/Overhead_fisheye/LOAF_YOLO/images/train/0005_03465.jpg'
# img = cv2.imread(image_path)
# if img is None:
#     print(f"Warning: failed to read {image_path}")



# import matplotlib.pyplot as plt
# plt.close('all')

#from pycocotools.cocoeval import COCOeval

# from ultralytics import YOLO
#
# # 加载 YOLO 模型
# model = YOLO("yolov8n.pt")
#
# # 输出模型信息（包含参数量和 FLOPS）
# model.info()




# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#
# # from ultralytics.models.yolo.detect import DetectionTrainer
# #
# # # args = dict(model="yolov8m-custom.yaml", data="custom_coco.yaml", epochs=3)
# # args = dict(model="yolov8m.yaml", data="coco8.yaml", epochs=3)
# # trainer = DetectionTrainer(overrides=args)
# # trainer.train()
#
# from ultralytics.models.yolo.obb import OBBTrainer
#
# augment = dict(degrees=90.0, translate=0, scale=0)
#
# args = dict(model='yolov8n-obb.yaml', data="dota8.yaml", epochs=3)
# args.update(augment)
#
# trainer = OBBTrainer(overrides=args)
# trainer.train()

# import cv2
# import numpy as np
# pts = np.array([[0,0],[0,4],[1,4],[1,0]])
# (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
# print(cx, cy, w, h, angle)
#
# print(len(False))

# def train_yolo_obb_from_source(dataset_yaml, epochs=100, imgsz=640, batch=16, device=''):
#     """
#     使用YOLOv8 OBB模型训练
#     """
#     from ultralytics.models.yolo.obb import OBBTrainer
#
#     print("正在加载YOLOv8 OBB模型...")
#     # 加载预训练的OBB模型
#
#     augment = dict(degrees=90.0, translate=0, scale=0)
#     args = dict(model = 'yolov8n-obb.yaml', data = dataset_yaml, epochs = epochs, imgsz = imgsz, batch = batch, name = 'habbof_obb',
#                 patience = 50, save = True, device=device if device else None)
#     args.update(augment)
#
#     trainer = OBBTrainer(overrides=args)
#
#
#     # 检查CUDA是否可用
#     if torch.cuda.is_available():
#         print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
#     else:
#         print("GPU不可用，使用CPU训练")
#
#     # 开始训练
#     print(f"开始训练，数据集: {dataset_yaml}, 轮数: {epochs}, 图像大小: {imgsz}, 批次大小: {batch}")
#
#     # trainer.train()
#
#     # 验证模型
#     print("训练完成，开始验证...")
#     model = trainer.model  # 访问已训练的 YOLO 模型
#     # 进行模型验证
#     print("训练完成，开始验证...")
#     val_results = model.val(
#         data=dataset_yaml,
#         imgsz=imgsz,
#         batch=batch,
#         device=device if device else None
#     )
#
#     print(f"验证结果: mAP50-95 = {val_results.box.map}")
#
#     return model, val_results