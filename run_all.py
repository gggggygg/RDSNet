import os

import numpy as np
from pathlib import Path
import shutil
import math
import argparse
import torch
# torch.set_num_threads(4)
# import git
from tqdm import tqdm
import yaml

PLOT_BATCHES = 1000

import config

import os
from collections import OrderedDict

def predict_yolo_obb(model_path, image_path, conf=0.25, device=''):
    """
    使用训练好的YOLOv8 OBB模型进行预测
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("未找到ultralytics包，正在安装...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO

    # 设置设备
    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 预测
    print(f"开始预测，图像: {image_path}, 置信度阈值: {conf}")
    results = model.predict(
        source=image_path,
        conf=conf,
        save=True,
        device=device if device else None
    )

    return results

def validate_yolo_obb(model_path, dataset_yaml, imgsz=640, batch=16, device='', val_split=['val'], val_conf = 0.001):
    """
    使用训练好的YOLOv8 OBB模型进行验证
    """
    # 设置设备
    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    from ultralytics import YOLO

    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    model.info()

    # 验证模型
    for split in val_split:
        print(f"开始验证，{split} 数据集: {dataset_yaml}, 图像大小: {imgsz}, 批次大小: {batch}")
        val_results = model.val(
            data=dataset_yaml,
            imgsz=imgsz,
            batch=batch,
            device=device if device else None,
            # save_json=True,
            split=split,  # 指定使用 split 数据集
            conf = val_conf,   # 置信度阈值
        )

        print(f"{split} 验证结果: mAP50-95 = {val_results.box.map}")

        for key, value in val_results.results_dict.items():
            print(f"{split} {key}: {value}")

        F_measure = 2 * val_results.results_dict['metrics/precision(B)'] * val_results.results_dict['metrics/recall(B)'] / (
                    val_results.results_dict['metrics/precision(B)'] + val_results.results_dict['metrics/recall(B)'])

        print(f"{split} F-measure: {F_measure}")

        print(f'{split} mAP75: {val_results.box.map75}')

        # # 计算 FPS
        inference_time = (val_results.speed["preprocess"] + val_results.speed["inference"] + val_results.speed["postprocess"]) / 1000  # 转换为秒
        fps = 1 / inference_time / batch

        print(f"{split} 模型推理 FPS: {fps:.2f} 帧/秒")

    return

def train_yolo_obb(model_yaml, dataset_yaml, args, epochs=100, imgsz=640, batch=16, device='', save_period = -1, fine_tune=False):
    """
    使用YOLOv8 OBB模型训练
    """

    # 设置设备
    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    from ultralytics import YOLO

    print("正在加载YOLOv8 模型...")
    # 加载预训练的OBB模型
    model = YOLO(model_yaml)  # 使用YOLOv8n-obb模型

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU不可用，使用CPU训练")

    optimizer_args = {}
    if 'optimizer_dict' in args:
        optimizer_args = args.optimizer_dict

    # 开始训练
    print(f"开始训练，数据集: {dataset_yaml}, 轮数: {epochs}, 图像大小: {imgsz}, 批次大小: {batch}")

    if fine_tune==False:
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='obb',
            save=True,  # 保存最佳模型
            device=device if device else None,
            **args.augment_dict,  # 动态解包数据增强参数
            save_period = save_period,
            **optimizer_args,  # 动态解包优化器增强参数
            resume=args.resume,
        )
    else:
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='obb',
            save=True,  # 保存最佳模型
            device=device if device else None,
            **args.augment_dict,  # 动态解包数据增强参数
            save_period = save_period,
            optimizer="SGD", # 微调
            lr0=0.0001,
            momentum=0.9,
            weight_decay=0.0001,
            resume=args.resume,
        )

    # 验证模型
    print("训练完成，开始验证...")
    val_results = model.val(
        data=dataset_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device if device else None,
        amp=False,
    )

    print(f"验证结果: mAP50-95 = {val_results.box.map}")

    for key, value in val_results.results_dict.items():
        print(f"{key}: {value}")

    F_measure = 2 * val_results.results_dict['metrics/precision(B)'] * val_results.results_dict['metrics/recall(B)'] / (
            val_results.results_dict['metrics/precision(B)'] + val_results.results_dict['metrics/recall(B)'] + 10**-7)

    print(f"F-measure: {F_measure}")

    print(f'mAP75: {val_results.box.map75}')

    # # 计算 FPS
    inference_time = (val_results.speed["preprocess"] + val_results.speed["inference"] + val_results.speed["postprocess"]) / 1000  # 转换为秒
    fps = 1 / inference_time / batch

    print(f"模型推理 FPS: {fps:.2f} 帧/秒")

    print(f"验证结果: mAP50-95 = {val_results.box.map}")

    return model, results, val_results


def main():
    parser = argparse.ArgumentParser(description='使用YOLOv8 OBB训练鱼眼相机行人检测数据集')
    # parser.add_argument('--dataset_path', type=str, required=True, help='原始数据集路径')
    parser.add_argument('--dataset_path', type=str, default='path/to/dataset', help='数据集路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    parser.add_argument('--class_names', type=str, default='person', help='类别名称，多个类别用逗号分隔')
    parser.add_argument('--device', type=str, default='2', help='训练设备，例如 "0" 表示使用第一个GPU')
    parser.add_argument('--download_source', action='store_true', help='是否下载YOLOv8源码')
    parser.add_argument('--mode', type=str, choices=['train', 'validate', 'predict', 'test_fps'], default='train', help='运行模式')
    parser.add_argument('--model_path', type=str, default='', help='用于验证或预测的模型路径')
    parser.add_argument('--predict_path', type=str, default='', help='用于预测的图像或目录路径')
    parser.add_argument('--conf', type=float, default=0.25, help='预测的置信度阈值')
    parser.add_argument('--model_yaml', type=str, default='yolov8s-obb-dcnv3.yaml', help='')
    parser.add_argument('--config', type=str, default='custom_folder/val_yaml/val_RDSNet_PF_CA_in_LOAF.yaml', help='')
    parser.add_argument('--CUSTOM_OPTIMIZER_SCHEDULER', type=bool, default=False, help='')
    parser.add_argument('--CUSTOM_PLOT_NUMS', type=int, default=False, help='')
    parser.add_argument('--save_period', type=int, default=-1, help='')
    parser.add_argument('--data_yaml', type=str, default='data.yaml', help='')
    parser.add_argument('--fine_tune', type=bool, default=False, help='')
    parser.add_argument('--val_split', type=str, nargs='+', default=['val'], help='Validation split(s)')
    parser.add_argument('--val_conf', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--experiment_name', type=str, default=None, help='experiment name')
    parser.add_argument('--custom_augment', type=bool, default=False, help='')
    parser.add_argument('--resume', type=bool, default=False, help='')
    parser.add_argument('--val_plot_cnt', type=int, default=0, help='')
    parser.add_argument('--val_plot_mod', type=int, default=1000, help='')
    parser.add_argument('--val_imgsz', type=int, default=-1, help='')
    parser.add_argument('--val_plot_low', type=bool, default=False, help='')
    parser.add_argument('--val_plot_low_limit', type=float, default=0.5, help='')
    parser.add_argument('--rapid_transforms', type=bool, default=False, help='')
    parser.add_argument('--loaf_transforms', type=bool, default=False, help='')
    parser.add_argument('--coco', type=bool, default=False, help='')
    parser.add_argument('--limit_whr', type=bool, default=False, help='')
    parser.add_argument('--keep_label', type=bool, default=False, help='')
    parser.add_argument('--custom_trainer', type=bool, default=False, help='')
    parser.add_argument('--custom_scale', type=float, default=0.5, help='')
    parser.add_argument('--area_thr', type=float, default=0, help='')
    parser.add_argument('--pretrain_model_yaml', type=str, default='', help='')
    parser.add_argument('--mosaic_fish', type=bool, default=False, help='')
    parser.add_argument('--cut_scale_up', type=bool, default=False, help='')
    parser.add_argument('--limit_loss_wh', type=int, default=0, help='')
    parser.add_argument('--radial_distortion', type=bool, default=False, help='')
    parser.add_argument('--radial_distortion_k', type=float, default=0.1, help='')
    parser.add_argument('--keep_label_loaf', type=bool, default=False, help='')
    parser.add_argument('--split_metrics', type=bool, default=False, help='')
    parser.add_argument('--Conv_Sparse_sampling', type=bool, default=False, help='')
    parser.add_argument('--coco_pretrain_augment', type=bool, default=False, help='')
    parser.add_argument('--Conv_Sparse_sampling_traing', type=bool, default=False, help='')
    parser.add_argument('--fitness_focus_ap50', type=bool, default=False, help='')
    parser.add_argument('--loaf_transforms_ori', type=bool, default=False, help='')
    parser.add_argument('--location_eval', type=bool, default=False, help='')
    parser.add_argument('--close_degrees', type=bool, default=False, help='')
    parser.add_argument('--close_degrees_epoch', type=int, default=50, help='')
    parser.add_argument('--nw', type=int, default=None, help='')
    parser.add_argument('--split_distance_metrics', type=bool, default=False, help='')
    parser.add_argument('--split_seen_un_seen', type=bool, default=False, help='')
    parser.add_argument('--save_all_val_results', type=bool, default=False, help='')
    parser.add_argument('--Conv_sampling_only_inner', type=bool, default=False, help='')


    #参数已修改，请搜索： modified plot_batches

    args = parser.parse_args()

    args.augment_dict = {}

    if args.config:
        # 2️⃣ 读取 YAML 文件
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)

        print('yaml_config: ', yaml_config)

        # 3️⃣ 使用 YAML 更新 `args`
        for key, value in yaml_config.items():
            if hasattr(args, key):  # 仅更新 `args` 中已有的参数
                setattr(args, key, value)
            elif key == 'optimizer_dict':
                # 如果是 optimizer_dict，直接赋值
                args.optimizer_dict = value
            elif key == 'augment_dict':
                args.augment_dict = value

    if args.experiment_name is not None:
        print(args.experiment_name)

    # 输出参数
    print('args: ', args)

    config.CUSTOM_OPTIMIZER_SCHEDULER = args.CUSTOM_OPTIMIZER_SCHEDULER
    config.CUSTOM_PLOT_NUMS = args.CUSTOM_PLOT_NUMS
    config.TRAIN_YMAL = args.data_yaml[:-5]
    config.val_plot_cnt = args.val_plot_cnt
    config.val_plot_mod = args.val_plot_mod
    config.val_imgsz = args.val_imgsz
    config.val_plot_low = args.val_plot_low
    config.val_plot_low_limit = args.val_plot_low_limit
    config.rapid_transforms = args.rapid_transforms
    config.coco = args.coco
    config.limit_whr = args.limit_whr
    config.keep_label = args.keep_label
    config.custom_trainer = args.custom_trainer
    config.custom_scale = args.custom_scale
    config.imgsz = args.imgsz
    config.random_imgsz = args.imgsz
    config.loaf_transforms = args.loaf_transforms
    config.area_thr = args.area_thr
    config.mosaic_fish = args.mosaic_fish
    config.cut_scale_up = args.cut_scale_up
    config.limit_loss_wh = args.limit_loss_wh
    config.radial_distortion = args.radial_distortion
    config.radial_distortion_k = args.radial_distortion_k
    config.keep_label_loaf = args.keep_label_loaf
    config.split_metrics = args.split_metrics
    config.Conv_Sparse_sampling = args.Conv_Sparse_sampling
    config.coco_pretrain_augment = args.coco_pretrain_augment
    config.Conv_Sparse_sampling_traing = args.Conv_Sparse_sampling_traing
    config.fitness_focus_ap50 = args.fitness_focus_ap50
    config.loaf_transforms_ori = args.loaf_transforms_ori
    config.location_eval = args.location_eval
    config.close_degrees = args.close_degrees
    config.close_degrees_epoch = args.close_degrees_epoch
    config.nw = args.nw
    config.split_distance_metrics = args.split_distance_metrics
    config.split_seen_un_seen = args.split_seen_un_seen
    config.save_all_val_results = args.save_all_val_results
    config.Conv_sampling_only_inner = args.Conv_sampling_only_inner

    config.rapid_batsize = args.batch

    if args.mode == 'train':
        config.training = True

    if args.mode == 'train':

        yaml_path = os.path.join(args.dataset_path, args.data_yaml)

        # 训练模型
        model, result, val_results = train_yolo_obb(args.model_yaml, yaml_path, args, args.epochs, args.imgsz, args.batch, args.device, args.save_period, args.fine_tune)

        print(f"训练完成！最佳模型保存在: {os.path.join('runs', 'obb', 'obb', 'weights', 'best.pt')}  ,model: {args.model_yaml}")

    elif args.mode == 'validate':
        if not args.model_path:
            if not os.path.exists(args.model_path):
                raise ValueError(f"未找到模型文件: {args.model_path}，请指定正确的模型路径")

        # 验证模型
        yaml_path = os.path.join(args.dataset_path, args.data_yaml)

        val_results = validate_yolo_obb(args.model_path, yaml_path, args.imgsz, args.batch, args.device, args.val_split, args.val_conf)

        print(f'验证完成！配置: {args.config}')

    elif args.mode == 'predict':
        if not args.model_path:
            args.model_path = os.path.join('runs', 'obb', 'obb', 'weights', 'best.pt')
            if not os.path.exists(args.model_path):
                raise ValueError(f"未找到模型文件: {args.model_path}，请指定正确的模型路径")

        if not args.predict_path:
            raise ValueError("请指定用于预测的图像或目录路径")

        # 预测
        results = predict_yolo_obb(args.model_path, args.predict_path, args.conf, args.device)
        print(f"预测结果已保存在: {os.path.join('runs', 'obb', 'predict')}")


if __name__ == "__main__":
    config.cache_path = "/home/wei/RDSNet_cache"  #
    # config.cache_path = "datasets/" #"/you/cache/path"  # 修改为你的缓存路径
    main()