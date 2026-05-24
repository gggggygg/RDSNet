import os
import numpy as np
from pathlib import Path
import shutil
import math
import argparse
import torch
# import git
from tqdm import tqdm
# import yaml
from collections import defaultdict
import json
from PIL import Image
from ultralytics.utils import ops


def extract_rbox_labels_and_images_to_yolo_xyxyxyxy(image_files, json_files, output_path):

    imgid2anns = defaultdict(list)
    imgfile2imgid = {}
    imgid2imfile = {}
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for ann in data['annotations']:
                imgid2anns[ann['image_id']].append(ann)

            for img in data['images']:
                imgfile2imgid[img['file_name']] = img['id']
                imgid2imfile[img['id']] = img['file_name']

    output_path_img = output_path[0]
    output_path_label = output_path[1]

    debug_cnt = 30

    # 处理训练集
    print("处理训练集...")

    for img_path in tqdm(image_files):
        if img_path.name not in imgfile2imgid:
            continue

        # if debug_cnt<0:
        #     break
        # debug_cnt -= 1

        # 复制图像
        shutil.copy(img_path, os.path.join(output_path_img, img_path.name))

        # 读取图像
        img = Image.open(img_path)

        # 获取尺寸 (宽, 高)
        im_width, im_height = img.size

        imgid = imgfile2imgid[img_path.name]
        labels = imgid2anns[imgid]

        with open(os.path.join(output_path_label, img_path.stem + '.txt'), 'w') as f_out:
            for label in labels:
                rotated_box = label['rotated_box'][:-1]
                label_rotated_box = np.array(rotated_box)
                label_rotated_box_xyxy = ops.loaf_xywh2xyxyxyxy(torch.from_numpy(label_rotated_box)).numpy()
                parts = label_rotated_box_xyxy.reshape(-1, 8).tolist()[0]


                # parts = label['rotated_box'][:-1]
                #
                parts = [p/im_width if i%2==0 else p/im_height for i, p in enumerate(parts)]

                class_id = 0
                    # 写入YOLO OBB格式
                f_out.write(f"{class_id} {' '.join([f'{p:.6f}' for p in parts])}\n")


def LOAF_convert_rbox_labels_to_yolo_xyxyxyxy(dataset_path, output_path):
    """
    将LOAF_YOLO数据集的标签转换为YOLO OBB格式
    将 class x_center y_center width height angle 转换为 class x1 y1 x2 y2 x3 y3 x4 y4
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'test'), exist_ok=True)

    # 获取图像文件
    train_image_files = list(Path(os.path.join(dataset_path, 'images/resolution_1k/train')).glob('*.jpg'))
    val_image_files = list(Path(os.path.join(dataset_path, 'images/resolution_1k/val')).glob('*.jpg'))
    test_image_files = list(Path(os.path.join(dataset_path, 'images/resolution_1k/test')).glob('*.jpg'))

    train_json_files = [os.path.join(dataset_path, 'annotations/resolution_1k/instances_train.json')]
    val_json_files = [os.path.join(dataset_path, 'annotations/resolution_1k/instances_val.json')]
    test_json_files = [os.path.join(dataset_path, 'annotations/resolution_1k/instances_test.json')]


    if len(test_image_files) == 0:
        raise ValueError(f"在 {os.path.join(dataset_path, 'images')} 中未找到图像文件")

    extract_rbox_labels_and_images_to_yolo_xyxyxyxy(train_image_files, train_json_files, [os.path.join(output_path, 'images', 'train'), os.path.join(output_path, 'labels', 'train')])
    extract_rbox_labels_and_images_to_yolo_xyxyxyxy(val_image_files, val_json_files, [os.path.join(output_path, 'images', 'val'), os.path.join(output_path, 'labels', 'val')])
    extract_rbox_labels_and_images_to_yolo_xyxyxyxy(test_image_files, test_json_files, [os.path.join(output_path, 'images', 'test'), os.path.join(output_path, 'labels', 'test')])

    print(f"{len(val_image_files)} 个验证样本 和 {len(test_image_files)} 个测试样本")
    return len(val_image_files), len(test_image_files)



def CEPDOF_convert_labels_to_yolo_obb(dataset_path, output_path):
    """
    将HABBOF_YOLO数据集的标签转换为YOLO OBB格式
    将 class x_center y_center width height angle 转换为 class x1 y1 x2 y2 x3 y3 x4 y4
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'All_off'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'Edge_cases'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'High_activity'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'IRfilter'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'IRill'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'Lunch1'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'Lunch2'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'Lunch3'), exist_ok=True)

    os.makedirs(os.path.join(output_path, 'labels', 'All_off'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'Edge_cases'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'High_activity'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'IRfilter'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'IRill'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'Lunch1'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'Lunch2'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'Lunch3'), exist_ok=True)

    # 获取所有图像文件
    image_files_All_off = list(Path(os.path.join(dataset_path, 'All_off')).glob('*.jpg'))
    image_files_Edge_cases = list(Path(os.path.join(dataset_path, 'Edge_cases')).glob('*.jpg'))
    image_files_High_activity = list(Path(os.path.join(dataset_path, 'High_activity')).glob('*.jpg'))
    image_files_IRfilter = list(Path(os.path.join(dataset_path, 'IRfilter')).glob('*.jpg'))
    image_files_IRill = list(Path(os.path.join(dataset_path, 'IRill')).glob('*.jpg'))
    image_files_Lunch1 = list(Path(os.path.join(dataset_path, 'Lunch1')).glob('*.jpg'))
    image_files_Lunch2 = list(Path(os.path.join(dataset_path, 'Lunch2')).glob('*.jpg'))
    image_files_Lunch3 = list(Path(os.path.join(dataset_path, 'Lunch3')).glob('*.jpg'))

    json_files_All_off = os.path.join(dataset_path, 'annotations', 'All_off.json')
    json_files_Edge_cases = os.path.join(dataset_path, 'annotations', 'Edge_cases.json')
    json_files_High_activity = os.path.join(dataset_path, 'annotations', 'High_activity.json')
    json_files_IRfilter = os.path.join(dataset_path, 'annotations', 'IRfilter.json')
    json_files_IRill = os.path.join(dataset_path, 'annotations', 'IRill.json')
    json_files_Lunch1 = os.path.join(dataset_path, 'annotations', 'Lunch1.json')
    json_files_Lunch2 = os.path.join(dataset_path, 'annotations', 'Lunch2.json')
    json_files_Lunch3 = os.path.join(dataset_path, 'annotations', 'Lunch3.json')

    if len(image_files_All_off) == 0:
        raise ValueError(f"在 {os.path.join(dataset_path, 'images')} 中未找到图像文件")

    def xywha_to_xyxyxyxy(im_width, im_height, x_center, y_center, width, height, angle_deg):
        """
        将中心点、宽高和角度转换为四个角点坐标
        angle_deg: 角度，单位为度
        """
        x_center/=im_width
        y_center/=im_height
        width/=im_width
        height/=im_height


        angle_rad = math.radians(angle_deg)
        half_w = width / 2
        half_h = height / 2

        # 计算旋转后的四个角点
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 顺时针方向的四个角点相对于中心的偏移
        pts = [
            (-half_w, -half_h),  # 左上
            (half_w, -half_h),  # 右上
            (half_w, half_h),  # 右下
            (-half_w, half_h)  # 左下
        ]

        # 旋转并平移到中心点
        rotated_pts = []
        for px, py in pts:
            rx = px * cos_a - py * sin_a + x_center
            ry = px * sin_a + py * cos_a + y_center
            # 确保坐标在[0,1]范围内
            rx = max(0, min(1, rx))
            ry = max(0, min(1, ry))
            rotated_pts.extend([rx, ry])

        return rotated_pts

    def extract_labels_and_images(image_files, json_file, output_path):

        imgid2anns = defaultdict(list)

        # for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for ann in data['annotations']:
                imgid2anns[ann['image_id']].append(ann)

        debug_cnt = 30

        split = str(image_files[0]).split('/')[-2]

        # 处理数据集
        print(f"处理数据集...{split}")
        for img_path in tqdm(image_files):
            # if debug_cnt<0:
            #     break
            # debug_cnt -= 1

            # 复制图像
            shutil.copy(img_path, os.path.join(output_path, 'images', split, img_path.name))

            # 读取图像
            img = Image.open(img_path)

            # 获取尺寸 (宽, 高)
            im_width, im_height = img.size

            labels = imgid2anns[img_path.stem]

            with open(os.path.join(output_path, 'labels', split, img_path.stem + '.txt'), 'w') as f_out:

                # # 处理标签
                for label in labels:
                    parts = label['bbox']
                    if len(parts) >= 5:  # 确保有足够的元素
                        class_id = 0
                        x_center = float(parts[0])
                        y_center = float(parts[1])
                        width = float(parts[2])
                        height = float(parts[3])
                        angle = float(parts[4])

                        # 转换为四个角点坐标
                        corner_points = xywha_to_xyxyxyxy(im_width, im_height, x_center, y_center, width, height, angle)

                        # 写入YOLO OBB格式
                        f_out.write(f"{class_id} {' '.join([f'{p:.6f}' for p in corner_points])}\n")

    extract_labels_and_images(image_files_All_off, json_files_All_off, output_path)
    extract_labels_and_images(image_files_Edge_cases, json_files_Edge_cases, output_path)
    extract_labels_and_images(image_files_High_activity, json_files_High_activity, output_path)
    extract_labels_and_images(image_files_IRfilter, json_files_IRfilter, output_path)
    extract_labels_and_images(image_files_IRill, json_files_IRill, output_path)
    extract_labels_and_images(image_files_Lunch1, json_files_Lunch1, output_path)
    extract_labels_and_images(image_files_Lunch2, json_files_Lunch2, output_path)
    extract_labels_and_images(image_files_Lunch3, json_files_Lunch3, output_path)



def HABBOF_convert_labels_to_yolo_obb(dataset_path, output_path):
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'Lab1'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'Lab2'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'Meeting1'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'Meeting2'), exist_ok=True)

    os.makedirs(os.path.join(output_path, 'labels', 'Lab1'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'Lab2'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'Meeting1'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'Meeting2'), exist_ok=True)

    # 获取所有图像文件
    image_files = [list(Path(os.path.join(dataset_path, 'Lab1')).glob('*.jpg')), list(Path(os.path.join(dataset_path, 'Lab2')).glob('*.jpg')),\
                   list(Path(os.path.join(dataset_path, 'Meeting1')).glob('*.jpg')), list(Path(os.path.join(dataset_path, 'Meeting2')).glob('*.jpg'))]

    json_files = [os.path.join(dataset_path, 'annotations', 'Lab1.json'), os.path.join(dataset_path, 'annotations', 'Lab2.json'),\
                  os.path.join(dataset_path, 'annotations', 'Meeting1.json'), os.path.join(dataset_path, 'annotations', 'Meeting2.json')]

    if len(image_files[0]) == 0:
        raise ValueError(f"在 {os.path.join(dataset_path, 'images')} 中未找到图像文件")

    def xywha_to_xyxyxyxy(im_width, im_height, x_center, y_center, width, height, angle_deg):
        """
        将中心点、宽高和角度转换为四个角点坐标
        angle_deg: 角度，单位为度
        """
        x_center/=im_width
        y_center/=im_height
        width/=im_width
        height/=im_height


        angle_rad = math.radians(angle_deg)
        half_w = width / 2
        half_h = height / 2

        # 计算旋转后的四个角点
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 顺时针方向的四个角点相对于中心的偏移
        pts = [
            (-half_w, -half_h),  # 左上
            (half_w, -half_h),  # 右上
            (half_w, half_h),  # 右下
            (-half_w, half_h)  # 左下
        ]

        # 旋转并平移到中心点
        rotated_pts = []
        for px, py in pts:
            rx = px * cos_a - py * sin_a + x_center
            ry = px * sin_a + py * cos_a + y_center
            # 确保坐标在[0,1]范围内
            rx = max(0, min(1, rx))
            ry = max(0, min(1, ry))
            rotated_pts.extend([rx, ry])

        return rotated_pts

    def extract_labels_and_images(image_files, json_file, output_path):

        imgid2anns = defaultdict(list)

        # for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for ann in data['annotations']:
                imgid2anns[ann['image_id']].append(ann)

        debug_cnt = 30

        split = str(image_files[0]).split('/')[-2]

        # 处理数据集
        print(f"处理数据集...{split}")
        for img_path in tqdm(image_files):
            # if debug_cnt<0:
            #     break
            # debug_cnt -= 1

            # 复制图像
            shutil.copy(img_path, os.path.join(output_path, 'images', split, img_path.name))

            # 读取图像
            img = Image.open(img_path)

            # 获取尺寸 (宽, 高)
            im_width, im_height = img.size

            labels = imgid2anns[img_path.stem]

            with open(os.path.join(output_path, 'labels', split, img_path.stem + '.txt'), 'w') as f_out:

                # # 处理标签
                for label in labels:
                    parts = label['bbox']
                    if len(parts) >= 5:  # 确保有足够的元素
                        class_id = 0
                        x_center = float(parts[0])
                        y_center = float(parts[1])
                        width = float(parts[2])
                        height = float(parts[3])
                        angle = float(parts[4])

                        # 转换为四个角点坐标
                        corner_points = xywha_to_xyxyxyxy(im_width, im_height, x_center, y_center, width, height, angle)

                        # 写入YOLO OBB格式
                        f_out.write(f"{class_id} {' '.join([f'{p:.6f}' for p in corner_points])}\n")

    for i in range(4):
        extract_labels_and_images(image_files[i], json_files[i], output_path)


def COCO_extract_labels_and_images(image_files, json_files, output_path):

    imgid2anns = defaultdict(list)
    imgfile2imgid = {}
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for ann in data['annotations']:
                imgid2anns[ann['image_id']].append(ann)

            for img in data['images']:
                imgfile2imgid[img['file_name']] = img['id']

    output_path_img = output_path[0]
    output_path_label = output_path[1]

    # debug_cnt = 30

    # 处理训练集
    print("处理训练集...")
    for img_path in tqdm(image_files):
        imgid = imgfile2imgid[img_path.name]
        labels = imgid2anns[imgid]

        if any(ann['iscrowd'] for ann in labels):
            continue

        if not any(ann['category_id'] == 1 for ann in labels):
            continue

        # if debug_cnt<0:
        #     break
        # debug_cnt -= 1

        # 复制图像
        shutil.copy(img_path, os.path.join(output_path_img, img_path.name))

        # 读取图像
        # img = Image.open(img_path)

        with Image.open(img_path) as img:
            im_width, im_height = img.size

        # 获取尺寸 (宽, 高)
        # im_width, im_height = img.size

        li=0
        with open(os.path.join(output_path_label, img_path.stem + '.txt'), 'w') as f_out:

        # # 处理标签
            for label in labels:
                parts = label['bbox']
                area = parts[2]*parts[3]/im_height/im_height

                if label['category_id'] != 1 or area<=0.001:
                    continue

                li+=1
                if li>=50:
                    print(img_path)
                    break

                point1 = [parts[0]/im_width, parts[1]/im_height]
                point2 = [parts[0]/im_width,(parts[1]+parts[3])/im_height]
                point3 = [(parts[0]+parts[2])/im_width,(parts[1]+parts[3])/im_height]
                point4 = [(parts[0]+parts[2])/im_width,parts[1]/im_height]

                point = point1 + point2 + point3 + point4

                class_id = 0
                    # 写入YOLO OBB格式
                f_out.write(f"{class_id} {' '.join([f'{p:.6f}' for p in point])}\n")


def COCO2017_convert_labels_to_yolo_obb(dataset_path, output_path):
    """
    将HABBOF_YOLO数据集的标签转换为YOLO OBB格式
    将 class x_center y_center width height angle 转换为 class x1 y1 x2 y2 x3 y3 x4 y4
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)

    # 获取图像文件
    train_image_files = list(Path(os.path.join(dataset_path, 'train2017')).glob('*.jpg'))

    train_json_files = [os.path.join(dataset_path, 'annotations/instances_train2017.json')]

    if len(train_image_files) == 0:
        raise ValueError(f"在 {os.path.join(dataset_path, 'train2017')} 中未找到图像文件")

    COCO_extract_labels_and_images(train_image_files, train_json_files, [os.path.join(output_path, 'images', 'train'), os.path.join(output_path, 'labels', 'train')])

    print(f"已转换 {len(train_image_files)} 个训练样本")
    return len(train_image_files)


if __name__ == '__main__':
    #LOAF dataset
    # LOAF_convert_rbox_labels_to_yolo_xyxyxyxy('path/to/LOAF','datasets/LOAF_YOLO_xyxy_from_rbox')

    #CEPDOF dataset
    # CEPDOF_convert_labels_to_yolo_obb('datasets/raw/CEPDOF', 'datasets/CEPDOF_YOLO')
    # CEPDOF_convert_labels_to_yolo_obb('path/to/CEPDOF', 'datasets/CEPDOF_YOLO')

    #HABBOF dataset
    # HABBOF_convert_labels_to_yolo_obb('path/to/HABBOF', 'dataset/HABBOF_YOLO')
    # HABBOF_convert_labels_to_yolo_obb('datasets/raw/HABBOF', 'datasets/HABBOF_YOLO')

    #COCO2017 dataset
    COCO2017_convert_labels_to_yolo_obb('datasets/raw/COCO2017', 'datasets/COCO2017_YOLO')