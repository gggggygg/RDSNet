import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
import torch

def draw_rotated_boxes_from_tensor(pred_boxes, gt_boxes, path = '/ric_nas/Overhead_fisheye/tiny_val_YOLO/images/one/ytb_000001.jpg', save_dir = None, conf = None):
    from ultralytics.utils import ops

    xyxy1 = ops.xywhr2xyxyxyxy(pred_boxes).cpu().numpy()
    xyxy2 = ops.xywhr2xyxyxyxy(gt_boxes).cpu().numpy()

    xyxy1 = xyxy1.reshape(-1,8).tolist()
    xyxy2 = xyxy2.reshape(-1,8).tolist()

    draw_rotated_boxes(path, xyxy1, xyxy2, save_dir = save_dir, conf = conf)


def draw_rotated_boxes_from_tensor_loaf_box(pred_boxes, gt_boxes, path = '/ric_nas/Overhead_fisheye/tiny_val_YOLO/images/one/ytb_000001.jpg', save_dir = None, conf = None):
    # from ultralytics.utils import ops

    xyxy1 = loaf_xyxy2xyxyxyxy(pred_boxes).cpu().numpy()
    xyxy2 = loaf_xyxy2xyxyxyxy(gt_boxes).cpu().numpy()

    xyxy1 = xyxy1.reshape(-1,8).tolist()
    xyxy2 = xyxy2.reshape(-1,8).tolist()

    draw_rotated_boxes(path, xyxy1, xyxy2, save_dir = save_dir, conf = conf)

def draw_rotated_boxes_from_tensor_boxes(pred_boxes, gt_boxes, path = '/ric_nas/Overhead_fisheye/tiny_val_YOLO/images/one/ytb_000001.jpg', save_dir = None, conf = None):

    xyxy1 = pred_boxes.cpu().numpy()
    xyxy2 = gt_boxes.cpu().numpy()

    xyxy1 = xyxy1.reshape(-1,8).tolist()
    xyxy2 = xyxy2.reshape(-1,8).tolist()

    draw_rotated_boxes(path, xyxy1, xyxy2, save_dir, conf = conf)


def draw_rotated_boxes(image_path, pred_boxes, gt_boxes, save_dir=None, conf = None):
    """
    绘制预测框和标注框在图像上。
    :param image_path: 图像路径
    :param pred_boxes: 预测框，形状 [N, 8]
    :param gt_boxes:   标注框，形状 [M, 8]
    """
    # 读取图像
    with Image.open(image_path) as im:
        img = np.array(im.convert("RGB"))

    # img = np.array(Image.open(image_path).convert("RGB"))

    # 创建画布
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)

    if conf is not None:
        # 画预测框（红色）
        for box, score in zip(pred_boxes, conf):
            if score<0.25:
                continue
            box = np.array(box).reshape(4, 2)
            polygon = patches.Polygon(box, closed=True, edgecolor='red', fill=False, linewidth=1, label='Prediction')
            ax.add_patch(polygon)
    else:
        # 画预测框（红色）
        for box in zip(pred_boxes):
            box = np.array(box).reshape(4, 2)
            polygon = patches.Polygon(box, closed=True, edgecolor='red', fill=False, linewidth=1, label='Prediction')
            ax.add_patch(polygon)

    # 画标注框（绿色）
    for box in gt_boxes:
        box = np.array(box).reshape(4, 2)
        polygon = patches.Polygon(box, closed=True, edgecolor='green', fill=False, linewidth=1, label='Ground Truth')
        ax.add_patch(polygon)

    # 显示图像
    ax.axis('off')
    # plt.title("Red: Prediction | Green: Ground Truth")
    plt.tight_layout()

    # 如果提供了保存路径，保存图像
    if save_dir is not None:
        # os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f"{image_name}_result.png")
        plt.savefig(save_path, bbox_inches='tight')
        # print(f"Saved result to: {save_path}")
    plt.close()  # 关闭当前活动的 figure

    return


    if conf is not None:
        # 创建画布
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(img)

        # 画预测框（红色）
        for box, score in zip(pred_boxes, conf):
            if score < 0.01:
                continue
            box = np.array(box).reshape(4, 2)
            polygon = patches.Polygon(box, closed=True, edgecolor='red', fill=False, linewidth=2, label='Prediction')
            ax.add_patch(polygon)

            # 获取左上角点作为显示位置（你也可以换成 box.mean(axis=0)）
            text_x, text_y = box[0][0], box[0][1]

            ax.text(text_x, text_y, f"{score:.4f}", color='red', fontsize=6,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='red', boxstyle='round,pad=0.2'))

        # 显示图像
        ax.axis('off')
        plt.title("Red: Prediction | Green: Ground Truth")
        plt.tight_layout()

        # 如果提供了保存路径，保存图像
        if save_dir is not None:
            # os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(save_dir, f"{image_name}_result2.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            # print(f"Saved result to: {save_path}")
        plt.close()  # 关闭当前活动的 figure





    # 创建画布
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)

    # 画标注框（绿色）
    for box in gt_boxes:
        box = np.array(box).reshape(4, 2)
        polygon = patches.Polygon(box, closed=True, edgecolor='green', fill=False, linewidth=2, label='Ground Truth')
        ax.add_patch(polygon)

    # 显示图像
    ax.axis('off')
    plt.title("Red: Prediction | Green: Ground Truth")
    plt.tight_layout()

    # 如果提供了保存路径，保存图像
    if save_dir is not None:
        # os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f"{image_name}_result3.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        # print(f"Saved result to: {save_path}")
    plt.close()  # 关闭当前活动的 figure





    # # 创建画布
    # fig, ax = plt.subplots(1, figsize=(12, 12))
    # ax.imshow(img)
    #
    # if conf is not None:
    #     # 画预测框（红色）
    #     for box, score in zip(pred_boxes, conf):
    #         if score<0.01:
    #             continue
    #         box = np.array(box).reshape(4, 2)
    #         polygon = patches.Polygon(box, closed=True, edgecolor='red', fill=False, linewidth=2, label='Prediction')
    #         ax.add_patch(polygon)
    # else:
    #     # 画预测框（红色）
    #     for box in zip(pred_boxes):
    #         box = np.array(box).reshape(4, 2)
    #         polygon = patches.Polygon(box, closed=True, edgecolor='red', fill=False, linewidth=2, label='Prediction')
    #         ax.add_patch(polygon)
    #
    # # 显示图像
    # ax.axis('off')
    # plt.title("Red: Prediction | Green: Ground Truth")
    # plt.tight_layout()
    #
    # # 如果提供了保存路径，保存图像
    # if save_dir is not None:
    #     # os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）
    #     image_name = os.path.splitext(os.path.basename(image_path))[0]
    #     save_path = os.path.join(save_dir, f"{image_name}_result4.png")
    #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    #     # print(f"Saved result to: {save_path}")
    # plt.close()  # 关闭当前活动的 figure
    #
    # plt.close('all')

    # plt.show()

from math import pi
def draw_rotated4rapid_box(image_path, gt_boxes, save_dir=None):
    boxes1 = gt_boxes
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    assert boxes1.shape[1] == 5

    boxes1[:, 4] = boxes1[:, 4] * pi / 180

    g = xywha2vertex(boxes1, is_degree=False, stack=False).numpy()
    g = g.reshape(-1,8).tolist()
    d = []
    draw_rotated_boxes(image_path, g, d, save_dir=save_dir)

def xywha2vertex(box, is_degree, stack=True):
    '''
    Args:
        box: tensor, shape(batch,5), 5=(x,y,w,h,a), xy is center,
             angle is radian

    Return:
        tensor, shape(batch,4,2): topleft, topright, br, bl
    '''
    assert is_degree == False and box.dim() == 2 and box.shape[1] >= 5
    batch = box.shape[0]
    device = box.device

    center = box[:,0:2]
    w = box[:,2]
    h = box[:,3]
    rad = box[:,4]

    # calculate two vector
    verti = torch.empty((batch,2), dtype=torch.float32, device=device)
    verti[:,0] = (h/2) * torch.sin(rad)
    verti[:,1] = - (h/2) * torch.cos(rad)

    hori = torch.empty(batch,2, dtype=torch.float32, device=device)
    hori[:,0] = (w/2) * torch.cos(rad)
    hori[:,1] = (w/2) * torch.sin(rad)


    tl = center + verti - hori
    tr = center + verti + hori
    br = center - verti + hori
    bl = center - verti - hori

    if not stack:
        return torch.cat([tl,tr,br,bl], dim=1)
    return torch.stack((tl,tr,br,bl), dim=1)


def loaf_xyxy2xyxyxyxy(xyxy, offset=512):
    x_c, y_c = (xyxy[:,0] + xyxy[:,2])/2, (xyxy[:,1] + xyxy[:,3])/2
    w,h = xyxy[:,2]-xyxy[:,0], xyxy[:,3]-xyxy[:,1]

    # x_c, y_c, w, h = cxcy.unbind(-1)

    x_c = x_c-offset
    y_c = y_c-offset
    R = torch.pow(torch.pow(x_c,2)+torch.pow(y_c,2), 0.5)
    R[R==0]=1e-4

    cosine = x_c/R
    sine = y_c/R

    # cosine = y_c/R
    # sine = x_c/R

    left_x = x_c+w/2*sine
    left_y = y_c-w/2*cosine
    right_x = x_c-w/2*sine
    right_y = y_c+w/2*cosine

    gap_x = h/2*cosine
    gap_y = h/2*sine

    xyxy = [right_x+gap_x, right_y+gap_y,
            left_x+gap_x, left_y+gap_y,
            left_x-gap_x, left_y-gap_y,
            right_x-gap_x, right_y-gap_y]

    return torch.stack(xyxy, dim=-1)+offset # N, 8
