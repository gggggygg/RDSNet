"""
作者：linwenwei
日期：2024年04月19日
"""
import math

import numpy as np
import torch
import torch.nn as nn


def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=0)

def get_pixel_neighbors_angle_delta(grid, angle, distance, arc_step):
    """
    grid: (2, H, W)
    angle: (H, W)
    distance: (H, W)
    """
    H, W = grid.shape[-2:]
    center = (H/2-0.5, W/2-0.5)
    distance_up = distance + 1
    distance_down = distance - 1
    distance_w = torch.stack([distance_up, distance, distance_down], dim=0)

    delta_theta = arc_step / distance_w
    angle = angle - 90
    angle = torch.deg2rad(angle)[None]

    # 计算左侧点的坐标
    x1 = center[0] + distance_w * torch.cos(angle - delta_theta)
    y1 = center[1] + distance_w * torch.sin(angle - delta_theta)
    grid_left = torch.stack([x1, y1], dim=1)        # 合并w维度上的坐标

    # 计算中间点的坐标
    x1 = center[0] + distance_w * torch.cos(angle)
    y1 = center[1] + distance_w * torch.sin(angle)
    grid_mid = torch.stack([x1, y1], dim=1)         # 合并w维度上的坐标

    # 计算右侧点的坐标
    x1 = center[0] + distance_w * torch.cos(angle + delta_theta)
    y1 = center[1] + distance_w * torch.sin(angle + delta_theta)
    grid_right = torch.stack([x1, y1], dim=1)       # 合并w维度上的坐标

    # 合并h维度上的坐标
    grid_sampled = torch.stack([grid_left, grid_mid, grid_right], dim=1)

    # 减去中心点计算每个kernel的中心点
    offset = grid_sampled - grid[None, None]

    kernel_size = 3
    y, x = torch.meshgrid(
        torch.arange(kernel_size).float()-kernel_size//2,
        torch.arange(kernel_size).float()-kernel_size//2)
    kernel_offset = torch.stack([x, y], dim=-1)
    offset = offset - kernel_offset[..., None, None]

    offset = torch.flip(offset, dims=[2])

    return offset

def calculate_adjacent_points(center, point, arc_length):
    h, k = center
    x, y = point

    # 计算半径
    r = torch.sqrt((x - h)**2 + (y - k)**2)

    # 计算角度
    theta = torch.atan2(y - k, x - h)

    # 计算弧长对应的角度
    delta_theta = arc_length / r

    # 计算左侧点的坐标
    x1 = h + r * torch.cos(theta - delta_theta)
    y1 = k + r * torch.sin(theta - delta_theta)

    # 计算右侧点的坐标
    x2 = h + r * torch.cos(theta + delta_theta)
    y2 = k + r * torch.sin(theta + delta_theta)

    return (x1.item(), y1.item()), (x2.item(), y2.item())


def get_pixel_angle(grid):
    center = torch.tensor([grid.shape[2] / 2, grid.shape[1] / 2]) - 0.5
    distance_x = grid[0] - center[0]
    distance_y = grid[1] - center[1]
    angle = torch.rad2deg(torch.atan2(distance_y, distance_x)) + 90  #表示从原点 (0, 0) 到点 (x, y) 的向量相对于 x 轴的夹角（顺时针为正方向），单位是弧度，结果范围是：(-pi,pi]
    angle[angle < 0] = angle[angle < 0] + 360  # 将负角度转换为正角度  [0, 360)  中心向上为0度，顺时针正方向

    distances = torch.sqrt(distance_x**2 + distance_y**2)
    distances[distances == 0] = 1  # 防止除零错误
    return angle, distances


def get_omni_offset(input_size, kernel_size=3, arc=False):
    H, W = input_size
    grid_ori = coords_grid(H, W)
    angles, distances = get_pixel_angle(grid_ori)
    if arc:
        offsets = get_pixel_neighbors_angle_delta(grid_ori, angles, distances, 1)
    else:
        offsets = omni_offset(angles, distances, kernel_size=kernel_size)   ##得到每个像素处、每个卷积采样点应偏移多少。 3*3*2*h*w
    offsets = offsets.view(-1, H, W)  #偶数下标存 offset_h、奇数存 offset_w
    offsets = nn.Parameter(torch.Tensor(offsets), requires_grad=False)
    return offsets



def omni_offset(angles, distances=None, kernel_size=3, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(kernel_size).to(**kwargs).float()-kernel_size//2,
        torch.arange(kernel_size).to(**kwargs).float()-kernel_size//2)
    kernel_offset = torch.stack([x, y], dim=-1)

    kernel_offset = kernel_offset[..., None, None]

    theta = torch.deg2rad(angles)[None, None]

    rotated_offset_x = kernel_offset[:, :, 0] * torch.cos(theta) - kernel_offset[:, :, 1] * torch.sin(theta)
    rotated_offset_y = kernel_offset[:, :, 0] * torch.sin(theta) + kernel_offset[:, :, 1] * torch.cos(theta)
    rotated_offset = torch.stack([rotated_offset_x, rotated_offset_y], dim=2)

    offset = rotated_offset - kernel_offset   #得到每个像素处、每个卷积采样点应偏移多少。
    offset = torch.flip(offset, dims=[2])   ## flip x/y 顺序 => [k, k, 2, H, W]

    return offset


# 帮我写一个函数，输入是一个3*3*2的grid，给定一个角度，将这个grid旋转一定角度，以时钟0时刻为基准，顺时针方向为正方向
def rotate_point(x, y, cx, cy, angle):
    # 计算相对于中心点的坐标偏移
    dx = x - cx
    dy = y - cy

    # 将角度转换为弧度
    theta = math.radians(angle)

    # 计算旋转后的坐标偏移
    rotated_dx = dx * math.cos(theta) - dy * math.sin(theta)
    rotated_dy = dx * math.sin(theta) + dy * math.cos(theta)

    # 返回旋转后的坐标相对于中心点的偏移
    return rotated_dx, rotated_dy



