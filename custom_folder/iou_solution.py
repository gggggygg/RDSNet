#####under code from LOAF
'''
torch implementation of 2d oriented box intersection

author: lanxiao li
2020.8
'''
import torch


EPSILON = 1e-8

import torch
from torch import nn
from torch.autograd import Function
# from ultralytics.utils.ops import xywhr2xyxyxyxy
from custom_folder.min_enclosing_box import smallest_bounding_box
import numpy as np


def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)



def box_area(boxes):
    x0, y0, x1, y1, x2, y2, x3, y3 = boxes.unbind(-1)  # [N,]
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 * \
        ((x3 - x0) ** 2 + (y3 - y0) ** 2) ** 0.5  # [N,]


def cal_iou(box1: torch.Tensor, box2: torch.Tensor):
    """calculate iou

    Args:
        box1 (torch.Tensor): (N, 8)
        box2 (torch.Tensor): (M, 8)

    Returns:
        iou (torch.Tensor): (N, M)
    """
    corners1 = box1.reshape(box1.shape[0], 1, 4, 2).repeat(1, box2.shape[0], 1, 1).cuda()
    corners2 = box2.reshape(1, box2.shape[0], 4, 2).repeat(box1.shape[0], 1, 1, 1).cuda()
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)  #
    area1 = box_area(corners1.flatten(2))
    area2 = box_area(corners2.flatten(2))
    u = area1 + area2 - inter_area
    iou = inter_area / u
    return iou, corners1, corners2, u


def cal_iou_per_pair(box1: torch.Tensor, box2: torch.Tensor):
    """
    Calculate IoU for each corresponding pair of OBBs (i.e., one-to-one).

    Args:
        box1 (torch.Tensor): (N, 8) predicted boxes
        box2 (torch.Tensor): (N, 8) ground truth boxes

    Returns:
        iou (torch.Tensor): (N,)
    """
    assert box1.shape == box2.shape and box1.shape[1] == 8, "Input should be (N, 8) for both boxes"

    corners1 = box1.reshape(box1.shape[0], 1, 4, 2)
    corners2 = box2.reshape(box2.shape[0], 1, 4, 2)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)  #
    area1 = box_area(corners1.flatten(2))
    area2 = box_area(corners2.flatten(2))
    u = area1 + area2 - inter_area
    iou = inter_area / u
    return iou, corners1, corners2, u


    # # [N, 4, 2]
    # corners1 = box1.reshape(-1, 4, 2)
    # corners2 = box2.reshape(-1, 4, 2)
    #
    # # Expand dims for batch-wise intersection
    # inter_area, _ = oriented_box_intersection_2d(
    #     corners1[:, None, :, :], corners2[:, None, :, :]
    # )  # returns shape [N, 1], we extract [:,0]
    # inter_area = inter_area[:, 0]
    #
    # area1 = box_area(corners1)
    # area2 = box_area(corners2)
    # union = area1 + area2 - inter_area
    # iou = inter_area / (union + 1e-7)
    # return iou

def cal_giou(box1:torch.Tensor, box2:torch.Tensor, enclosing_type:str="smallest"):
    iou, corners1, corners2, u = cal_iou_per_pair(box1, box2)
    w, h = enclosing_box(corners1, corners2, enclosing_type)
    area_c =  w*h
    giou =  iou - ( area_c - u )/area_c
    return giou, iou

def enclosing_box(corners1:torch.Tensor, corners2:torch.Tensor, enclosing_type:str="smallest"):
    if enclosing_type == "aligned":
        return enclosing_box_aligned(corners1, corners2)
    elif enclosing_type == "pca":
        return enclosing_box_pca(corners1, corners2)
    elif enclosing_type == "smallest":
        return smallest_bounding_box(torch.cat([corners1, corners2], dim=-2))
    else:
        ValueError("Unknow type enclosing. Supported: aligned, pca, smallest")


def coco_iou(box1: torch.Tensor, box2: torch.Tensor):
    """calculate iou

    Args:
        box1 (torch.Tensor): (b, n, 5).
        box2 (torch.Tensor): (b, n, 5).

    Returns:
        iou (torch.Tensor): (b, n)
    """
    box1 = xywhr2xyxyxyxy(box1)  # (b, n, 4, 2).
    box2 = xywhr2xyxyxyxy(box2)

    giou = generalized_rotated_box_iou(box1, box2)
    return giou


def generalized_rotated_box_iou(boxes1, boxes2):
    giou, iou = cal_giou(boxes1, boxes2)
    #     if iou.size(0)==iou.size(1):
    #         print(torch.diag(iou))

    return giou


from math import pi
def cal_iou_4_rapid(boxes1, boxes2, xywha, is_degree=True, **kwargs):
    r'''
    use mask method to calculate IOU between boxes1 and boxes2

    Arguments:
        boxes1: tensor or numpy, shape(N,5), 5=(x, y, w, h, angle 0~90)
        boxes2: tensor or numpy, shape(M,5), 5=(x, y, w, h, angle 0~90)
        xywha: True if xywha, False if xyxya
        is_degree: True if degree, False if radian

    Return:
        iou_matrix: tensor, shape(N,M), float32,
                    ious of all possible pairs between boxes1 and boxes2
    '''
    assert xywha == True and is_degree == True

    # start_time = time.time()

    if not (torch.is_tensor(boxes1) and torch.is_tensor(boxes2)):
        print('Warning: bounding boxes are np.array. converting to torch.tensor')
        # convert to tensor, (batch, (x,y,w,h,a))
        boxes1 = torch.from_numpy(boxes1).float()
        boxes2 = torch.from_numpy(boxes2).float()
    assert boxes1.device == boxes2.device
    device = boxes1.device

    boxes1, boxes2 = boxes1.cpu().clone().detach(), boxes2.cpu().clone().detach()
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)
    assert boxes1.shape[1] == boxes2.shape[1] == 5

    size = kwargs.get('img_size', 2048)
    h, w = size if isinstance(size, tuple) else (size, size)

    if 'normalized' in kwargs and kwargs['normalized'] == True:
        # the [x,y,w,h] are between 0~1
        # assert (boxes1[:,:4] <= 1).all() and (boxes2[:,:4] <= 1).all()
        boxes1[:, 0] *= w
        boxes1[:, 1] *= h
        boxes1[:, 2] *= w
        boxes1[:, 3] *= h
        boxes2[:, 0] *= w
        boxes2[:, 1] *= h
        boxes2[:, 2] *= w
        boxes2[:, 3] *= h
    if is_degree:
        # convert to radian
        boxes1[:, 4] = boxes1[:, 4] * pi / 180
        boxes2[:, 4] = boxes2[:, 4] * pi / 180

    d = xywha2vertex(boxes1, is_degree=False, stack=False).reshape(-1, 8).to(torch.float32)
    g = xywha2vertex(boxes2, is_degree=False, stack=False).reshape(-1, 8).to(torch.float32)

    if d.shape[0] == 0 or g.shape[0] == 0:
        return np.zeros((d.shape[0], g.shape[0]))
    ious, corners1, corners2, u = cal_iou(d, g)
    return ious

    # b1 = xywha2vertex(boxes1, is_degree=False, stack=False).tolist()
    # b2 = xywha2vertex(boxes2, is_degree=False, stack=False).tolist()
    # debug = 1
    #
    # b1 = maskUtils.frPyObjects(b1, h, w)
    # b2 = maskUtils.frPyObjects(b2, h, w)
    # ious = maskUtils.iou(b1, b2, [0 for _ in b2])
    #
    # return torch.from_numpy(ious).to(device=device)


def cal_iou_4_rapid_loaf(boxes1, boxes2, xywha, is_degree=True, **kwargs):
    r'''
    use mask method to calculate IOU between boxes1 and boxes2

    Arguments:
        boxes1: tensor or numpy, shape(N,5), 5=(x, y, w, h, angle 0~90)
        boxes2: tensor or numpy, shape(M,5), 5=(x, y, w, h, angle 0~90)
        xywha: True if xywha, False if xyxya
        is_degree: True if degree, False if radian

    Return:
        iou_matrix: tensor, shape(N,M), float32,
                    ious of all possible pairs between boxes1 and boxes2
    '''
    assert xywha == True and is_degree == True

    # start_time = time.time()

    if not (torch.is_tensor(boxes1) and torch.is_tensor(boxes2)):
        print('Warning: bounding boxes are np.array. converting to torch.tensor')
        # convert to tensor, (batch, (x,y,w,h,a))
        boxes1 = torch.from_numpy(boxes1).float()
        boxes2 = torch.from_numpy(boxes2).float()
    assert boxes1.device == boxes2.device
    device = boxes1.device

    boxes1, boxes2 = boxes1.cpu().clone().detach(), boxes2.cpu().clone().detach()
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)
    assert boxes1.shape[1] == boxes2.shape[1] == 5

    size = kwargs.get('img_size', 2048)
    h, w = size if isinstance(size, tuple) else (size, size)

    if 'normalized' in kwargs and kwargs['normalized'] == True:
        # the [x,y,w,h] are between 0~1
        # assert (boxes1[:,:4] <= 1).all() and (boxes2[:,:4] <= 1).all()
        boxes1[:, 0] *= w
        boxes1[:, 1] *= h
        boxes1[:, 2] *= w
        boxes1[:, 3] *= h
        boxes2[:, 0] *= w
        boxes2[:, 1] *= h
        boxes2[:, 2] *= w
        boxes2[:, 3] *= h
    if is_degree:
        # convert to radian
        boxes1[:, 4] = boxes1[:, 4] * pi / 180
        boxes2[:, 4] = boxes2[:, 4] * pi / 180

    # d = xywha2vertex4loaf(boxes1, offset=512).reshape(-1, 8).to(torch.float32)
    d = xywha2vertex(boxes1, is_degree=False, stack=False).reshape(-1, 8).to(torch.float32)
    g = xywha2vertex(boxes2, is_degree=False, stack=False).reshape(-1, 8).to(torch.float32)

    im_path = kwargs.get('im_path', None)
    save_dir = 'results/plot_ious'
    if im_path:
        draw_rotated_boxes_from_tensor_boxes(d, g, path = im_path, save_dir = save_dir)

    if d.shape[0] == 0 or g.shape[0] == 0:
        return np.zeros((d.shape[0], g.shape[0]))
    ious, corners1, corners2, u = cal_iou(d, g)
    return ious

from custom_folder.visual import draw_rotated_boxes_from_tensor_boxes
def cal_iou_4_rapid_loaf4nms(boxes1, boxes2, xywha, is_degree=True, **kwargs):
    r'''
    use mask method to calculate IOU between boxes1 and boxes2

    Arguments:
        boxes1: tensor or numpy, shape(N,5), 5=(x, y, w, h, angle 0~90)
        boxes2: tensor or numpy, shape(M,5), 5=(x, y, w, h, angle 0~90)
        xywha: True if xywha, False if xyxya
        is_degree: True if degree, False if radian

    Return:
        iou_matrix: tensor, shape(N,M), float32,
                    ious of all possible pairs between boxes1 and boxes2
    '''
    assert xywha == True and is_degree == True

    # start_time = time.time()

    if not (torch.is_tensor(boxes1) and torch.is_tensor(boxes2)):
        print('Warning: bounding boxes are np.array. converting to torch.tensor')
        # convert to tensor, (batch, (x,y,w,h,a))
        boxes1 = torch.from_numpy(boxes1).float()
        boxes2 = torch.from_numpy(boxes2).float()
    assert boxes1.device == boxes2.device
    device = boxes1.device

    boxes1, boxes2 = boxes1.cpu().clone().detach(), boxes2.cpu().clone().detach()
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)
    assert boxes1.shape[1] == boxes2.shape[1] == 5

    size = kwargs.get('img_size', 2048)
    h, w = size if isinstance(size, tuple) else (size, size)

    if 'normalized' in kwargs and kwargs['normalized'] == True:
        # the [x,y,w,h] are between 0~1
        # assert (boxes1[:,:4] <= 1).all() and (boxes2[:,:4] <= 1).all()
        boxes1[:, 0] *= w
        boxes1[:, 1] *= h
        boxes1[:, 2] *= w
        boxes1[:, 3] *= h
        boxes2[:, 0] *= w
        boxes2[:, 1] *= h
        boxes2[:, 2] *= w
        boxes2[:, 3] *= h
    if is_degree:
        # convert to radian
        boxes1[:, 4] = boxes1[:, 4] * pi / 180
        boxes2[:, 4] = boxes2[:, 4] * pi / 180

    d = xywha2vertex4loaf(boxes1, offset=size//2).reshape(-1, 8).to(torch.float32)
    g = xywha2vertex4loaf(boxes2, offset=size//2).reshape(-1, 8).to(torch.float32)

    # im_path = kwargs.get('im_path', None)
    # save_dir = 'results/plot_nms'
    # if im_path:
    #     draw_rotated_boxes_from_tensor_boxes(d, g, path = im_path, save_dir = save_dir)

    if d.shape[0] == 0 or g.shape[0] == 0:
        return np.zeros((d.shape[0], g.shape[0]))
    ious, corners1, corners2, u = cal_iou(d, g)
    return ious

def xywha2vertex4loaf(cxcy, offset=0.5):
    x_c, y_c, w, h,r = cxcy.unbind(-1)

    # 假设 w 和 h 是 torch.Tensor
    mask = w > h
    w[mask], h[mask] = h[mask].clone(), w[mask].clone()

    x_c = x_c-offset
    y_c = y_c-offset
    R = torch.pow(torch.pow(x_c,2)+torch.pow(y_c,2), 0.5)
    R[R==0]=1e-4

    # cosine = x_c/R
    # sine = y_c/R

    cosine = y_c/R
    sine = x_c/R

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

import sort_vertices

class SortVertices(Function):
    @staticmethod
    def forward(ctx, vertices, mask, num_valid):
        idx = sort_vertices.sort_vertices_forward(vertices.to(torch.float32), mask, num_valid)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, gradout):
        return ()


sort_v = SortVertices.apply

def box_intersection_th(corners1: torch.Tensor, corners2: torch.Tensor):
    """find intersection points of rectangles
    Convention: if two edges are collinear, there is no intersection point

    Args:
        corners1 (torch.Tensor): B, N, 4, 2
        corners2 (torch.Tensor): B, N, 4, 2

    Returns:
        intersectons (torch.Tensor): B, N, 4, 4, 2
        mask (torch.Tensor) : B, N, 4, 4; bool
    """
    # build edges from corners
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3)  # B, N, 4, 4: Batch, Box, edge, point
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    # duplicate data to pair each edges from the boxes
    # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(3).repeat([1, 1, 1, 4, 1])
    line2_ext = line2.unsqueeze(2).repeat([1, 1, 4, 1, 1])
    x1 = line1_ext[..., 0]
    y1 = line1_ext[..., 1]
    x2 = line1_ext[..., 2]
    y2 = line1_ext[..., 3]
    x3 = line2_ext[..., 0]
    y3 = line2_ext[..., 1]
    x4 = line2_ext[..., 2]
    y4 = line2_ext[..., 3]
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    num = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    den_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = den_t / num
    t[num == .0] = -1.
    mask_t = (t > 0) * (t < 1)  # intersection on line segment 1
    den_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    u = -den_u / num
    u[num == .0] = -1.
    mask_u = (u > 0) * (u < 1)  # intersection on line segment 2
    mask = mask_t * mask_u
    t = den_t / (num + EPSILON)  # overwrite with EPSILON. otherwise numerically unstable
    intersections = torch.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask


def box1_in_box2(corners1: torch.Tensor, corners2: torch.Tensor):
    """check if corners of box1 lie in box2
    Convention: if a corner is exactly on the edge of the other box, it's also a valid point

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        c1_in_2: (B, N, 4) Bool
    """
    a = corners2[:, :, 0:1, :]  # (B, N, 1, 2)
    b = corners2[:, :, 1:2, :]  # (B, N, 1, 2)
    d = corners2[:, :, 3:4, :]  # (B, N, 1, 2)
    ab = b - a  # (B, N, 1, 2)
    am = corners1 - a  # (B, N, 4, 2)
    ad = d - a  # (B, N, 1, 2)
    p_ab = torch.sum(ab * am, dim=-1)  # (B, N, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)  # (B, N, 1)
    p_ad = torch.sum(ad * am, dim=-1)  # (B, N, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)  # (B, N, 1)
    # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
    # also stable with different scale of bboxes
    cond1 = (p_ab / norm_ab > - 1e-6) * (p_ab / norm_ab < 1 + 1e-6)  # (B, N, 4)
    cond2 = (p_ad / norm_ad > - 1e-6) * (p_ad / norm_ad < 1 + 1e-6)  # (B, N, 4)
    return cond1 * cond2


def box_in_box_th(corners1: torch.Tensor, corners2: torch.Tensor):
    """check if corners of two boxes lie in each other

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        c1_in_2: (B, N, 4) Bool. i-th corner of box1 in box2
        c2_in_1: (B, N, 4) Bool. i-th corner of box2 in box1
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1


def build_vertices(corners1: torch.Tensor, corners2: torch.Tensor,
                   c1_in_2: torch.Tensor, c2_in_1: torch.Tensor,
                   inters: torch.Tensor, mask_inter: torch.Tensor):
    """find vertices of intersection area

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
        c1_in_2 (torch.Tensor): Bool, (B, N, 4)
        c2_in_1 (torch.Tensor): Bool, (B, N, 4)
        inters (torch.Tensor): (B, N, 4, 4, 2)
        mask_inter (torch.Tensor): (B, N, 4, 4)

    Returns:
        vertices (torch.Tensor): (B, N, 24, 2) vertices of intersection area. only some elements are valid
        mask (torch.Tensor): (B, N, 24) indicates valid elements in vertices
    """
    # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0). 
    # can be used as trick
    B = corners1.size()[0]
    N = corners1.size()[1]
    vertices = torch.cat([corners1, corners2, inters.view([B, N, -1, 2])], dim=2)  # (B, N, 4+4+16, 2)
    mask = torch.cat([c1_in_2, c2_in_1, mask_inter.view([B, N, -1])], dim=2)  # Bool (B, N, 4+4+16)
    return vertices, mask


def sort_indices(vertices: torch.Tensor, mask: torch.Tensor):
    """[summary]

    Args:
        vertices (torch.Tensor): float (B, N, 24, 2)
        mask (torch.Tensor): bool (B, N, 24)

    Returns:
        sorted_index: bool (B, N, 9)

    Note:
        why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X) 
        and X indicates the index of arbitary elements in the last 16 (intersections not corners) with 
        value 0 and mask False. (cause they have zero value and zero gradient)
    """
    num_valid = torch.sum(mask.int(), dim=2).int()  # (B, N)
    mean = torch.sum(vertices * mask.float().unsqueeze(-1), dim=2, keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
    vertices_normalized = vertices - mean  # normalization makes sorting easier
    return sort_v(vertices_normalized, mask, num_valid).long()


def calculate_area(idx_sorted: torch.Tensor, vertices: torch.Tensor):
    """calculate area of intersection

    Args:
        idx_sorted (torch.Tensor): (B, N, 9)
        vertices (torch.Tensor): (B, N, 24, 2)

    return:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding 
    """
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1, 1, 1, 2])
    selected = torch.gather(vertices, 2, idx_ext)
    total = selected[:, :, 0:-1, 0] * selected[:, :, 1:, 1] - selected[:, :, 0:-1, 1] * selected[:, :, 1:, 0]
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected


def oriented_box_intersection_2d(corners1: torch.Tensor, corners2: torch.Tensor):
    """calculate intersection area of 2d rectangles 

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding 
    """
    inters, mask_inter = box_intersection_th(corners1, corners2)
    c12, c21 = box_in_box_th(corners1, corners2)
    vertices, mask = build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
    sorted_indices = sort_indices(vertices, mask)
    return calculate_area(sorted_indices, vertices)


def nms(detections, is_degree=True, nms_thres=0.45, img_size=2048, **kwargs):
    '''
    Single-class non-maximum suppression for bounding boxes with angle.

    Args:
        detections: rows of (x,y,w,h,angle,conf,...)
        is_degree: True -> input angle is degree, False -> radian
        nms_thres: suppresion IoU threshold
        img_size: int, preferably the image size
    '''
    assert (detections.dim() == 2) and (detections.shape[1] >= 6)
    device = detections.device
    if detections.shape[0] == 0:
        return detections
    # sort by confidence
    idx = torch.argsort(detections[:, 5], descending=True)
    detections = detections[idx, :]

    boxes = detections[:, 0:5]  # only [x,y,w,h,a]
    valid = torch.zeros(boxes.shape[0], dtype=torch.bool, device=device)
    # the first one is always valid
    valid[0] = True
    # only one candidate at the beginning. Its votes number is 1 (it self)
    votes = [1]
    for i in range(1, boxes.shape[0]):
        # compute IoU with valid boxes
        # ious = iou_mask(boxes[i], boxes[valid,:], True, 32, is_degree=is_degree)
        # ious = iou_rle(boxes[i], boxes[valid,:], xywha=True, is_degree=is_degree,
        #               img_size=img_size)

        # ious = cal_iou_4_rapid(boxes[i], boxes[valid, :], xywha=True, is_degree=is_degree,
        #                        img_size=img_size, im_path = kwargs.get('im_path', None))

        ious = cal_iou_4_rapid_loaf4nms(boxes[i], boxes[valid, :], xywha=True, is_degree=is_degree,
                               img_size=img_size, im_path = kwargs.get('im_path', None))
        # the i'th BB is invalid if it is similar to any valid BB
        if (ious >= nms_thres).any():
            continue
        # else, this box is valid
        valid[i] = True
        # the votes number of the new candidate BB is 1 (it self)
        votes.append(1)

    selected = detections[valid, :]
    return selected
