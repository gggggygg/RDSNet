import json
import os
import cv2

input_filename = "../annotations/resolution_1k/instances_train.json"
image_id = 100

with open(input_filename, "r") as f:
    loaf_json = json.load(f)

o_images = loaf_json["images"]
o_categories = loaf_json["categories"]
o_annotations = loaf_json["annotations"]
print('images', len(o_images),' anns',len(o_annotations))
file_name = ''
for i in o_images:
    # print(i)
    if i['id'] == image_id:
        file_name = i['file_name']
        print(i)
        # break

objs = []
for i in o_annotations:
    if i['image_id'] == image_id:
        objs.append(i)



import math
import numpy as np


def ritbox2poly(ritbox, im_center=None):
    cx, cy, h, wt, wb, alpha = ritbox
    p1 = [cx - wt / 2, cy - h / 2]
    p2 = [cx + wt / 2, cy - h / 2]
    p3 = [cx + wb / 2, cy + h / 2]
    p4 = [cx - wb / 2, cy + h / 2]

    if im_center is None:
        raise NotImplementedError
        # TODO not verified
        alpha_rad = (alpha + 180) % 360 * math.pi / 180
        cos_theta, sin_theta = math.cos(alpha_rad), math.sin(alpha_rad)
    else:
        im_cx, im_cy = im_center
        c = math.sqrt((cx - im_cx) ** 2 + (cy - im_cy) ** 2)
        a = im_cx - cx
        b = im_cy - cy

        cos_theta = b / c
        sin_theta = a / c

    rp1 = [(p1[0] - cx) * cos_theta + (p1[1] - cy) * sin_theta + cx,
           (p1[1] - cy) * cos_theta - (p1[0] - cx) * sin_theta + cy]
    rp2 = [(p2[0] - cx) * cos_theta + (p2[1] - cy) * sin_theta + cx,
           (p2[1] - cy) * cos_theta - (p2[0] - cx) * sin_theta + cy]
    rp3 = [(p3[0] - cx) * cos_theta + (p3[1] - cy) * sin_theta + cx,
           (p3[1] - cy) * cos_theta - (p3[0] - cx) * sin_theta + cy]
    rp4 = [(p4[0] - cx) * cos_theta + (p4[1] - cy) * sin_theta + cx,
           (p4[1] - cy) * cos_theta - (p4[0] - cx) * sin_theta + cy]
    poly = rp1 + rp2 + rp3 + rp4

    return poly


def vis_ritbox(im, ritbox, bbox_color):
    """Visualizes a ritbox."""
    h, w = im.shape[:2]
    poly = ritbox2poly(ritbox, (w / 2, h / 2))
    quad = [[poly[0], poly[1]], [poly[2], poly[3]], [poly[4], poly[5]], [poly[6], poly[7]]]
    quad = np.asarray([quad], dtype=np.int32)
    cv2.polylines(im, quad, 1, bbox_color, thickness=2)
    return im


def vis_rotatedbox(im, rotated_bbox, bbox_color):
    """Visualizes a rotated bounding box."""
    (xc, yc, w, h, a) = rotated_bbox
    h_bbox = [(xc - w / 2, yc - h / 2),
              (xc + w / 2, yc - h / 2),
              (xc + w / 2, yc + h / 2),
              (xc - w / 2, yc + h / 2)]
    a = a * math.pi / 180
    cos = math.cos(a)
    sin = math.sin(a)
    o_bbox = []
    for i in range(len(h_bbox)):
        x_i = int(sin * (h_bbox[i][1] - yc) + cos * (h_bbox[i][0] - xc) + xc)
        y_i = int(cos * (h_bbox[i][1] - yc) - sin * (h_bbox[i][0] - xc) + yc)
        o_bbox.append((x_i, y_i))
    for j in range(len(o_bbox)):
        cv2.line(im, o_bbox[j], o_bbox[(j + 1) % len(o_bbox)], bbox_color, thickness=2)
    return im


def rotatedbox2bbox(rotated_bbox):
    (xc, yc, w, h, a) = rotated_bbox
    c = math.fabs(math.cos(a * math.pi / 180.0))
    s = math.fabs(math.sin(a * math.pi / 180.0))
    # This basically computes the horizontal bounding rectangle of the rotated box
    new_w = c * w + s * h
    new_h = c * h + s * w
    # convert center to top-left corner
    x0 = xc - new_w / 2.0
    y0 = yc - new_h / 2.0
    return [x0, y0, new_w, new_h]


def bbox2rotatedbox(bbox, im_center=(2952, 2952)):
    x0, y0, w, h = bbox
    xc, yc = x0 + w / 2.0, y0 + h / 2.0
    xc_euc = xc - im_center[0] / 2.0
    yc_euc = im_center[1] / 2.0 - yc
    r = math.sqrt(xc_euc ** 2 + yc_euc ** 2)
    rit_alpha = math.acos(xc_euc / r) if r != 0 else 0
    rit_alpha = 180 * rit_alpha / math.pi if yc_euc >= 0 else 360 - 180 * rit_alpha / math.pi
    rot_alpha = rit_alpha - 90 if rit_alpha < 180 else rit_alpha - 270

    c = math.fabs(math.cos(rot_alpha * math.pi / 180.0))
    s = math.fabs(math.sin(rot_alpha * math.pi / 180.0))
    new_w = (s * h - c * w) / (s ** 2 - c ** 2) if s != c else w * math.sqrt(2) / 3
    new_h = (s * w - c * h) / (s ** 2 - c ** 2) if s != c else 2 * w * math.sqrt(2) / 3
    return [xc, yc, new_w, new_h, rot_alpha]


def rotatedbox2tightbbox(rotated_bbox):
    (xc, yc, w, h, a) = rotated_bbox
    h_bbox = [(xc - w / 2, yc - h / 2),
              (xc + w / 2, yc - h / 2),
              (xc + w / 2, yc + h / 2),
              (xc - w / 2, yc + h / 2)]
    x0, y0 = h_bbox[0][0], h_bbox[0][1]
    new_w, new_h = math.fabs(h_bbox[2][0] - x0), math.fabs(h_bbox[2][1] - y0)
    return [x0, y0, new_w, new_h]


def vis_bbox(im, bbox, bbox_color):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(im, (x0, y0), (x1, y1), bbox_color, thickness=2)
    return im


def vis_location(im, person_location, circle_color, rs=(1024, 1024)):
    r, a, w, h, R = person_location
    euclid_x = r * math.cos(a * math.pi / 180.0)
    euclid_y = r * math.sin(a * math.pi / 180.0)
    cv2_x = euclid_x + rs[1] / 2.0
    cv2_y = -(euclid_y - rs[0] / 2.0)
    im = cv2.circle(im, (int(cv2_x), int(cv2_y)), rs[0] // 300, circle_color, thickness=-1)
    return im


img = cv2.imread("../images/resolution_1k/train/" + file_name)

for i in objs:
    img = vis_rotatedbox(img, i['rotated_box'], (255, 0, 0))
cv2.imwrite("vis.jpg", img)



