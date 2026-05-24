import numpy as np
import cv2
import matplotlib.pyplot as plt


def create_line_grid(size=(800, 800), grid_spacing=50):
    """生成水平+垂直线条的网格"""
    grid = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    color = (0, 0, 0)  # 黑色线条

    # 水平线条
    for y in range(grid_spacing, size[1], grid_spacing):
        cv2.line(grid, (0, y), (size[0], y), color, 2)

    # 垂直线条
    for x in range(grid_spacing, size[0], grid_spacing):
        cv2.line(grid, (x, 0), (x, size[1]), color, 2)

    return grid


def apply_barrel_distortion_fixed(img, k1=0.00005, center=None):
    """你的原始畸变函数（保持不变）"""
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    x_min, x_max, y_min, y_max = w, 0, h, 0

    for y in range(h):
        for x in range(w):
            dx = x - center[0]
            dy = y - center[1]
            r2 = dx * dx + dy * dy
            radial_factor = 1 + k1 * r2

            x_dist = center[0] + dx * radial_factor
            y_dist = center[1] + dy * radial_factor

            x_min = min(x_min, x_dist)
            x_max = max(x_max, x_dist)
            y_min = min(y_min, y_dist)
            y_max = max(y_max, y_dist)

    scale_x = w / (x_max - x_min + 1e-8)
    scale_y = h / (y_max - y_min + 1e-8)

    for y in range(h):
        for x in range(w):
            dx = x - center[0]
            dy = y - center[1]
            r2 = dx * dx + dy * dy
            radial_factor = 1 + k1 * r2

            x_dist = center[0] + dx * radial_factor
            y_dist = center[1] + dy * radial_factor

            x_remap = (x_dist - x_min) * scale_x
            y_remap = (y_dist - y_min) * scale_y

            map_x[y, x] = x_remap
            map_y[y, x] = y_remap

    distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)  # 边界填白色
    return distorted


# 生成并畸变线条网格
line_grid = create_line_grid(size=(800, 800), grid_spacing=50)
distorted_grid = apply_barrel_distortion_fixed(line_grid, k1=0.000002)

# 可视化
plt.figure(figsize=(12, 6))
plt.subplot(121)
# plt.title("原始线条网格")
plt.imshow(line_grid)
plt.axis('off')

plt.subplot(122)
# plt.title("桶形畸变效果（线条版）")
plt.imshow(distorted_grid)
plt.axis('off')

plt.show()