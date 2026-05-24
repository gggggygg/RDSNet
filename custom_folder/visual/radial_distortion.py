import numpy as np
import cv2
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.units import inch

def create_chessboard(size=(800, 800), grid_num=8):
    """创建原始棋盘格（兼容彩色/灰度）"""
    chessboard = np.ones((size[0], size[1]), dtype=np.uint8) * 255
    grid_size = size[0] // grid_num
    for i in range(grid_num):
        for j in range(grid_num):
            if (i + j) % 2 == 0:
                x1, y1 = j * grid_size, i * grid_size
                x2, y2 = x1 + grid_size, y1 + grid_size
                cv2.rectangle(chessboard, (x1, y1), (x2, y2), 0, -1)
    return chessboard


def apply_barrel_distortion_fixed(img, k1=0.00005, center=None):
    """畸变 + 自动放大修正：保证输出尺寸与原图一致"""
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    # 1. 正向计算畸变范围（找到最大/最小坐标，确定缩放系数）
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    x_min, x_max, y_min, y_max = w, 0, h, 0  # 初始化边界

    for y in range(h):
        for x in range(w):
            dx = x - center[0]
            dy = y - center[1]
            r2 = dx * dx + dy * dy
            radial_factor = 1 + k1 * r2

            x_dist = center[0] + dx * radial_factor
            y_dist = center[1] + dy * radial_factor

            # 更新坐标边界
            x_min = min(x_min, x_dist)
            x_max = max(x_max, x_dist)
            y_min = min(y_min, y_dist)
            y_max = max(y_max, y_dist)

    # 2. 计算缩放系数（将畸变后的坐标映射回原图尺寸）
    scale_x = w / (x_max - x_min + 1e-8)  # +1e-8避免除零
    scale_y = h / (y_max - y_min + 1e-8)

    # 3. 重新计算映射表（带缩放修正）
    for y in range(h):
        for x in range(w):
            dx = x - center[0]
            dy = y - center[1]
            r2 = dx * dx + dy * dy
            radial_factor = 1 + k1 * r2

            x_dist = center[0] + dx * radial_factor
            y_dist = center[1] + dy * radial_factor

            # 缩放并平移坐标，填充原图尺寸
            x_remap = (x_dist - x_min) * scale_x
            y_remap = (y_dist - y_min) * scale_y

            map_x[y, x] = x_remap
            map_y[y, x] = y_remap

    # 4. 应用重映射（带边界填充）
    distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return distorted


# 生成测试图像
chessboard = create_chessboard(size=(800, 800), grid_num=15)
distorted = apply_barrel_distortion_fixed(chessboard, k1=0.000002)

# 对比显示
plt.figure(figsize=(12, 6))
plt.subplot(121)
# plt.title("原始棋盘格")
plt.imshow(chessboard, cmap='gray')
plt.axis('off')

plt.subplot(122)
# plt.title("畸变+放大修正")
plt.imshow(distorted, cmap='gray')
plt.axis('off')
plt.show()

cv2.imwrite('temp_original.png', chessboard)
cv2.imwrite('temp_distorted.png', distorted)


# 用 reportlab 生成 PDF
def create_pdf():
    doc = SimpleDocTemplate("chessboard_report.pdf", pagesize=letter)
    elements = []

    # 添加原始图
    img_original = Image('temp_original.png', width=4 * inch, height=4 * inch)
    elements.append(img_original)

    # 添加畸变图
    img_distorted = Image('temp_distorted.png', width=4 * inch, height=4 * inch)
    elements.append(img_distorted)

    doc.build(elements)


create_pdf()