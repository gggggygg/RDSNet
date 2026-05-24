import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.colors as mcolors

# 设置图像参数
image_size = 512
center = (image_size // 2, image_size // 2)
max_radius = image_size // 2 * 0.8  # 最大半径（避免太靠近边缘）

# 创建画布
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, image_size)
ax.set_ylim(0, image_size)
ax.set_aspect('equal')
ax.invert_yaxis()  # 图像坐标系（左上角为原点）
# ax.set_title('Deformable Convolution Kernels in Fisheye Image', fontsize=15)
# ax.set_xlabel('X Position', fontsize=12)
# ax.set_ylabel('Y Position', fontsize=12)

# 绘制图像中心标记
ax.plot(center[0], center[1], 'ro', markersize=8, label='Image Center')
ax.text(center[0] + 10, center[1] + 10, 'Center', color='red', fontsize=12)

def plot_radial_kernel(num_rays, num_kernels_per_ray = 4, plot_ri = [0,1], dilation_min = 1.0, dilation_max = 2.5):
    # 参数设置
    # num_rays = 16  # 辐射方向数量
    # num_kernels_per_ray = 4  # 每条射线上的卷积核数量
    # dilation_min = 1.0  # 边缘膨胀系数
    # dilation_max = 2.5  # 中心膨胀系数

    # 生成卷积核位置
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    radii = np.linspace(35, max_radius, num_kernels_per_ray)

    # 创建颜色映射表示膨胀系数
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=dilation_min, vmax=dilation_max)

    # 用于图例的标记
    legend_elements = []

    # 绘制每个卷积核
    for angle in angles:
        # for ri, r in enumerate(radii):
        #     if ri<2:
        #         continue
        for ri in plot_ri:
            r = radii[ri]
            # 计算卷积核中心坐标
            kernel_center_x = center[0] + r * np.cos(angle)
            kernel_center_y = center[1] + r * np.sin(angle)

            # 计算当前半径下的膨胀系数（线性插值）
            # dilation = dilation_max - (dilation_max - dilation_min) * (r / max_radius)

            # 计算当前半径下的膨胀系数（非线性插值，中心变化更快）
            # 使用二次函数使中心变化更明显
            t = (r / max_radius) ** 0.5
            dilation = dilation_min + (dilation_max - dilation_min) * (1 - t)

            # 计算指向中心的方向向量
            dx = center[0] - kernel_center_x
            dy = center[1] - kernel_center_y
            direction_angle = np.arctan2(dy, dx)

            # 计算单位方向向量
            length = np.sqrt(dx ** 2 + dy ** 2)
            if length > 0:
                ux = dx / length
                uy = dy / length
            else:
                ux, uy = 0, 0  # 处理中心点特殊情况

            # 固定箭头参数
            arrow_length = 15  # 固定长度（像素）
            arrow_offset = 0  # 从卷积核中心开始的偏移量

            # 绘制指向中心的箭头
            arrow = FancyArrowPatch(
                (kernel_center_x + ux * arrow_offset,  # 起点：从中心向外偏移
                 kernel_center_y + uy * arrow_offset),
                (kernel_center_x + ux * (arrow_offset - arrow_length),  # 终点：延长固定长度
                 kernel_center_y + uy * (arrow_offset - arrow_length)),
                arrowstyle='->',
                mutation_scale=15,  # 固定箭头大小
                color=cmap(norm(dilation)),  # 使用卷积核颜色
                linewidth=1.5,
                alpha=0.8
            )
            ax.add_patch(arrow)


            # 绘制指向中心的虚线
            # ax.plot([kernel_center_x, center[0]], [kernel_center_y, center[1]],
            #         'k--', alpha=0.3, linewidth=0.7)

            # 绘制卷积核中心
            ax.plot(kernel_center_x, kernel_center_y, 'bo', markersize=6, alpha=0.7)

            # 生成3x3采样网格
            grid_size = 0.5

            grid = np.array([[-grid_size, -grid_size], [-grid_size, 0], [-grid_size, grid_size],
                             [0, -grid_size], [0, 0], [0, grid_size],
                             [grid_size, -grid_size], [grid_size, 0], [grid_size, grid_size]])

            # 应用膨胀系数
            grid = grid * dilation * 10  # 缩放因子使点在图中可见

            # 旋转网格使其指向中心
            rotation_angle = np.arctan2(center[1] - kernel_center_y, center[0] - kernel_center_x)
            rotation_matrix = np.array([
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)]
            ])

            # 绘制采样点
            for point in grid:
                rotated_point = np.dot(rotation_matrix, point)
                px = kernel_center_x + rotated_point[0]
                py = kernel_center_y + rotated_point[1]
                ax.plot(px, py, 'o', markersize=5,
                        color=plt.cm.viridis(1 - r / max_radius),  # 颜色表示膨胀系数
                        alpha=0.8)

dilation_min = 1.0  # 边缘膨胀系数
dilation_max = 2.0  # 中心膨胀系数
num_kernels_per_ray = 6

plot_radial_kernel(num_rays=6, num_kernels_per_ray=num_kernels_per_ray, plot_ri=[0], dilation_min=dilation_min, dilation_max=dilation_max)
plot_radial_kernel(num_rays=10, num_kernels_per_ray=num_kernels_per_ray, plot_ri=[1], dilation_min=dilation_min, dilation_max=dilation_max)
plot_radial_kernel(num_rays=20, num_kernels_per_ray=num_kernels_per_ray, plot_ri=[2], dilation_min=dilation_min, dilation_max=dilation_max)
plot_radial_kernel(num_rays=25, num_kernels_per_ray=num_kernels_per_ray, plot_ri=[3], dilation_min=dilation_min, dilation_max=dilation_max)
plot_radial_kernel(num_rays=55, num_kernels_per_ray=num_kernels_per_ray, plot_ri=[4], dilation_min=dilation_min, dilation_max=dilation_max)
plot_radial_kernel(num_rays=70, num_kernels_per_ray=num_kernels_per_ray, plot_ri=[5], dilation_min=dilation_min, dilation_max=dilation_max)
# plot_radial_kernel(num_rays=64, num_kernels_per_ray=num_kernels_per_ray, plot_ri=[4], dilation_min=dilation_min, dilation_max=dilation_max)

# 创建颜色条表示膨胀系数
sm = plt.cm.ScalarMappable(cmap='viridis',
                           norm=plt.Normalize(vmin=dilation_min, vmax=dilation_max))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Dilation Factor', fontsize=12)

# 添加图例和注释
# ax.text(5, 30, 'Kernel Properties:', fontsize=12, weight='bold')
# ax.text(5, 60, '- Center: Large dilation (sparse)', fontsize=10, color='blue')
# ax.text(5, 90, '- Edge: Small dilation (dense)', fontsize=10, color='green')
# ax.text(center[0] - 250, center[1] - 300,
#         'Convolution kernels adapt to fisheye distortion:\n'
#         '• Radial orientation from center\n'
#         '• Dense sampling near edges (small dilation)\n'
#         '• Sparse sampling at center (large dilation)',
#         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('deformable_conv_fisheye.png', dpi=300, bbox_inches='tight')
plt.show()