import json
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import root_scalar

def get_uvr_list(json_file = '/ric_nas/Overhead_fisheye\LOAF/annotations/resolution_1k/instances_val-seen.json', scene = None):
    u_list, v_list, r_list = [], [], []
    h_list = []
    theta_list = []
    R_list = []
    H_list = []
    Fai_list = []

    imgid2img = {}
    with open(json_file, 'r') as f:
        data = json.load(f)

        for image in data['images']:
            imgid2img[image['id']] = image

        for ann in data['annotations']:
            imgid = ann['image_id']
            img_file_name = imgid2img[imgid]['file_name']
            img_scene = img_file_name.split('_')[0]

            if scene==None or img_scene == scene:
                img_u = ann['rotated_box'][0]
                img_v = ann['rotated_box'][1]
                img_h = ann['rotated_box'][3]
                img_r = ann['person_location'][0]
                img_Fai = ann['world_location'][1]

                u_list.append(img_u)
                v_list.append(img_v)
                r_list.append(img_r)
                h_list.append(img_h)
                Fai_list.append(img_Fai)

                R = ann['person_location'][4]
                H = ann['camera_height']

                theta = np.arctan2(R, H)  # 入射角

                theta_list.append(theta)
                R_list.append(R)
                H_list.append(H)

    return u_list, v_list, r_list, h_list, theta_list, R_list, H_list, Fai_list


def solve_principal_point(u_list, v_list, r_list):
    """
    求解主点坐标(u0, v0)
    参数:
        u_list: 像素u坐标列表
        v_list: 像素v坐标列表
        r_list: 对应r值列表
    返回:
        u0, v0: 求解的主点坐标
        residuals: 每个样本的残差（用于验证精度）
    """
    # 确保输入数据长度一致
    if len(u_list) != len(v_list) or len(u_list) != len(r_list):
        raise ValueError("u, v, r列表长度必须一致")
    N = len(u_list)
    if N < 2:
        raise ValueError("样本数量至少为2")

    # 提取第一个样本作为基准
    u1, v1, r1 = u_list[0], v_list[0], r_list[0]
    A = []
    b = []

    # 构建方程组（i从1开始，与第一个样本作差）
    for i in range(1, N):
        ui, vi, ri = u_list[i], v_list[i], r_list[i]
        # 系数矩阵A的行：[2*(ui - u1), 2*(vi - v1)]
        A_row = [2 * (ui - u1), 2 * (vi - v1)]
        # 常数项b的元素：(ui² + vi² - ri²) - (u1² + v1² - r1²)
        b_val = (ui ** 2 + vi ** 2 - ri ** 2) - (u1 ** 2 + v1 ** 2 - r1 ** 2)
        A.append(A_row)
        b.append(b_val)

    # 转换为numpy数组
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    # 最小二乘法求解：x = (A^T A)^(-1) A^T b
    A_T = A.T
    x = np.linalg.inv(A_T @ A) @ (A_T @ b)
    u0, v0 = x[0], x[1]

    # 计算残差（验证求解精度）
    residuals = []
    for u, v, r in zip(u_list, v_list, r_list):
        r_pred = np.sqrt((u - u0) ** 2 + (v - v0) ** 2)
        residuals.append(abs(r_pred - r))
    residuals = np.array(residuals)

    return u0, v0, residuals



def estimate_camera_center(u, v, r, h):
    """
    根据已知 u, v, r 值估计相机主点 (u0, v0)
    """
    u = np.array(u)
    v = np.array(v)
    h = np.array(h)
    r = np.array(r)

    def residuals(params):
        u0, v0 = params
        pred_r = np.sqrt((u - u0)**2 + (v - v0)**2) - h/2
        return pred_r - r

    # 初始猜测设为图像中心（假设图像为 1024x1024）
    u0_init = 512
    v0_init = 512

    result = least_squares(residuals, x0=[u0_init, v0_init])
    return result.x  # [u0, v0]


def estimate_fisheye_distortion_coeffs(theta, r, n=5):
    """
    拟合鱼眼畸变模型 r = Σ k_i * θ^(2i - 1)，共 n 项。

    参数:
        theta: ndarray, 入射角 (单位：弧度)
        r: ndarray, 像素半径
        n: int, 多项式的项数（即拟合的系数个数）

    返回:
        coeffs: ndarray, shape (n,), 拟合得到的 [k1, k2, ..., kn]
    """
    theta = np.asarray(theta)
    r = np.asarray(r)

    # 构建设计矩阵：每列是 θ 的奇数次幂 θ^(2i - 1)
    X = np.stack([theta ** (2 * i - 1) for i in range(1, n + 1)], axis=1)

    # 最小二乘拟合
    coeffs, residuals, rank, s = np.linalg.lstsq(X, r, rcond=None)
    return coeffs


def calculate_coeffs(theta_list, r_list, save_path = "fisheye_fit_results.json", n_max = 30):
    results = {}

    for k_n in range(1, n_max):
        coeffs = estimate_fisheye_distortion_coeffs(theta_list, r_list, n=k_n)

        residuals = np.array([
            abs(sum(coeffs[i] * theta ** (2 * i + 1) for i in range(len(coeffs))) - r)
            for r, theta in zip(r_list, theta_list)
        ])

        sorted_residuals = np.sort(residuals)
        if len(sorted_residuals) >= 100:
            top_100_residual = sorted_residuals[-100]
        else:
            top_100_residual = sorted_residuals[0]

        # 保存为 dict，注意 numpy -> float 以便 json 序列化
        results[str(k_n)] = {
            "coeffs": coeffs.tolist(),
            "max": float(residuals.max()),
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "top100": float(top_100_residual)
        }

        print(f"k_n: {k_n}, max: {residuals.max()}, mean: {residuals.mean()}, std: {residuals.std()}, top100: {top_100_residual}")

    # 保存为 JSON 文件
    print(f"Saving results to {save_path}")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

def theta_from_r(r_value, coeffs, theta_max=np.pi/2):
    """
    根据 r 和拟合系数 coeffs，求解对应的 theta。
    参数:
        r_value: float，输入的像素半径 r
        coeffs: list 或 ndarray，拟合得到的畸变模型系数 [k1, k2, ..., kn]
        theta_max: 最大可能的 θ 值，默认 π/2（可根据视场调大）

    返回:
        theta: float, 入射角 θ（单位：弧度）
    """

    def distortion_function(theta):
        return sum(coeffs[i] * theta**(2 * i + 1) for i in range(len(coeffs))) - r_value

    # 数值解方程 f(θ) = 0，初始区间 [0, theta_max]
    sol = root_scalar(distortion_function, bracket=[0, theta_max], method='brentq')

    if sol.converged:
        return sol.root
    else:
        raise RuntimeError(f"Failed to solve θ for r = {r_value}")

def compute_fai(u, v, u0 = 512, v0 = 512):
    du = u - u0
    dv = v - v0
    fai = np.arctan2(-dv, du)  # 注意是 -dv，确保 Y 轴向下变为向上

    fai_deg = np.degrees(fai)
    if fai_deg < 0:
        fai_deg += 360  # 转为 [0, 360) 度范围
    return fai_deg  # 单位是角度，逆时针为正，右方为0


def polar_to_cartesian(R, fai_deg):
    """
    将极坐标 [R, fai] 转换为笛卡尔坐标 [X, Y]

    参数:
        R: float or np.ndarray，目标距离（单位：米）
        fai_deg: float or np.ndarray，方向角（单位：度，右方为0°，逆时针为正）

    返回:
        X, Y: 对应的笛卡尔坐标
    """
    fai_rad = np.radians(fai_deg)  # 转为弧度
    X = R * np.cos(fai_rad)
    Y = R * np.sin(fai_rad)
    return X, Y


if __name__ == "__main__":
    u_list, v_list, r_list, h_list, theta_list, R_list, H_list, Fai_list = get_uvr_list(json_file='/ric_nas/Overhead_fisheye/LOAF/annotations/resolution_1k/instances_val.json', scene=None)

    # u0, v0, residuals = solve_principal_point(u_list, v_list, r_list)
    # u0, v0 = estimate_camera_center(u_list, v_list, r_list, h_list)

    # print(u0,v0)
    #
    # # 计算残差（验证求解精度）
    # residuals = []
    # for u, v, r, h in zip(u_list, v_list, r_list, h_list):
    #     r_pred = np.sqrt((u - u0) ** 2 + (v - v0) ** 2) - h/2
    #     residuals.append(abs(r_pred - r))
    # residuals = np.array(residuals)

    # calculate_coeffs(theta_list, r_list, save_path="fisheye_fit_results.json", n_max=30)

    with open("fisheye_fit_results.json", 'r') as f:
        coeffs_data = json.load(f)

    for k_n in range(14,15):
        coeffs = np.array(coeffs_data[str(k_n)]["coeffs"], dtype=np.float64)

        ##### #计算从theta到r的残差，   计算标定残差（验证拟合精度，从theta到r），在k_n为14时最为准确。
        # residuals = np.array([
        #     abs(sum(coeffs[i] * theta ** (2 * i + 1) for i in range(len(coeffs))) - r)
        #     for r, theta in zip(r_list, theta_list)
        # ])

        #####计算从r到theta的残差，比较准确：k_n: 14, max: 0.001060966319898382, mean: 1.838590417292822e-05, std: 2.4399305497062443e-05, top100: 0.00022994823927641184
        # residuals = np.array([
        #     abs(theta_from_r(r, coeffs) - theta)
        #     for r, theta in zip(r_list, theta_list)
        # ])

        ####计算从r到R的残差，  计算 坐标转换残差 有几个离谱的离群点但是数量很少，原因未知，可能是标注错误，基本能用 k_n: 14, max: 94.84705777916224, mean: 0.09130449115017436, std: 0.4854535158464048, top10: 0.46061118409352275
        # residuals = np.array([
        #     abs(H*np.tan(theta_from_r(r, coeffs)) - R)
        #     for r, R, H in zip(r_list, R_list, H_list)
        # ])

        ###计算从u,v到fai的残差，  验证极坐标系角度计算公式，比较准确：k_n: 14, max: 1.1368683772161603e-13, mean: 5.213104929259402e-15, std: 1.1562967255902449e-14, top10: 5.684341886080802e-14
        residuals = np.array([
            abs(compute_fai(u, v) - Fai)
            for u,v,Fai in zip(u_list, v_list, Fai_list)
        ])

        # 计算残差的统计信息
        sorted_residuals = np.sort(residuals)
        if len(sorted_residuals) >= 10:
            top_10_residual = sorted_residuals[-10]
        else:
            top_10_residual = sorted_residuals[0]

        print(f"k_n: {k_n}, max: {residuals.max()}, mean: {residuals.mean()}, std: {residuals.std()}, top10: {top_10_residual}")


    x = 1

    # print('u_list:', u_list)
    # print('v_list:', v_list)
    # print('r_list:', r_list)
    # print('len(u_list):', len(u_list))


