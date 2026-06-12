#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: WLH
@file: example_mar.py
@time: 2025/9/28 11:35
"""
'''
实现线性插值金属伪影消除，流程如下
配置文件：settings.bin
原始投影图：processed_raw
1. 调用接口对数据进行一次重建得到v1
2. 对v1进行阈值分割，得到体数据掩码v_mask
3. 对v_mask调用接口进行前向投影得到投影图像的p_mask
4. 投影图中的mask区域，用mask以外的区域的值线性插值填充（只在行方向进行）
5. 对填充的投影图进行重建，得到v2
6. 对v2中的v_mask用v1的值进行替换，边界加一些过渡，得到最终结果
'''

import numpy as np
import scipy.ndimage as ndimage
from recon_api import ReconAPI, load_projections_from_raw_files, save_raw, save_mhd
import os

def linear_interpolate_row(row_data, mask):
    """
    对单行数据进行线性插值，填充mask区域

    Args:
        row_data: 一维数组，行数据
        mask: 一维布尔数组或0/1数组，1表示需要插值的位置

    Returns:
        插值后的行数据
    """
    result = row_data.copy()
    mask_bool = mask > 0.5  # 转换为布尔类型

    if not np.any(mask_bool):
        return result  # 没有需要插值的点

    # 找到所有非mask的点
    valid_indices = np.where(~mask_bool)[0]

    if len(valid_indices) == 0:
        return result  # 整行都是mask，无法插值

    # 找到需要插值的点
    interp_indices = np.where(mask_bool)[0]

    # 使用numpy的线性插值
    # 对于超出范围的点，使用最近的有效值
    result[interp_indices] = np.interp(interp_indices, valid_indices, row_data[valid_indices])

    return result


def interpolate_projection_linear(proj_data, p_mask, threshold=0.5):
    """
    对投影数据进行行方向线性插值

    Args:
        proj_data: 投影数据，形状 (noView, vNum, uNum)
        p_mask: 投影掩码，形状 (noView, vNum, uNum)，值越大表示金属
        threshold: 掩码阈值，大于该值的区域需要插值

    Returns:
        插值后的投影数据
    """
    proj_interpolated = proj_data.copy()
    binary_mask = (p_mask > threshold).astype(np.float32)

    num_views, v_num, u_num = proj_data.shape

    print(f"  Interpolating projections...")
    for view_idx in range(num_views):
        if view_idx % 100 == 0:
            print(f"    Processing view {view_idx}/{num_views}")

        for v_idx in range(v_num):
            # 对每一行进行插值
            row = proj_data[view_idx, v_idx, :]
            mask_row = binary_mask[view_idx, v_idx, :]

            if np.any(mask_row > 0.5):
                proj_interpolated[view_idx, v_idx, :] = linear_interpolate_row(row, mask_row)

    return proj_interpolated


def apply_gaussian_dilation(mask, kernel_size=5):
    """
    使用三维高斯滤波扩大金属区域

    Args:
        mask: 二值掩码
        kernel_size: 高斯核大小（像素），控制膨胀的像素数
                    实际膨胀约 kernel_size//2 个像素

    Returns:
        扩大后的掩码（二值）
    """
    # 根据核大小计算 sigma
    sigma = kernel_size / 6.0

    # 对二值掩码进行高斯滤波
    smoothed = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)

    # 取滤波后大于0的区域
    dilated_mask = (smoothed > 0).astype(np.float32)

    return dilated_mask


def create_smooth_transition_mask(v_mask, kernel_size=9):
    """
    使用三维高斯滤波创建平滑过渡掩码，用于边界融合

    Args:
        v_mask: 二值掩码
        kernel_size: 高斯核大小（像素），控制过渡区域的宽度
                    过渡宽度约为 kernel_size//2 个像素

    Returns:
        平滑过渡掩码，0-1之间的值
    """
    # 根据核大小计算 sigma
    sigma = kernel_size / 6.0

    # 对二值掩码进行高斯滤波，得到0-1之间的平滑过渡
    # 金属区域中心为1，边界处平滑过渡到0
    smooth_mask = ndimage.gaussian_filter(v_mask.astype(np.float32), sigma=sigma, radius=kernel_size // 2)

    # 归一化到0-1范围
    if smooth_mask.max() > 0:
        smooth_mask = smooth_mask / smooth_mask.max()

    return smooth_mask


def merge_volumes_with_transition(v1, v2, v_mask, kernel_size=9):
    """
    将v1和v2融合，在金属区域使用v1，非金属区域使用v2，边界平滑过渡

    Args:
        v1: 原始重建体数据
        v2: 插值后重建的体数据
        v_mask: 金属区域掩码
        kernel_size: 高斯核大小（像素），控制过渡的平滑程度

    Returns:
        融合后的体数据
    """
    # 创建平滑过渡掩码
    smooth_mask = create_smooth_transition_mask(v_mask, kernel_size)

    # 融合：金属区域用v1，非金属区域用v2，边界平滑过渡
    result = v1 * smooth_mask + v2 * (1.0 - smooth_mask)

    return result


def metal_artifact_reduction(config_file, proj_data_dir,
                             metal_threshold=1.0,
                             dilation_kernel=5,
                             transition_kernel=9):
    """
    执行金属伪影消除

    Args:
        config_file: 配置文件路径
        proj_data_dir: 投影数据目录
        metal_threshold: 金属阈值（重建��）
        dilation_kernel: 膨胀高斯核大小（像素），控制掩码扩大范围
        transition_kernel: 过渡高斯核大小（像素），控制边界融合平滑程度

    Returns:
        (v1, v2, v_final): 原始重建、插值重建、最终结果
    """
    print("=" * 60)
    print("Metal Artifact Reduction (Linear Interpolation)")
    print("=" * 60)

    # 初始化API
    print("\n[1] Initializing ReconAPI...")
    api = ReconAPI()
    print(f"    Loaded library: {api.lib_path}")

    # 获取数据形状
    print("\n[2] Loading configuration...")
    vol_shape, proj_shape = api.get_shape(config_file)
    print(f"    Volume shape: {vol_shape}")
    print(f"    Projection shape: {proj_shape}")

    # 加载投影数据
    print("\n[3] Loading projection data...")
    proj_data = load_projections_from_raw_files(
        proj_data_dir, proj_shape[2], proj_shape[0], proj_shape[1]
    )
    print(f"    Loaded projection data: {proj_data.shape}")

    # Step 1: 初始FBP重建
    print("\n[4] Step 1: Initial FBP reconstruction...")
    v1 = api.fbp(proj_data, config_file)
    print(f"    Reconstructed v1: {v1.shape}")
    print(f"    v1 range: [{v1.min():.2f}, {v1.max():.2f}]")

    # Step 2: 阈值分割得到金属掩码
    print(f"\n[5] Step 2: Threshold segmentation (threshold={metal_threshold})...")
    v_mask = (v1 > metal_threshold).astype(np.float32)
    metal_voxels = np.sum(v_mask)
    total_voxels = v_mask.size
    print(f"    Metal voxels: {metal_voxels} / {total_voxels} ({100 * metal_voxels / total_voxels:.2f}%)")
    # 使用高斯滤波扩大掩码，确保覆盖金属伪影
    if dilation_kernel > 0:
        print(f"    Applying Gaussian dilation (kernel_size={dilation_kernel})...")
        v_mask = apply_gaussian_dilation(v_mask, kernel_size=dilation_kernel)
        metal_voxels_dilated = np.sum(v_mask)
        print(f"    Dilated metal voxels: {metal_voxels_dilated} ({100 * metal_voxels_dilated / total_voxels:.2f}%)")

    # Step 3: 前向投影得到投影掩码
    print("\n[6] Step 3: Forward projection of metal mask...")
    # 需要将v_mask转换为与v1相同的形状
    v_mask_volume = np.zeros_like(v1)
    v_mask_volume[:] = v_mask
    p_mask = api.fp(v_mask_volume, config_file)
    print(f"    Projection mask: {p_mask.shape}")
    print(f"    p_mask range: [{p_mask.min():.2f}, {p_mask.max():.2f}]")

    # Step 4: 线性插值填充投影
    print("\n[7] Step 4: Linear interpolation in projection domain...")
    p_threshold = 0.1
    proj_interpolated = interpolate_projection_linear(proj_data, p_mask, threshold=p_threshold)
    print(f"    Interpolated projection data: {proj_interpolated.shape}")

    # Step 5: 对插值后的投影重建
    print("\n[8] Step 5: Second FBP reconstruction with interpolated projections...")
    v2 = api.fbp(proj_interpolated, config_file)
    print(f"    Reconstructed v2: {v2.shape}")
    print(f"    v2 range: [{v2.min():.2f}, {v2.max():.2f}]")

    # Step 6: 融合v1和v2
    print(f"\n[9] Step 6: Merging volumes (transition_kernel={transition_kernel})...")
    v_final = merge_volumes_with_transition(v1, v2, v_mask, transition_kernel)
    print(f"    Final volume: {v_final.shape}")
    print(f"    v_final range: [{v_final.min():.2f}, {v_final.max():.2f}]")

    print("\n[10] MAR processing completed!")
    print("=" * 60)

    return v1, v2, v_final, v_mask


if __name__ == '__main__':
    # 配置参数
    data_path = r'E:\code\YOFO_重建工具\data\YOFO_No_Metal_001_Jirox_CT1613_00042_73078'
    config_file = os.path.join(data_path, "YOFO_config.bin")
    proj_data_dir = os.path.join(data_path, "YOFO_raw")

    # 金属吸收系数阈值：需要根据实际数据调整
    metal_threshold = 1.0

    # 高斯核大小：扩大金属区域以覆盖伪影
    # 实际膨胀约 dilation_kernel//2 个像素
    dilation_kernel = 5  # 膨胀约2-3个像素

    # 过渡高斯核大小：边界融合的平滑程度
    transition_kernel = 9  # 过渡宽度约4-5个像素

    # 执行MAR
    v1, v2, v_final, v_mask = metal_artifact_reduction(config_file, proj_data_dir, metal_threshold, dilation_kernel,
                                                       transition_kernel)

    # 保存结果
    print("\n[11] Saving results...")
    save_mhd(v1, "mar_v1_original.mhd")
    print("    Saved: mar_v1_original.mhd")

    save_mhd(v2, "mar_v2_interpolated.mhd")
    print("    Saved: mar_v2_interpolated.mhd")

    save_mhd(v_final, "mar_final_result.mhd")
    print("    Saved: mar_final_result.mhd")

    save_mhd(v_mask, "mar_metal_mask.mhd")
    print("    Saved: mar_metal_mask.mhd")

    print("\nAll results saved successfully!")
    print("\nOutput files:")
    print("  - mar_v1_original.mhd: Original FBP reconstruction")
    print("  - mar_v2_interpolated.mhd: Reconstruction from interpolated projections")
    print("  - mar_final_result.mhd: Final MAR result (merged)")
    print("  - mar_metal_mask.mhd: Metal segmentation mask")
