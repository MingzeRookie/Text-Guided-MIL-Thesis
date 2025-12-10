# 导入必要的库
import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from multiprocessing import Pool
from pathlib import Path
import subprocess

# 参数设置
SEG_AREA = 50
PATCH_SIZE = 224
thumbnail_level = -11  # 缩略图层级（用于获取低倍率图像）
target_mag = -2        # 切片时使用的倍率（20x）
slide_dirs = Path("../datasets/WSI")  # 输入文件夹路径（幻灯片所在文件夹） 
output_dir = Path('../datasets/patches')      # 输出文件夹（切片保存位置）

# 使用 ls 命令列出 .mrxs 文件
output = subprocess.run(["ls", str(slide_dirs)], capture_output=True, text=True)
slide_files = [f for f in output.stdout.splitlines() if f.endswith('.mrxs')]
print(f"Number of slide files: {len(slide_files)}")

# 创建输出文件夹（如果不存在）
output_dir.mkdir(parents=True, exist_ok=True)

def is_white_patch(patch_img, white_threshold=0.9):
    """检测白色像素占比超过阈值的切片"""
    # 转换到HSV空间提取亮度通道
    hsv = cv2.cvtColor(patch_img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    
    # 计算高亮度像素占比（V值>220）
    white_pixels = np.sum(v > 220)
    total_pixels = v.size
    return (white_pixels / total_pixels) > white_threshold

def process_slide(slide_file):
    slide_path = slide_dirs / slide_file
    slide_name = slide_file.split('.')[0]
    slide_output_dir = output_dir / slide_name

    # 如果输出文件夹已存在且包含切片文件，则跳过该幻灯片（可根据需要调整判断条件）
    if slide_output_dir.exists() and any(slide_output_dir.glob("*.png")):
        print(f"Slide {slide_name} already processed. Skipping.")
        return

    # 打开幻灯片并创建 DeepZoomGenerator 对象
    slide = open_slide(str(slide_path))
    dz = DeepZoomGenerator(slide, PATCH_SIZE, overlap=0, limit_bounds=True)

    # 获取缩略图用于检测病变区域
    thumbnail = dz.get_tile(dz.level_count + thumbnail_level, (0, 0))
    thumbnail_img = cv2.cvtColor(np.asarray(thumbnail, dtype=np.uint8), cv2.COLOR_RGB2BGR)

    # 创建该幻灯片的输出文件夹
    slide_output_dir.mkdir(parents=True, exist_ok=True)

    # 转换到 HSV 颜色空间，使用颜色阈值检测紫色病变区域（可根据具体样本调整）
    hsv_img = cv2.cvtColor(thumbnail_img, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120, 40, 40])
    upper_purple = np.array([160, 255, 255])
    purple_mask = cv2.inRange(hsv_img, lower_purple, upper_purple)

    # 应用形态学操作去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    cnts, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 选择面积最大的轮廓
    max_area = 0
    largest_contour = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area > max_area and area > SEG_AREA:
            max_area = area
            largest_contour = c

    # 如果找到了病变区域
    if largest_contour is not None:
        # 在缩略图上绘制最大轮廓，并保存
        cv2.drawContours(thumbnail_img, [largest_contour], -1, (0, 0, 255), 5)
        plt.imshow(cv2.cvtColor(thumbnail_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(slide_output_dir / 'thumbnail.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # 计算病变区域的最小矩形边界
        x, y, w, h = cv2.boundingRect(largest_contour)
        # 根据不同层级的倍率关系，计算缩放因子
        factor = target_mag - thumbnail_level
        t = 2 ** factor
        # 将 bounding rectangle 转换为 DeepZoom 切片的行列号
        xt = math.floor(x * t / PATCH_SIZE)
        yt = math.floor(y * t / PATCH_SIZE)
        wt = math.ceil(w * t / PATCH_SIZE)
        ht = math.ceil(h * t / PATCH_SIZE)

        # 对 bounding rectangle 内的每个 tile 进行切片，并以坐标命名保存图片
        for col in range(xt, xt + wt):
            for row in range(yt, yt + ht):
                try:
                    tile = dz.get_tile(dz.level_count + target_mag, (col, row))
                    patch_img = cv2.cvtColor(np.asarray(tile, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                    # 检测白色切片并跳过
                    if is_white_patch(patch_img):
                        continue
                    # 使用 "列_行.png" 命名切片
                    filename = f"{col}_{row}.png"
                    cv2.imwrite(str(slide_output_dir / filename), patch_img)
                except ValueError as e:
                    print(f"Skipping tile at (col, row): ({col}, {row}) due to error: {e}")

        print(f"Finished processing: {slide_name}")
    else:
        print(f"No lesion region found in slide: {slide_name}")

if __name__ == "__main__":
    # 使用多进程处理每个幻灯片
    with Pool(processes=4) as pool:
        pool.map(process_slide, slide_files)
