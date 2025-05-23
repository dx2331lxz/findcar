#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Canny算子和边缘点统计的车辆轮廓识别方法
"""

import cv2
import numpy as np
import os
import argparse


class SimpleDetector:
    """简化版车辆轮廓检测器类"""
    
    def __init__(self):
        # 默认参数 - 对小图像更友好
        self.gaussian_kernel_size = 3  # 小图像用小内核
        
        # 动态参数调整
        self.width_threshold_min = 20   # 最小宽度阈值
        self.height_threshold_min = 20  # 最小高度阈值
        self.area_min_ratio = 0.05      # 最小面积比例

        # 启用调试模式
        self.debug = False
    
    def detect(self, image_path, output_path=None, debug=False):
        """检测车辆轮廓的主函数"""
        self.debug = debug
        print(f"开始处理图像: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误：无法读取图像 {image_path}")
            return None
        
        # 获取图像尺寸
        height, width = image.shape[:2]
        print(f"图像尺寸: {image.shape}")
        
        # 对于小图像，调整参数
        is_small_image = width < 200 or height < 200
        
        # 步骤1: 灰度转换
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("步骤1完成：灰度转换")
        
        # 步骤2: 高斯模糊
        kernel_size = 3 if is_small_image else 5
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        print(f"步骤2完成：高斯模糊（内核大小={kernel_size}）")
        
        # 步骤3-5: 边缘检测（直接使用OpenCV的Canny）
        # 对于小图像，使用更保守的阈值
        if is_small_image:
            low = 50
            high = 150
        else:
            # 自适应阈值
            median = np.median(blurred)
            low = int(max(0, 0.5 * median))
            high = int(min(255, 1.5 * median))
        
        edges = cv2.Canny(blurred, low, high)
        print(f"步骤3-5完成：边缘检测（阈值:{low}-{high}）")
        
        # 形态学操作以连接边缘
        if is_small_image:
            kernel = np.ones((2, 2), np.uint8)
        else:
            kernel = np.ones((3, 3), np.uint8)
            
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # 步骤6: 轮廓检测和筛选
        contours = self._detect_contours(edges, image, is_small_image)
        print(f"步骤6完成：检测到 {len(contours)} 个车辆轮廓")
        
        # 绘制结果
        result_image = image.copy()
        if contours:
            for i, (x, y, w, h) in enumerate(contours):
                # 绘制矩形
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 添加标签
                cv2.putText(result_image, f'Vehicle {i+1}', (x, max(y-5, 15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 保存结果
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_result.jpg"
            binary_output_path = f"{base_name}_binary.jpg"
        else:
            binary_output_path = f"{os.path.splitext(output_path)[0]}_binary.jpg"
        
        cv2.imwrite(binary_output_path, edges)
        cv2.imwrite(output_path, result_image)
        
        print(f"结果已保存:")
        print(f"  - 二值边缘图: {binary_output_path}")
        print(f"  - 轮廓检测结果: {output_path}")
        
        return {
            'contours': contours,
            'edges': edges,
            'result': result_image,
            'thresholds': (low, high)
        }
    
    def _detect_contours(self, edges, original_image, is_small_image=False):
        """检测和筛选轮廓"""
        height, width = edges.shape
        image_area = height * width
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 调整参数
        if is_small_image:
            min_area = image_area * 0.05  # 小图像降低面积阈值
            min_width = min(self.width_threshold_min, width * 0.2)
            min_height = min(self.height_threshold_min, height * 0.2)
            max_aspect_ratio = 3.0  # 宽高比限制
        else:
            min_area = image_area * self.area_min_ratio
            min_width = max(self.width_threshold_min, width * 0.1)
            min_height = max(self.height_threshold_min, height * 0.1)
            max_aspect_ratio = 4.0
        
        # 筛选有效轮廓
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # 计算轮廓的填充率和宽高比
            rect_area = w * h
            filling_ratio = area / rect_area if rect_area > 0 else 0
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
            
            # 调试信息
            if self.debug and rect_area > image_area * 0.01:
                print(f"轮廓: {x},{y},{w},{h}, 面积={area}, 填充率={filling_ratio:.2f}, 宽高比={aspect_ratio:.2f}")
            
            # 使用多条件筛选
            if (w >= min_width and h >= min_height and 
                area >= min_area and 
                aspect_ratio <= max_aspect_ratio):
                valid_contours.append((x, y, w, h))
        
        # 如果没有检测到有效轮廓，尝试使用整个图像边缘
        if not valid_contours and np.sum(edges) > 0:
            if is_small_image:
                # 对于小图像，假设整个图像是车辆
                border = 5  # 边界宽度
                x, y = border, border
                w, h = width - 2*border, height - 2*border
                valid_contours.append((x, y, w, h))
                print("使用整体检测法识别车辆")
        
        return valid_contours


def process_image(image_path, output_path=None, debug=False):
    """处理单个图像"""
    detector = SimpleDetector()
    result = detector.detect(image_path, output_path, debug)
    
    if result:
        print("\n=== 检测完成 ===")
        print(f"检测到 {len(result['contours'])} 个车辆轮廓")
        return True
    return False


def process_directory(directory, debug=False):
    """处理目录中的所有图像"""
    image_files = [f for f in os.listdir(directory) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')) and 
                  not any(keyword in f.lower() for keyword in ['binary', 'result', 'detected'])]
    
    if not image_files:
        print("未找到图像文件")
        return
    
    print(f"发现 {len(image_files)} 个图像文件")
    detector = SimpleDetector()
    
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        print(f"\n{'='*50}")
        print(f"处理图像: {image_file}")
        print(f"{'='*50}")
        
        detector.detect(image_path, debug=debug)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简化版车辆轮廓检测器')
    parser.add_argument('input', help='输入图像或目录')
    parser.add_argument('-o', '--output', help='输出图像路径')
    parser.add_argument('-d', '--debug', action='store_true', help='启用调试模式')
    parser.add_argument('-a', '--all', action='store_true', help='处理目录下所有图像')
    
    args = parser.parse_args()
    
    if args.all or os.path.isdir(args.input):
        # 处理目录
        directory = args.input if os.path.isdir(args.input) else os.path.dirname(args.input) or '.'
        process_directory(directory, args.debug)
    else:
        # 处理单张图像
        process_image(args.input, args.output, args.debug)


if __name__ == "__main__":
    main()
