import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def enhance_underwater_edges(image_path, output_dir="./results/"):
    # 1. 准备环境并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 2. 安全读取图像
    img_data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图像读取失败")
    
    # 3. 预处理 - 多步骤增强
    ## 3.1 色差校正
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    l = clahe.apply(l)  # 强力增强亮度通道
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_color = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    ## 3.2 盲去卷积去模糊
    gray = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2GRAY)
    psf = np.ones((7, 7)) / 49  # 点扩散函数估计
    deblurred = cv2.filter2D(gray, -1, psf)
    
    # 4. 多尺度边缘检测融合
    ## 4.1 小尺度边缘 - 细节特征
    edges_fine = cv2.Laplacian(deblurred, cv2.CV_64F)
    edges_fine = np.uint8(np.abs(edges_fine))
    
    ## 4.2 中尺度边缘 - 主要轮廓
    denoised = cv2.bilateralFilter(deblurred, 9, 75, 75)
    edges_medium = cv2.Canny(denoised, 15, 40)
    
    ## 4.3 大尺度边缘 - 主要结构
    blur_gaussian = cv2.GaussianBlur(deblurred, (15, 15), 0)
    edges_large = cv2.Canny(blur_gaussian, 10, 20)
    
    ## 4.4 融合多尺度结果
    combined = cv2.addWeighted(edges_fine, 0.3, edges_medium, 0.5, 0)
    final_edges = cv2.addWeighted(combined, 0.7, edges_large, 0.3, 0)
    _, final_edges = cv2.threshold(final_edges, 30, 255, cv2.THRESH_BINARY)
    
    # 5. 后处理优化
    final_edges = cv2.morphologyEx(final_edges, cv2.MORPH_CLOSE, 
                                 np.ones((3, 3), np.uint8), iterations=2)
    final_edges = cv2.dilate(final_edges, np.ones((2, 2), np.uint8), iterations=1)
    
    # 6. 结果输出和可视化
    ## 原始图像叠加结果
    output_image = cv2.addWeighted(img, 0.7, 
                                  cv2.cvtColor(final_edges, cv2.COLOR_GRAY2BGR), 
                                  0.3, 0)
    
    ## 纯边缘图
    edge_result = cv2.cvtColor(final_edges, cv2.COLOR_GRAY2BGR)
    
    ## 保存所有结果
    cv2.imwrite(os.path.join(output_dir, "00_original.jpg"), img)
    cv2.imwrite(os.path.join(output_dir, "01_enhanced.jpg"), enhanced_color)
    cv2.imwrite(os.path.join(output_dir, "02_deblurred.jpg"), deblurred)
    cv2.imwrite(os.path.join(output_dir, "03_fine_edges.jpg"), edges_fine)
    cv2.imwrite(os.path.join(output_dir, "04_medium_edges.jpg"), edges_medium)
    cv2.imwrite(os.path.join(output_dir, "05_large_edges.jpg"), edges_large)
    cv2.imwrite(os.path.join(output_dir, "06_final_edges.jpg"), final_edges)
    cv2.imwrite(os.path.join(output_dir, "07_output_overlay.jpg"), output_image)
    cv2.imwrite(os.path.join(output_dir, "08_pure_edges.jpg"), edge_result)
    
    return final_edges

# 使用示例
if __name__ == "__main__":
    # 处理您提供的水下图像
    result = enhance_underwater_edges(
        r'C:\Users\MFF\Desktop\eph\000027.jpg', 
        output_dir=r'C:\Users\MFF\Desktop\eph\processed_results'
    )
    
    print(f"处理完成! 结果已保存到指定目录")
