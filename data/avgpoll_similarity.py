#coding=utf-8
'''
用于选择出 source 中不同的图片，使用平均池化对图片进行降维，之后对比相似度，根据帧间的相似度进行筛选。
'''
import cv2
import os
import torch
import torch.nn.functional as F
import numpy as np

count = 0
previous_image = None

def select_frames(source_folder, output_folder, threshold=0.9):
    """
    从源文件夹中选择不同的图片帧，并将其保存到输出文件夹
    使用平均池化对图片进行降维，之后对比相似度，根据帧间的相似度进行筛选。

    参数:
        source_folder (str): 输入源图片的文件夹路径
        output_folder (str): 输出选择的图片的文件夹路径
        threshold (float): 相似度阈值，默认值为0.9
    """
    global count, previous_image
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.png'))]
    
    if not image_files:
        print("没有找到任何图片文件。")
        return
    
    for image_file in image_files:
        image_path = os.path.join(source_folder, image_file)
        current_image = cv2.imread(image_path)
        
        if current_image is None:
            print(f"无法读取图片: {image_path}")
            continue
        
        # 转换为灰度图并进行平均池化
        reshape_image = cv2.resize(current_image, (64, 64))  # 调整大小以减少计算量
        gray_image = cv2.cvtColor(reshape_image, cv2.COLOR_BGR2GRAY)
        pooled_image = F.avg_pool2d(torch.tensor(gray_image).unsqueeze(0).unsqueeze(0).float(), kernel_size=8, stride=8).squeeze().numpy()
        
        # 太黑的图片也不要
        if np.mean(gray_image) < 40:
            continue
        
        if previous_image is not None:
            # 计算余弦相似度
            similarity = F.cosine_similarity(torch.tensor(pooled_image.flatten()).unsqueeze(0), torch.tensor(previous_image.flatten()).unsqueeze(0)).item()
            
            if similarity < threshold:
                # 保存当前帧到输出文件夹
                output_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
                cv2.imwrite(output_path, current_image)
                count += 1
        
        previous_image = pooled_image
    
    print(f"成功选择 {count} 帧到 {output_folder} 文件夹")
    
if __name__ == "__main__":
    source_folders = os.listdir(os.path.join('source', 'frames'))
    output_folder = os.path.join('diff', 'images')
    
    for source_folder in source_folders:
        print(f"正在处理源文件夹: {source_folder}")
        source_folder = os.path.join('source', 'frames', source_folder)
        output_folder = os.path.join('diff', 'images')
        
        if not os.path.exists(source_folder):
            print(f"源文件夹 {source_folder} 不存在，跳过。")
            continue
        
        select_frames(source_folder, output_folder, threshold=0.8)