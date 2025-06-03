import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import re
import json
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
from scipy import signal
import ddddocr

# 全局配置
# sns.set_theme(style="darkgrid", palette="pastel")
ocr = ddddocr.DdddOcr(show_ad=False)
size_info = []

# 二值化检测器
def threhold_detector(gray, image, plot=False, show=False):
    '''
    使用灰度化和高斯模糊确定目标点, FFT 和 Canny 算子都太不稳定了
    '''
    # 对灰度图进行二值化
    _, binary_image = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
    # 高斯平滑后再采样
    blured_image = cv2.blur(binary_image, (11,11))
    # 重新二值化采样
    _, rebinary_image = cv2.threshold(blured_image, 155, 255, cv2.THRESH_BINARY)
    
    # 掩膜
    height, width = gray.shape
    border_ratio = 0.1
    mask = np.zeros_like(gray, dtype=np.uint8)
    border_x = int(width * border_ratio)
    border_y = int(height * border_ratio)
    inner_width = width - 2 * border_x
    inner_height = height - 2 * border_y

    cv2.rectangle(mask, (border_x, border_y), (border_x + inner_width, border_y + inner_height), 255, -1)
    masked_rebinary_image = cv2.bitwise_and(mask, rebinary_image, mask=mask)
    
    # 提取大范围亮色特征 (轮廓分析)
    targets = []
    origin_image = None
    if plot:
        origin_image = image.copy()
        cv2.rectangle(origin_image, (border_x, border_y), (border_x + inner_width, border_y + inner_height), 255, 2)
    
    contours, _ = cv2.findContours(masked_rebinary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2
        targets.append((center_x, center_y, w, h))
        
        if plot:
            cv2.rectangle(origin_image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绿色矩形
            cv2.circle(origin_image, (center_x, center_y), 5, (255, 0, 0), -1)  # 红色点

    if show:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.imshow(origin_image)
        axes.set_title("Analyzed Image")
        plt.show()
    
    return targets, origin_image

# 圆形检测器
def hough_detector(gray, image, plot=False, show=False):
    '''
    使用霍夫变换检测圆形目标, 主要提供我们的识别区域
    '''
    min_radius=1
    max_radius=35

    # 边界去除
    height, width = gray.shape
    border_ratio = 0.1
    mask = np.zeros_like(gray, dtype=np.uint8)
    border_x = int(width * border_ratio)
    border_y = int(height * border_ratio)
    inner_width = width - 2 * border_x
    inner_height = height - 2 * border_y

    cv2.rectangle(mask, (border_x, border_y), (border_x + inner_width, border_y + inner_height), 255, -1)
    masked_edges = cv2.bitwise_and(mask, gray, mask=mask)

    # 使用霍夫圆变换检测圆形
    circles = cv2.HoughCircles(
        masked_edges,               # 输入图像（边缘图）
        cv2.HOUGH_GRADIENT,         # 检测方法（梯度法）
        dp=1,                       # 累加器图像的分辨率与原图之比
        minDist=50,                 # 检测到的圆的圆心之间的最小距离
        param1=50,                  # Canny边缘检测器的高阈值
        param2=25,                  # 累加器阈值（越小检测到的圆越多）
        minRadius=min_radius,       # 最小圆半径
        maxRadius=max_radius        # 最大圆半径
    )

    # 复制原图用于绘制结果
    origin_image = None
    if plot:
        origin_image = image.copy()
        cv2.rectangle(origin_image, (border_x, border_y), (border_x + inner_width, border_y + inner_height), 255, 2)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if plot:
            for (x, y, r) in circles:
                cv2.circle(origin_image, (x, y), r, (0, 255, 0), 4)
        
    if show:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.imshow(origin_image, cmap="gray")
        axes.set_title("Masked Hough transform Image")
        plt.show()
        
    return circles, origin_image

def gradient_detector(arr):
    """
    一阶离散梯度寻找切割平面
    """
    global size_info
    # 数据太少直接使用旧值, 所以我们检测的时候一定要从数据多的一侧开始检测
    if arr.shape[0] < 4:
        # 直接返回超平面平均值
        if len(size_info) > 0:
            return False, np.mean(size_info), None
        else:
            # 吓人的三无数据
            return False, None, None
        
    abs_gradients = np.abs(np.diff(arr))
    idx = np.argmax(abs_gradients)

    # 切割超平面
    split_surface = np.mean(arr[idx:idx+2])
    # 平均梯度统计
    explore = np.ones((abs_gradients.shape[0]+2,))*np.mean(abs_gradients)
    explore[1:-1] = abs_gradients
    output = signal.medfilt(explore, kernel_size=3) # 等宽中值滤波, 两边补平均值
    mean_gradients = np.mean(output)

    # 如果最大梯度与平均梯度的变化不足 50%
    if abs(split_surface-mean_gradients)/mean_gradients < 0.5:
        return False, split_surface, mean_gradients
    else:
        size_info.append(split_surface)
        return True, split_surface, mean_gradients

# 双因素检测器
def tfa_detector(gray, image, plot=False, show=False):
    '''
    TWO FACTOR 双因素检测器, 结合阈值检测器和霍夫变换检测器
    '''
    targets, _ = threhold_detector(gray, image)
    circles, _ = hough_detector(gray, image)

    if targets is None or circles is None:
        return [], None

    target_array = np.array(targets)       # 形状: (n_targets, 4)
    circles_array = np.array(circles)      # 形状: (n_circles, 3)

    target_centers = target_array[:, :2]   # 形状: (n_targets, 2)
    circle_centers = circles_array[:, :2]  # 形状: (n_circles, 2)

    circle_radius = circles_array[:, 2].reshape(1, -1)  # 形状: (1, n_circles)
    
    # 计算所有目标中心与所有圆心之间的距离矩阵
    # 形状: (n_targets, n_circles) - (1, n_circles)
    radius_bitmap = np.sqrt(np.sum((target_centers[:, np.newaxis, :] - circle_centers[np.newaxis, :, :])**2, axis=2)) - circle_radius    
    # 提取前两个维度的负数索引（对应targets和circles的索引）
    target_indices, circle_indices = np.where(radius_bitmap < 0)
    
    tfa_targets = []
    for target_idx, circle_idx in zip(target_indices, circle_indices):
        target_data = tuple(targets[target_idx])
        circle_data = tuple(circles[circle_idx])
        tfa_targets.append([target_data, circle_data])

    # 校验 Radius 值获取切割平面, 但不一定要切割
    r_list = np.array(sorted([target[1][2] for target in tfa_targets]))
    flag, split_surface, mean_gradients = gradient_detector(r_list)
    # print("离散梯度切割超平面{}".format("成功, 开始分类任务" if flag==True else "失败, 将所有内容标注为待办"))

    # 如果 flag 为 True 说明切割成功, 可喜可贺的度过了困难
    origin_image = None
    
    if plot:
        origin_image = image.copy()
        
        for i, ((x, y, w, h), (cx, cy, r)) in enumerate(tfa_targets):
            # 我们旧的 xywh 是针对于二值化最后的框选, 而我们要做的是对圆圈整个进行选区
            
            if flag == True and split_surface is not None:
                # 为什么是大的留下呢, 因为大的有圆圈, 小的没有!!!
                if r > split_surface:
                    cv2.rectangle(origin_image, (cx-r, cy-r), (cx+r, cy+r), (255, 0, 0), 2)
                    tfa_targets[i].append("I")
                else:
                    cv2.rectangle(origin_image, (cx-r, cy-r), (cx+r, cy+r), (0, 255, 0), 2)
                    tfa_targets[i].append("U")
            else:
                cv2.rectangle(origin_image, (cx-r, cy-r), (cx+r, cy+r), (255, 0, 0), 2)
                tfa_targets[i].append("I")
        
    if show:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.imshow(origin_image, cmap="gray")
        axes.set_title("Masked Hough transforAm Image")
        plt.show()
        
    return tfa_targets, origin_image

def ocr_detector(gray, border_ratio=0.1):
    '''
    ddddocr 图像文字识别
    '''
    global ocr
    height, width = gray.shape
    mask = np.zeros_like(gray, dtype=np.uint8)
    border_x = int(width * border_ratio)
    border_y = int(height * border_ratio)
    
    text_area = gray[height-border_y:, :border_x]    
    _, encoded_img = cv2.imencode('.png', text_area)
    img_bytes = encoded_img.tobytes()

    # 只能检测编码字节流
    result = ocr.classification(img_bytes)
    return result

def re_keyword_detector(texts):
    """
    ASCII字符清洗检测子串, 子串是提前规定好的
    """
    patterns = [r'添[\x00-\x7F]{0,3}加[\x00-\x7F]{0,3}好[\x00-\x7F]{0,3}友', r'好[\x00-\x7F]{0,3}友', r'挚[\x00-\x7F]{0,3}友']
    dataframes = {
        '识别文本': [],
        '星盘页': [],
        '添加好友': [],
        '好友': [],
        '挚友': [],
    }

    for text in texts:
        matchs = list([bool(re.search(pattern, text)) for pattern in patterns])
        dataframes['星盘页'].append(True in matchs)
        dataframes['添加好友'].append(matchs[0] == True)
        dataframes['好友'].append(matchs[1] == True)
        dataframes['挚友'].append(matchs[2] == True)
        dataframes['识别文本'].append(text)
        
    return pd.DataFrame(dataframes)

def multi_detector(gray, image, border_ratio=0.1, plot=False, show=False):
    '''
    多重鉴别器, 用于检测是否在星盘页并且判断类型
    '''
    text = ocr_detector(gray)
    pf = re_keyword_detector([text])

    if not bool(pf['星盘页'].values[0]):
        # print("不在星盘页, 不进行后续检测")
        if show:
            fig, axes = plt.subplots(1, 1, figsize=(15, 5))
            axes.imshow(image)
            axes.set_title("Useless Image")
            axes.axis('off')
            plt.show()
        return {"code":-1, "info":"不在星盘页"}, None

    if bool(pf['添加好友'].values[0]):
        # print("当前页面为添加好友页")
        if show:
            fig, axes = plt.subplots(1, 1, figsize=(15, 5))
            axes.imshow(image)
            axes.set_title("Make Friends Image")
            axes.axis('off')
            plt.show()
        return {"code":1, "info":"本页面为添加好友页"}, None

    if bool(pf['好友'].values[0]) or bool(pf['挚友'].values[0]):
        # print("检测为有效的星盘页")
        tfa_targets, img =  tfa_detector(gray, image, plot=plot, show=False)
        if show:
            r_list = np.array(sorted([target[1][2] for target in tfa_targets]))
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            plt.grid(True, linestyle="--", alpha=1)
            sns.lineplot(x="Count", y="Radius", label="Radius", data={"Count":np.arange(1, r_list.shape[0]+1, 1), "Radius": r_list}, ax=axes[0])
            axes[0].set_title("Radius Plot Image")
            axes[1].imshow(img)
            axes[1].set_title("TFA Detector Result")
            axes[1].axis('off')
            plt.grid(False, linestyle="--", alpha=1)
            plt.tight_layout()
            plt.show()
        return {"code":0, "info":"识别成功", "tfa_targets":tfa_targets}, img
    return {"code":-2, "info":"无法预知的错误!"}, None

if __name__ == '__main__':
    # # 静态图片测试
    # for filename in os.listdir(os.path.join("source", "data")):
    #     if filename.endswith(".png") or filename.endswith(".jpg"):
    #         print(f"\nProcessing {filename}...")
            
    #         image = cv2.imread(os.path.join("source", "data", filename))
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
    #         json_data, img = multi_detector(gray, image, plot=True, show=False)
    #     else:
    #         print(f"Skipping {filename}, not an image file.")
            
    # 视频测试
    cap = cv2.VideoCapture(os.path.join("source", "valid.mp4"))
    
    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义编码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器为MP4
    writer = cv2.VideoWriter(os.path.join('runs', 'out.mp4'), fourcc, fps, (width, height))
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()
    
    print("开始处理视频文件")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        json_data, img = multi_detector(gray, image, plot=True, show=False)
        
        # 写入新的视频文件
        if img is not None:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)
            
    cap.release()
    writer.release()
    print("Video processing complete. Output saved to 'runs/out.mp4'.")