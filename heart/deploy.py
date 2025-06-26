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
from sklearn.metrics import classification_report, confusion_matrix
from scipy import signal
import ddddocr
import torch
from torchvision.models import resnet18, ResNet18_Weights
from copy import copy
import pyautogui
from datetime import datetime
import time
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import math
from scipy.ndimage import median_filter

# 全局配置
class NanokaDetector():
    def __init__(self, image=None, circle_radius=20):
        super(NanokaDetector, self).__init__()

        if image != None:
            if image.shape[0] == 1080 and image.shape[1] == 1920:
                self.image = image
            else:
                self.image = cv2.resize(image, (1080, 1920))
                
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = None
            self.gray = None

        self.circle_radius = circle_radius
        self.ocr = ddddocr.DdddOcr(show_ad=False)

    def update_image(self, image):
        '''
        更新使用的图片, 我们可以直接调取这个图片进行分析
        '''
        if image.shape[0] == 1080 and image.shape[1] == 1920:
            self.image = image
        else:
            self.image = cv2.resize(image, (1080, 1920))
            
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    # 红色爱心检测器
    def hue_detector(self, hue_threhold=15):
        '''
        检测 HSV 图片属性, 将爱心颜色取出并抑制椒盐噪声(来自星屑)
        '''
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        hue_ranges = []
        
        # 计算色相环, 其他阈值卡死为 30
        if hue_threhold > 5:
            hue_ranges.append([185-hue_threhold, 179])
            hue_ranges.append([0, hue_threhold+4])
        else:
            hue_ranges.append([5-hue_threhold, hue_threhold+4])
        
        height, width = hsv_image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        if hue_threhold > 5:
            hue_masks = [cv2.inRange(hsv_image, np.array([lower_hue, 100, 210]), np.array([upper_hue, 160, 270]))for (lower_hue, upper_hue) in hue_ranges]
            for hue_mask in hue_masks:
                mask = cv2.bitwise_or(mask, hue_mask)

        # 中值滤波去掉椒盐噪声
        heart_mask = cv2.medianBlur(cv2.medianBlur(mask, 11), 7)
        return heart_mask

    def hearts_detector(self, plot=False, show=False):
        '''
        用于将爱心图片位置计算出来
        '''
        heart_mask = self.hue_detector()

        # 掩膜
        height, width = self.gray.shape
        border_ratio = 0.1
        mask = np.zeros_like(self.gray, dtype=np.uint8)
        border_x = int(width * border_ratio)
        border_y = int(height * border_ratio)
        inner_width = width - 2 * border_x
        inner_height = height - 2 * border_y
        cv2.rectangle(mask, (border_x, border_y), (border_x + inner_width, border_y + inner_height), 255, -1)
        heart_mask = cv2.bitwise_and(mask, heart_mask, mask=mask)

        origin_image = None
        if plot:
            origin_image = self.image.copy()
            # 绘制掩膜
            cv2.rectangle(origin_image, (border_x, border_y), (border_x + inner_width, border_y + inner_height), 255, 2)
        
        targets = []
        contours, _ = cv2.findContours(heart_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2
            targets.append((center_x, center_y, w, h))
            
            if plot:
                cv2.rectangle(origin_image, (x-w, y-h), (x+2*w, y+2*h), (0, 255, 0), 2)  # 绿色矩形
                cv2.circle(origin_image, (center_x, center_y), 5, (255, 0, 0), -1)  # 红色点
                
        if show:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes.imshow(origin_image)
            axes.set_title("Hearts Image")
            plt.show()
                
        return targets, origin_image
        
        
    # 亮点综合检测器
    def threhold_detector(self, plot=False, show=False):
        '''
        利用检测点的圆形高亮特性确定目标点
        '''
        # 一顿操作最后只剩下需要的信息
        _, binary_image = cv2.threshold(self.gray, 155, 255, cv2.THRESH_BINARY)
        blured_image = cv2.blur(binary_image, (9,9))
        median_image = cv2.medianBlur(blured_image, 7)
        _, rebinary_image = cv2.threshold(median_image, 155, 255, cv2.THRESH_BINARY)
        
        # 掩膜
        height, width = self.gray.shape
        border_ratio = 0.1
        mask = np.zeros_like(self.gray, dtype=np.uint8)
        border_x = int(width * border_ratio)
        border_y = int(height * border_ratio)
        inner_width = width - 2 * border_x
        inner_height = height - 2 * border_y
        cv2.rectangle(mask, (border_x, border_y), (border_x + inner_width, border_y + inner_height), 255, -1)
        masked_rebinary_image = cv2.bitwise_and(mask, rebinary_image, mask=mask)
        
        # 提取大范围亮色特征 (轮廓分析)
        targets, anti_targets = [], []
        origin_image = None
        if plot:
            origin_image = self.image.copy()
            # 绘制掩膜
            cv2.rectangle(origin_image, (border_x, border_y), (border_x + inner_width, border_y + inner_height), 255, 2)
    
        # 抓住所有需要的点特征
        contours, _ = cv2.findContours(masked_rebinary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2
            targets.append((center_x, center_y, w, h))
            
            if plot:
                cv2.rectangle(origin_image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绿色矩形
                cv2.circle(origin_image, (center_x, center_y), 5, (255, 0, 0), -1)  # 红色点

        # 消除不需要的点特征
        masked_median_image = cv2.medianBlur(cv2.medianBlur(masked_rebinary_image, 11), 11)
        acontours, _ = cv2.findContours(masked_median_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in acontours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2
            anti_targets.append((center_x, center_y, w, h))
            
            if plot:
                cv2.rectangle(origin_image, (x-5, y-5), (x+w+5, y+h+5), (255, 0, 0), 2)  # 红色矩形
    
        if show:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes.imshow(origin_image)
            axes.set_title("Analyzed Image")
            plt.show()
        
        return np.array(targets, dtype=np.float32), np.array(anti_targets, dtype=np.float32), origin_image

    def ocr_detector(self, gray=None, border_ratio=0.1):
        '''
        ddddocr 图像文字识别
        '''
        result = None
        if gray is None:
            height, width = self.gray.shape
            border_x = int(width * border_ratio)
            border_y = int(height * border_ratio)
            
            text_area = self.gray[height-border_y:, :border_x]    
            _, encoded_img = cv2.imencode('.png', text_area)
            img_bytes = encoded_img.tobytes()
        
            # 只能检测编码字节流
            result = self.ocr.classification(img_bytes)
        else:
            height, width = gray.shape
            
            border_x = int(width * border_ratio)
            border_y = int(height * border_ratio)
            
            text_area_2 = gray[height-border_y:, :border_x]    
            
            _, encoded_img_2 = cv2.imencode('.png', text_area_2)
            img_bytes = encoded_img_2.tobytes()
        
            # 只能检测编码字节流
            result = self.ocr.classification(img_bytes)
        return result

    def re_keyword_detector(self, texts):
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

    def image_spiltor(self):
        '''
        用于将数据切分成我们想要的形式之后进行分类任务就可以了
        '''
        targets, anti_targets, plot_image = self.threhold_detector(plot=False, show=False)

        diff = np.expand_dims(targets[:,:2], axis=1) - np.expand_dims(anti_targets[:,:2], axis=0)
        distance = np.sqrt(np.sum(diff*diff, axis=2))
        point = np.where(distance < 15)[0]
        post = copy(targets[point])
        pre = [i for i in targets.tolist() if i not in post]

        # 不同的类别进行不同的切割方式, pre 是需要进一步处理的内容, post是确定的很大的不需要处理的内容
        pre_scaler, post_scaler = 5, 2
        pre_sub_gray = [self.gray[int(y-max(w,h)*pre_scaler):int(y+max(w, h)*pre_scaler), int(x-max(w, h)*pre_scaler):int(x+max(w, h)*pre_scaler)] for x, y, w, h in pre]
        post_sub_gray = [self.gray[int(y-max(w,h)*post_scaler):int(y+max(w, h)*post_scaler), int(x-max(w, h)*post_scaler):int(x+max(w, h)*post_scaler)] for x, y, w, h in post]
        return pre_sub_gray, post_sub_gray, pre, post

    def light_circle(self, image, radius=12):
        '''
        检测圆形掩码内是都不为 0
        '''
        x, y = 112, 112
        height, width = image.shape
        yy, xx = np.ogrid[:height, :width]
        distance_squared = (xx - x) ** 2 + (yy - y) ** 2
        circle_mask = distance_squared <= radius ** 2
        circle_pixels = image[circle_mask]
        return np.all(circle_pixels != 0)

    def analyze_star(self, images, bound=35):
        '''
        建立四个矩形选区检测是否存在星屑, 如果完全不存在那就是没有星屑, 因为文字只会最多占 3 选区, 这没测出来也是很倒霉了
        '''
        rotation_matrix = cv2.getRotationMatrix2D((112,112), 45, 1.0)
        resize_image = [cv2.threshold(cv2.resize(gray, (224, 224)), 50, 255, cv2.THRESH_BINARY)[1] for gray in images]
        route_resize_image = [cv2.warpAffine(gray, rotation_matrix, (224,224)) for gray in resize_image]
        diff = [np.abs(np.array(r1, dtype=np.float32) - np.array(r2, dtype=np.float32)) for r1, r2 in zip(route_resize_image, resize_image)]
        diff = [cv2.threshold(cv2.blur(gray, (5,5)), 150, 255, cv2.THRESH_BINARY)[1] for gray in diff]
        
        split_post = [[img_post[bound:112-bound, bound:112-bound], img_post[bound:112-bound, 112+bound:224-bound], img_post[112+bound:224-bound, bound:112-bound], img_post[112+bound:224-bound, 112+bound:224-bound]] for img_post in diff]
        labels = []
        for group in split_post:
            all_zero = False
            for img in group:
                if np.all(img == 0):  
                    all_zero = True
                    break
            labels.append(1 if all_zero else 0)
        return labels

    def analyze_octagram(self, images):
        '''
        检测图形中是否出现八角星星，如果没有出现我们就认为没有，如果过于混乱我们也不考虑, 我们可以校验圆心处的同心圆是否为单色
        '''
        if len(images) == 0:
            # 没有任何有效图像, 草这是为什么?
            return [], []
            
        # 旋转45度计算差值
        rotation_matrix = cv2.getRotationMatrix2D((112,112), 45, 1.0)
        resize_images = [cv2.threshold(cv2.resize(gray, (224, 224)), 50, 255, cv2.THRESH_BINARY)[1] for gray in images]
        route_resize_images = [cv2.warpAffine(gray, rotation_matrix, (224,224)) for gray in resize_images]
        diff_images = [np.abs(np.array(r1, dtype=np.float32) - np.array(r2, dtype=np.float32)) for r1, r2 in zip(route_resize_images, resize_images)]
        diff_images = [cv2.threshold(cv2.blur(gray, (5,5)), 150, 255, cv2.THRESH_BINARY)[1] for gray in diff_images]

        # 使图像旋转八次并且堆叠在一起
        feature_map = [diff_images]
        for i in range(7):
            feature_map.append([cv2.warpAffine(gray, rotation_matrix, (224,224)) for gray in feature_map[-1]])
        stacked_features = np.array(feature_map)
        image_result = np.ones((len(diff_images), 224, 224), dtype=np.uint8)
        for i in range(8):
            image_result = image_result * (stacked_features[i] > 0)
        image_result = image_result.astype(np.uint8)

        # 计算四条准线
        feature_list = []
        for gray in image_result:
            x = np.arange(224)
            y = np.arange(224)
            xx, yy = np.meshgrid(x, y)
            
            lines = [gray[112, :], gray[:, 112], gray[yy == xx], gray[yy == 223 - xx]]
            feature_list.append(np.array(lines))

        # 计算反向掩膜, 一般掩膜内不应该存在三个及以上圆环
        analyze_list = []
        for i in range(len(feature_list)):
            data = np.average(np.array([(feature_list[i][j] + feature_list[i][j][::-1])/2 for j in range(feature_list[i].shape[0])]), axis=0)
            binary_data = np.where(data != 0, 1, 0)
            first_half = binary_data[:112]
            second_half = binary_data[112:][::-1]
            comparison = np.logical_and(first_half, second_half)
            processed = median_filter(comparison, size=11)
            analyze_list.append([1 if i else 0 for i in processed])

        # 计算边沿
        radius_list, octagram_detect_result = [], []
        for i, data in enumerate(analyze_list):
            # bound_pixel = 15
            # diff = np.diff(data[bound_pixel:]) # 我们假设圆环不可能出现在外边界, 因为我毕竟抓了好几倍边长呢
            # rising_edges = np.where(diff == 1)[0] + bound_pixel + 1
            # falling_edges = np.where(diff == -1)[0] + bound_pixel + 1
            
            diff = np.diff(data) # 现实总是出乎意料的, 真的分布采样到外面去了 (流汗)
            rising_edges = np.where(diff == 1)[0] + 1
            falling_edges = np.where(diff == -1)[0] + 1
        
            if len(rising_edges) != len(falling_edges):
                # fig, axes = plt.subplots(2, 5, figsize=(12, 4))
                # for i, ax in enumerate(axes.flatten()):
                #     if i < len(analyze_list):
                #         ax.plot(analyze_list[i])
                #         ax.set_title('Analyze Plot Figure{}'.format(i))
                #         plt.ylim(0, 1)
                # plt.tight_layout()
                # plt.show()
                raise RuntimeError("上升沿与下降沿不匹配")
            if len(rising_edges) >= 3: # 不应该有这么多有效边沿, 除非是大文字
                octagram_detect_result.append(0) # cls 0 就是背景色
                radius_list.append([False, -1, -2])
                continue
            if len(rising_edges) == 0:
                if self.light_circle(cv2.threshold(cv2.resize(images[i], (224, 224)), 50, 255, cv2.THRESH_BINARY)[1], radius=self.circle_radius):
                    radius_list.append([False, -1, -1])
                    octagram_detect_result.append(2) # cls 2 就是已经送了心火的, 后续还要进一步检测是否有星屑
                else:
                    radius_list.append([False, -1, -2])
                    octagram_detect_result.append(0) # 中心圆不存在就是背景图
            else:
                index = np.argmax(falling_edges - rising_edges)
                radius = 112 - (falling_edges[index] + rising_edges[index])/2
                if falling_edges[index] - rising_edges[index] > 15: # 经验值
                    if self.light_circle(cv2.threshold(cv2.resize(images[i], (224, 224)), 50, 255, cv2.THRESH_BINARY)[1], radius=self.circle_radius):
                        radius_list.append([True, radius, falling_edges[index] - rising_edges[index]])
                        octagram_detect_result.append(1) # cls 1 就是还没有送心火的, 直接点击就可以了根据星盘状态去判断
                    else:
                        radius_list.append([False, -1, -2])
                        octagram_detect_result.append(0) # 中心圆不存在就是背景图
                else:
                    radius_list.append([False, -1, -1])
                    octagram_detect_result.append(0) # 识别错成了小星光, 也属于背景的一种
                    
        return octagram_detect_result, radius_list
        
    def multi_detector(self, border_ratio=0.1, plot=False, show=False):
        '''
        多重鉴别器, 用于检测是否在星盘页并且判断类型
        '''
        text = self.ocr_detector()
        pf = self.re_keyword_detector([text])
        img = None
    
        if not bool(pf['星盘页'].values[0]):
            # print("不在星盘页, 不进行后续检测")
            if show:
                fig, axes = plt.subplots(1, 1, figsize=(15, 5))
                axes.imshow(self.image)
                axes.set_title("Useless Image")
                axes.axis('off')
                plt.show()
            return {"code":-1, "info":"不在星盘页"}, None
    
        if bool(pf['添加好友'].values[0]):
            # print("当前页面为添加好友页")
            if show:
                fig, axes = plt.subplots(1, 1, figsize=(15, 5))
                axes.imshow(self.image)
                axes.set_title("Make Friends Image")
                axes.axis('off')
                plt.show()
            return {"code":1, "info":"本页面为添加好友页"}, None
    
        if bool(pf['好友'].values[0]) or bool(pf['挚友'].values[0]):
            # print("检测为有效的星盘页")

            hearts_info, _ = self.hearts_detector()
            
            # 首先计算出我们的小点和大点
            pre_sub_gray, post_sub_gray, pre_points, post_points =  self.image_spiltor()

            # 检测出每个 post 的类型用于确定星屑, 为 0 的就是需要收心火的, 为 1 的就是不需要收心火的
            labels = self.analyze_star(post_sub_gray, bound=40)

            pre_octagram_detect_result, pre_radius_list = self.analyze_octagram(pre_sub_gray)
            post_octagram_detect_result, post_radius_list = self.analyze_octagram(post_sub_gray)

            if plot:
                # 给每一个检测结果绘制圆圈
                img = self.image.copy()
                for i,(cls,r,(cx,cy,w,h)) in enumerate(zip(pre_octagram_detect_result, pre_radius_list, pre_points)):
                    if int(cls) == 1:
                        cv2.circle(img, (int(cx), int(cy)), int(np.abs(r[2])), (255,0,0), 3) # 未送心火绘制成红色
                    if int(cls) == 2:
                        cv2.circle(img, (int(cx), int(cy)), 22, (0,255,0), 3) # 送了心火绘制成绿色
                for i,(cls,r,(cx,cy,w,h), l) in enumerate(zip(post_octagram_detect_result, post_radius_list, post_points, labels)):
                    if int(cls) == 1:
                        cv2.circle(img, (int(cx), int(cy)), int(np.abs(r[2])), (255,0,0), 3) # 未送心火绘制成红色
                    if int(cls) == 2:
                        if l == 0:
                            cv2.circle(img, (int(cx), int(cy)), 22, (255,255,0), 3) # 需要领取心火的绘制成黄色
                        if l == 1:
                            cv2.circle(img, (int(cx), int(cy)), 22, (0,255,0), 3) # 这些是不需要动的部分, 绘制成绿色
                for i, (x,y,w,h) in enumerate(hearts_info):
                    cv2.rectangle(img, (x-w, y-h), (x+w, y+h), (0,255,0), 3) # 给爱心绘制绿色矩形框

            if show:
                print("图像点为:", pre_points)
                print("\n检测结果:", pre_octagram_detect_result, "\n{}".format(pre_radius_list))
                if len(pre_sub_gray) != 0:
                    fig, axes = plt.subplots((len(pre_sub_gray)-1)//5+1, 5, figsize=(10, int(2*((len(pre_sub_gray)-1)//5+1))))
                    for i, ax in enumerate(axes.flatten()):
                        if i < len(pre_sub_gray):
                            ax.imshow(pre_sub_gray[i], cmap='gray')
                            ax.set_title('pre_sub_gray {}'.format(i))
                        ax.axis('off')
                    plt.tight_layout()
                    plt.show()
                else:
                    print("len(pre_sub_gray) == 0 所以不进行绘制")
    
                print("图像点为:", post_points)
                print("\n检测结果:", post_octagram_detect_result, "\n{}".format(post_radius_list))
                if len(post_sub_gray) != 0:
                    fig, axes = plt.subplots((len(post_sub_gray)-1)//5+1, 5, figsize=(10, int(2*((len(post_sub_gray)-1)//5+1))))
                    for i, ax in enumerate(axes.flatten()):
                        if i < len(post_sub_gray):
                            ax.imshow(post_sub_gray[i], cmap='gray')
                            ax.set_title('post_sub_gray {}'.format(i))
                        ax.axis('off')
                    plt.tight_layout()
                    plt.show()
                else:
                    print("len(post_sub_gray) == 0 所以不进行绘制")

                fig, axes = plt.subplots(1, 1, figsize=(15, 6))
                axes.imshow(img)
                axes.set_title("MultiDetector Figure")
                axes.axis('off')
                plt.tight_layout()
                plt.show()
            return {
                    "code":0, 
                    "info":"识别成功", 
                    "hearts_info":hearts_info, 
                    "pre_points":pre_points,
                    "post_points":post_points,
                    "labels":labels,
                }, img
        return {"code":-2, "info":"无法预知的错误!"}, None

if __name__ == '__main__':
    '''首先创建空的分析器, 之后我们慢慢上传图片就可以了'''
    detector = NanokaDetector()  
    
    # # 静态图片测试
    # for filename in os.listdir(os.path.join("source", "data")):
    #     if filename.endswith(".png") or filename.endswith(".jpg"):
    #         print(f"\n\nProcessing {filename}...")

    #         detector.update_image(cv2.cvtColor(cv2.imread(os.path.join("source", "data", filename)), cv2.COLOR_BGR2RGB))
    #         json_data, img = detector.multi_detector(plot=True, show=True)
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
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("开始处理视频文件")
    with tqdm(total=total_frames, desc="Process ", unit="frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            detector.update_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            img = None
            try:
                pbar.update(1)
                json_data, img = detector.multi_detector(plot=True, show=False)
            except Exception as e:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print(f"Error processing frame: {e}")
            finally:
                # 写入新的视频文件
                if img is not None:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    writer.write(img_bgr)
                
    cap.release()
    writer.release()
    print("Video processing complete. Output saved to 'runs/out.mp4'.")