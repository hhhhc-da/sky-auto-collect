# coding=utf-8
import os
os.environ["QT_SCALE_FACTOR"] = "1.0"
os.environ['OMP_NUM_THREADS'] = '1'

import time
from PySide6.QtCore import QThread
import cv2
import pyautogui

import pydirectinput
pydirectinput.FAILSAFE = False

import numpy as np
import matplotlib.pyplot as plt
import random

class BaseThread(QThread):
    def __init__(self, sigma=0.07):
        super().__init__()
        
        screenshot = np.array(pyautogui.screenshot())
        self.height, self.width = int(screenshot.shape[0]), int(screenshot.shape[1])
        
        self.sigma = sigma
        self.page = 0

    def gauss_sleep(self, seconds:float=0.6, min_seconds:float=0.1) -> None:
        """暂停指定的秒数"""
        time.sleep(max(min_seconds, random.gauss(seconds, self.sigma)))
        
    def mouse_clear(self) -> None:
        """清除鼠标位置"""
        pyautogui.moveTo(1, 1)
        self.gauss_sleep(0.5)
        
    def screenshot(self) -> np.ndarray:
        """获取当前屏幕截图"""
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        return frame
    
    def check_text(self) -> bool:
        """检测左下角是否存在我们期待的文字, 如果有返回 True, 否则返回 False"""
        frame = self.screenshot()
        gray = cv2.cvtColor(cv2.resize(frame, (1920, 1080)), cv2.COLOR_RGB2GRAY)
        text = self.detector.ocr_detector(gray=gray)
        pf = self.detector.re_keyword_detector([text])
        return bool(pf['星盘页'].values[0])

    def goto_page(self, page=0) -> bool:
        """跳转到指定的页数"""
        pyautogui.moveTo(1, 1)  # 移动鼠标到屏幕左上角
        self.gauss_sleep(0.5)  # 等待鼠标移动完成
        
        timeout = 50
        while timeout > 0:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            self.detector.update_image(frame)  # 更新检测器的图像数据
            
            pdata = None
            timeout_detector = 3  # 每次检测最多尝试 3 次
            while timeout_detector > 0:
                try:
                    pdata, _ = self.detector.multi_detector(plot=False, show=False)
                    if pdata is not None:
                        break
                except Exception as e:
                    print("捕捉到 Exception:", e)
                    timeout_detector -= 1
                   
            if pdata['code'] != 1:
                timeout -= 1
                
                # 不停向左检测图片
                pydirectinput.keyDown('z')
                self.gauss_sleep(0.6)
                pydirectinput.keyUp('z')
                self.gauss_sleep(1.2)
                continue
            
            break
        
        if timeout <= 0:
            print("未检测到添加好友页")
            return False

        # 跳转到指定页面
        for _ in range(page):
            pydirectinput.keyDown('c')
            self.gauss_sleep(0.1)
            pydirectinput.keyUp('c')
            self.gauss_sleep(0.3)
        
        return True
    
    def next_page(self) -> None:
        """向右检测图片"""
        pydirectinput.keyDown('c')
        self.gauss_sleep(0.6)
        pydirectinput.keyUp('c')
        self.gauss_sleep(3)
        
        # 页数加一表示现在在那页否则会死循环
        self.page += 1
        
    def check_page(self, plot=False, show=False) -> tuple:
        """检查当前页面是否是正常星盘页"""
        pyautogui.moveTo(1, 1)  # 移动鼠标到屏幕左上角
        self.gauss_sleep(0.5)  # 等待鼠标移动完成
            
        pdata, img = None, None
        timeout_detector = 3  # 每次检测最多尝试 3 次
        while timeout_detector > 0:
            try:
                pdata, img = self.detector.multi_detector(plot=plot, show=show)
                if pdata is not None and pdata['code'] in [0, 1]:
                    break
            except Exception as e:
                print(f"检测失败: {e}")
                timeout_detector -= 1
                continue
            
            timeout_detector -= 1
            self.gauss_sleep(0.3) # 等待下一次检测, 每约 0.3s 检测一次
            
        if pdata['code'] == 1:
            print("检测到添加好友页，退出程序")
            pydirectinput.keyDown('esc')
            self.gauss_sleep(0.1)
            pydirectinput.keyUp('esc')
            return True, None, None
            
        if img is not None:
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(os.path.join('runs', 'predict', f'page{self.page}.png'))
            plt.close()
            
        return False, pdata, img