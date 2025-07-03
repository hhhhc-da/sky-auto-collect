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

    # virtual 虚函数
    def goto_page(self, page=0):
        '''
        跳转到指定页面的接口
        '''
        print("goto_page is not implemented in BaseThread, please override it in subclass.")
        
    # virtual 虚函数
    def check_page(self):
        '''
        检测当前页面是否是目标页面
        返回 True 表示是目标页面，False 表示不是
        '''
        print("check_page is not implemented in BaseThread, please override it in subclass.")
        return False
    
    def next_page(self) -> None:
        '''
        按 C 向右检测图片, 这个可以直接写出来
        '''
        pydirectinput.keyDown('c')
        self.gauss_sleep(0.6)
        pydirectinput.keyUp('c')
        self.gauss_sleep(3)
        
        # 页数加一表示现在在那页否则会死循环
        self.page += 1
        
    def faster_next_page(self) -> None:
        '''
        按 C 向右检测图片, 这个可以直接写出来
        速度更快的版本
        '''
        pydirectinput.keyDown('c')
        self.gauss_sleep(0.1)
        pydirectinput.keyUp('c')
        self.gauss_sleep(0.3)
        
        # 页数加一表示现在在那页否则会死循环
        self.page += 1
        
    def last_page(self) -> None:
        '''
        按 Z 向左检测图片, 这个可以直接写出来
        '''
        pydirectinput.keyDown('z')
        self.gauss_sleep(0.6)
        pydirectinput.keyUp('z')
        self.gauss_sleep(1.2)
