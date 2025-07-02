# coding=utf-8
import os
os.environ["QT_SCALE_FACTOR"] = "1.0"
os.environ['OMP_NUM_THREADS'] = '1'

import time
from PySide6.QtCore import QThread, Signal
import cv2
import pyautogui

import pydirectinput
pydirectinput.FAILSAFE = False

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import yaml
from tqdm import tqdm

from module.base import BaseThread
from module.deploy import NanokaDetector
from module.web import WebCrawler

# 爬虫线程 
class CrawlerProgramThread(BaseThread):
    finished = Signal()
    
    def __init__(self, sigma=0.07, yaml='config.yaml'):
        super().__init__(sigma=sigma)
        
        self.crawler = WebCrawler()
        self.yaml_path = yaml

    
    def search_friend_name(self) -> None:
        '''
        寻找对应名字的好友的星星位置
        '''
        pass
    
    def goto_meet(self) -> None:
        '''
        传送到对应房间并接收爱心
        '''
    
    def run(self):
        '''
        请求爱心主线程，用来从行为上控制程序
        '''
        data = None
        with open(self.yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            file.close()
        
        pf = pd.read_excel(data['file'], sheet_name="Sheet1", names=["url"])
        crawler = WebCrawler()
        
        for epoch in range(data['episode']):
            ts = time.time()
            print(f"第 {epoch + 1} 轮取心")
            for index, row in pf.iterrows():
                target_url = row['url']
                crawler.crawl_main(target_url, valid=True)
                
                if crawler.code is not None:
                    print(f"识别到好友码: {crawler.code}")
                    
                    friend_name = "AAA送心员{}".format(data['index'])
                    print(pd.DataFrame([[friend_name, crawler.code]], columns=['name', 'code']))
                    data['index'] += 1
                    
                    with open(self.yaml_path, 'w+', encoding='utf-8') as file:
                        yaml.dump(data, file, allow_unicode=True, sort_keys=True)
                        file.close()
                        
                    self.search_friend_name()
                        
                time.sleep(3)
                
            if epoch != data['episode'] - 1:
                delay_time = 5
                # 这里多加几秒的延迟响应时间, 同时这一轮最少也要等这么久，别响应太快了
                with tqdm(total=data['delay'] + delay_time, desc=f"Time", unit='s') as pbar:
                    diff = time.time() - ts
                    pbar.update(int(diff))  # 更新进度条为当前时间
                    for i in range(int(data['delay'] - int(diff) + delay_time)):
                        time.sleep(1)
                        pbar.update(1)
                        
        print("CrawlerProgramThread 线程已结束")
        self.finished.emit()

# 收集线程
class HeartProgramThread(BaseThread):
    finished = Signal()
    
    def __init__(self, sigma=0.07):
        super().__init__(sigma=sigma)
        
        self.detector = NanokaDetector()
    
    def receive_hearts(self, hearts_info) -> None:
        '''
        接收爱心的函数, 完全是操纵行为
        '''
        for i, (x, y, w, h) in enumerate(hearts_info):
            # 移动鼠标到目标中心, 考虑了屏幕分辨率的影响
            timeout = 3
            while timeout > 0:
                if self.check_text(): # 我们现在应该在星盘页才对
                    pyautogui.moveTo(int(x*self.width//1920), int(y*self.height//1080)) 
                    self.gauss_sleep(0.1)
                    pydirectinput.click()  # 点击目标，没送心火的话进入到了送心的人的星盘页
                    self.gauss_sleep(2)
                    break
                else:
                    print("检测到我们不在星盘页需要重新定位") # 由于是自动程序控制所以不用担心在这里卡死
                    timeout -= 1
                    
                    pydirectinput.keyDown('g')
                    self.gauss_sleep(0.1)
                    pydirectinput.keyUp('g')
                    self.gauss_sleep(2)
                    
                    self.goto_page(self.page)
                    continue
            if timeout == 0:
                raise RuntimeError("无法定位到星盘页，请检查程序运行状态")
            
            # 这里有一个分歧, 如果检测出来没有文字了那么我们就不再点一次了
            if self.check_text():
                pydirectinput.click()  # 点击目标， 这次一定进入大屏星盘页
                self.gauss_sleep(2)
            
            for _ in range(5):
                pydirectinput.keyDown('up')
                self.gauss_sleep(0.1)
                pydirectinput.keyUp('up')
                self.gauss_sleep(0.3)
                
            for _ in range(2):
                pydirectinput.keyDown('down')
                self.gauss_sleep(0.1)
                pydirectinput.keyUp('down')
                self.gauss_sleep(0.1)
                
            self.gauss_sleep(0.4)  # 等待响应
                
            # 如果这里意外颠倒了其他按钮, 尤其是删除或拉黑好友, 我们要做出应急响应
            pydirectinput.keyDown('space')
            self.gauss_sleep(0.1)
            pydirectinput.keyUp('space')
            self.gauss_sleep(0.5)
            
            # 加入我们点错了, 我们这一步不能回到星盘页
            pydirectinput.keyDown('esc')
            self.gauss_sleep(0.1)
            pydirectinput.keyUp('esc')
            
            wait_responce_sec = 2.0
            self.gauss_sleep(wait_responce_sec) # 等待页面加载完成, 这一步一定要等待充分
            
            if not self.check_text():
                print("检测到我们没回到星盘, 存在按钮误操作的可能性, 进行应急处理")
                pydirectinput.keyDown('esc') # 这次肯定回到星盘了
                self.gauss_sleep(0.1)
                pydirectinput.keyUp('esc')
                self.gauss_sleep(wait_responce_sec)
                
    def receive_stars(self, post_points, labels) -> None:
        '''
        接收心火的函数, 完全是操纵行为
        '''
        for i, ((x, y, w, h), cls) in enumerate(zip(post_points, labels)):
            # 星屑检测结果, 同时过滤鼠标效果
            if int(cls) == 0:
                # 第一轮判断
                timeout = 3
                while timeout > 0:
                    if self.check_text(): # 我们现在应该在星盘页才对
                        pyautogui.moveTo(int(x*self.width//1920), int(y*self.height//1080)) 
                        self.gauss_sleep(0.1)
                        pydirectinput.click()  # 点击目标，没送心火的话进入到了送心的人的星盘页
                        self.gauss_sleep(0.8)
                        break
                    else:
                        print("检测到我们不在星盘页需要重新定位") # 由于是自动程序控制所以不用担心在这里卡死
                        timeout -= 1
                        
                        pydirectinput.keyDown('g')
                        self.gauss_sleep(0.1)
                        pydirectinput.keyUp('g')
                        self.gauss_sleep(2)
                        
                        self.goto_page(self.page)
                        continue
                if timeout == 0:
                    raise RuntimeError("无法定位到星盘页，请检查程序运行状态")
                
                # 第二轮判断, 如果进去了就出来
                if not self.check_text(): # 我们现在应该在星盘页才对, 如果不在那就是在详情页里, 那我们就回退
                    pyautogui.moveTo(2*self.width//10, self.height//2) # 通过鼠标点击的方式更安全
                    self.gauss_sleep(0.5)
                    pydirectinput.click()
                    self.gauss_sleep(0.8)
                
                if not self.check_text():
                    pydirectinput.keyDown('g') # 如果还没有的话那就只能是卡退了, 按 g 回去并且恢复页面
                    self.gauss_sleep(0.1)
                    pydirectinput.keyUp('g')
                    self.gauss_sleep(2.5)
                    
                    self.goto_page(self.page)  # 恢复到当前页
                    
                if not self.check_page():
                    raise RuntimeError("无法定位到星盘页，请检查程序运行状态")
    
    def give_stars(self, pre_points) -> None:
        '''
        送出心火的函数, 完全是操纵行为
        '''
        # 逐个处理检测到的目标
        for i, (x, y, w, h) in enumerate(pre_points):
            # 第一轮判断
            timeout = 3
            while timeout > 0:
                if self.check_text(): # 我们现在应该在星盘页才对
                    pyautogui.moveTo(int(x*self.width//1920), int(y*self.height//1080)) 
                    self.gauss_sleep(0.1)
                    pydirectinput.keyDown('space') # 点击目标，没送心火的话进入到了送心的人的星盘页
                    self.gauss_sleep(0.1)
                    pydirectinput.keyUp('space')
                    self.gauss_sleep(1.5)
                    break
                else:
                    print("检测到我们不在星盘页需要重新定位") # 由于是自动程序控制所以不用担心在这里卡死
                    timeout -= 1
                    
                    pydirectinput.keyDown('g')
                    self.gauss_sleep(0.1)
                    pydirectinput.keyUp('g')
                    self.gauss_sleep(2)
                    
                    self.goto_page(self.page)
                    continue
            if timeout == 0:
                raise RuntimeError("无法定位到星盘页，请检查程序运行状态")
            
            if self.check_text():
                # print("检测没有进入星盘页, 需要重新点击")
                pydirectinput.keyDown('space') # 这都进不去鉴定为识别错了
                self.gauss_sleep(0.1)
                pydirectinput.keyUp('space')
                self.gauss_sleep(1.5)
                
                
            if not self.check_text(): 
                pydirectinput.keyDown('f')
                self.gauss_sleep(0.1)
                pydirectinput.keyUp('f')
                self.gauss_sleep(0.5)
                
                pydirectinput.keyDown('esc')
                self.gauss_sleep(0.1)
                pydirectinput.keyUp('esc')
                self.gauss_sleep(1.2)
            
    
    def run(self):
        '''
        赠送心火主线程，用来从行为上控制程序
        '''
        self.page = 0
        # 首先我们先把星盘定位到添加好友
        self.goto_page(0)  # 跳转到添加好友页
        
        timeout = 50  # 最多 50 页好友, 不能比这还多了吧???
        while timeout > 0:
            self.next_page() # 3s 延时
            self.mouse_clear() # 0.5s 延时

            # 截屏并分析
            frame = self.screenshot()
            self.detector.update_image(frame) 
            is_over, pdata, _ = self.check_page(plot=True, show=False)
            if is_over:
                print("检测到添加好友页，退出程序")
                break
            
            # 先处理需要送心的点，其他的不重要
            hearts_info = pdata['hearts_info']
            # 逐个处理检测到的目标
            self.receive_hearts(hearts_info=hearts_info)
                    
            # 重新分析当前页面
            frame = self.screenshot()
            self.detector.update_image(frame) 
            is_over, pdata, _ = self.check_page(plot=True, show=False)
            if is_over:
                print("检测到添加好友页，退出程序")
                break
            
            # 接下来处理所有需要收心的点, 先处理 post 的点
            post_points = pdata['post_points']
            labels = pdata['labels']
            self.receive_stars(post_points=post_points, labels=labels)
                    
            # 接下来开始送心火，处理 pre 的点, 这里因为没有爱心变化所以我们不需要重读
            pre_points = pdata['pre_points']
            self.give_stars(pre_points=pre_points)
        
        print("HeartProgramThread 线程已结束")
        self.finished.emit()
    