# coding=utf-8
import os
os.environ["QT_SCALE_FACTOR"] = "1.0"
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                            QVBoxLayout, QWidget, QHBoxLayout, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QRect, QPoint
from PySide6.QtGui import (QColor, QPalette, QRegion, QPainterPath, 
                          QCursor, QPainter, QPen)
from deploy import NanokaDetector
import cv2
import pyautogui

import pydirectinput
pydirectinput.FAILSAFE = False

import numpy as np
import matplotlib.pyplot as plt
import random

'''
我们最后输出的数据格式

pdata = {
    "code": 0, 
    "info": "识别成功", 
    "hearts_info": hearts_info, 
    "pre_points": pre_points,
    "post_points": post_points,
    "labels": labels,
}
''' 

class MainProgramThread(QThread):
    finished = Signal()
    
    def __init__(self, sigma=0.07):
        super().__init__()
        self.detector = NanokaDetector()
        screenshot = np.array(pyautogui.screenshot())
        self.height, self.width = int(screenshot.shape[0]), int(screenshot.shape[1])
        print("当前屏幕分辨率: {}x{}".format(self.width, self.height))
        self.sigma = sigma
        self.page = 0
        
###################################### 琐碎的函数 ######################################
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
        '''检测左下角是否存在我们期待的文字, 如果有返回 True, 否则返回 False'''
        frame = self.screenshot()
        gray = cv2.cvtColor(cv2.resize(frame, (1920, 1080)), cv2.COLOR_RGB2GRAY)
        text = self.detector.ocr_detector(gray=gray)
        pf = self.detector.re_keyword_detector([text])
        return bool(pf['星盘页'].values[0])

        
###################################### 页面控制函数 ######################################
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
        '''向右检测图片'''
        pydirectinput.keyDown('c')
        self.gauss_sleep(0.6)
        pydirectinput.keyUp('c')
        self.gauss_sleep(3)
        
        # 页数加一表示现在在那页否则会死循环
        self.page += 1
        
    def check_page(self, plot=False, show=False) -> tuple:
        '''检查当前页面是否是正常星盘页'''
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
    
###################################### 行为控制函数 ######################################
    def receive_hearts(self, hearts_info) -> None:
        """接收爱心的函数"""
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
        """接收心火的函数"""
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
        """送出心火的函数"""
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
            
            if self.check_text():
                # print("检测没有进入星盘页, 需要重新点击")
                pydirectinput.keyDown('space') # 这都进不去鉴定为识别错了
                self.gauss_sleep(0.1)
                pydirectinput.keyUp('space')
                self.gauss_sleep(0.6)
                
                
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
        
        self.finished.emit()

class TransparentWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口属性
        self.setWindowFlags(
            Qt.FramelessWindowHint |  # 无边框
            Qt.WindowStaysOnTopHint |  # 置顶
            Qt.Tool  # 不在任务栏显示
        )
        self.setAttribute(Qt.WA_TranslucentBackground)  # 背景透明
        
        # 设置窗口尺寸和位置（屏幕顶部）
        screen_geometry = QApplication.primaryScreen().geometry()
        self.setGeometry(100, 0, 300, 150)  # 调整宽度和高度
        
        # 鼠标拖动相关变量
        self.dragging = False
        self.offset = None
        self.border_radius = 15  # 窗口圆角半径
        
        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 顶部布局（用于退出按钮和拖动区域）
        top_layout = QHBoxLayout()
        
        self.horizontal_spacer = QSpacerItem(
            0,                      # 水平最小宽度（可设为0）
            30,                     # 垂直最小高度
            QSizePolicy.Expanding,  # 水平方向可扩展
            QSizePolicy.Minimum     # 垂直方向固定
        )
        
        self.mousePressEvent = self.start_drag
        self.mouseMoveEvent = self.drag_move
        self.mouseReleaseEvent = self.stop_drag
        
        # 圆角矩形退出按钮
        self.exit_button = QPushButton("×")
        self.exit_button.setFixedSize(25, 25)
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: white;
                border: none;
                border-radius: 12px;  /* 圆角半径为宽度的一半，形成圆形 */
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.exit_button.clicked.connect(self.quit_application)
        
        top_layout.addSpacerItem(self.horizontal_spacer)  # 占据剩余空间
        top_layout.addWidget(self.exit_button)
        
        main_layout.addLayout(top_layout)
        
        # 固定宽度(250px)的主程序按钮
        self.main_button = QPushButton("执行主程序")
        self.main_button.setFixedSize(250, 40)  # 固定宽度250px，高度40px
        self.main_button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: none;
                border-radius: 20px;  /* 圆角半径为高度的一半，形成圆角矩形 */
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #777777;
            }
            QPushButton:pressed {
                background-color: #444444;
            }
        """)
        self.main_button.clicked.connect(self.run_main_program)
        main_layout.addWidget(self.main_button, alignment=Qt.AlignCenter)
        
        # 设置主窗口
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        central_widget.setStyleSheet(f"background-color: rgba(0, 0, 0, 153);"  # 黑色，60%透明度
                                    f"border-radius: {self.border_radius}px;")  # 圆角矩形
        self.setCentralWidget(central_widget)
        
        # 创建主程序线程
        self.main_thread = MainProgramThread()
        self.main_thread.finished.connect(self.show_window)
    
    def run_main_program(self):
        # 隐藏窗口
        self.hide()
        time.sleep(1)  # 确保窗口隐藏后再执行主程序
        
        # 启动主程序线程
        self.main_thread.start()
    
    def show_window(self):
        # 显示窗口
        self.show()
    
    def quit_application(self):
        """安全退出应用程序并结束进程"""
        # 确保线程已停止
        if self.main_thread.isRunning():
            self.main_thread.quit()
            self.main_thread.wait()
        
        # 退出应用程序
        QApplication.quit()
    
    # 鼠标拖动功能实现
    def start_drag(self, event):
        """鼠标按下事件，开始拖动"""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.offset = event.globalPosition().toPoint() - self.pos()
    
    def drag_move(self, event):
        """鼠标移动事件，处理拖动"""
        if self.dragging:
            self.move(event.globalPosition().toPoint() - self.offset)
    
    def stop_drag(self, event):
        """鼠标释放事件，停止拖动"""
        self.dragging = False
    
    def resizeEvent(self, event):
        """调整窗口大小时，重新设置圆角区域"""
        path = QPainterPath()
        path.addRoundedRect(QRect(0, 0, self.width(), self.height()), 
                           self.border_radius, self.border_radius)
        self.setMask(path.toFillPolygon().toPolygon())
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置全局字体，确保中文显示正常
    font = app.font()
    font.setFamily("SimHei")  # 使用黑体等中文字体
    app.setFont(font)
    
    window = TransparentWindow()
    window.show()
    
    sys.exit(app.exec())