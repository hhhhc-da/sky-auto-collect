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
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

class MainProgramThread(QThread):
    """执行主程序的后台线程"""
    finished = Signal()
    
    def __init__(self):
        super().__init__()
        # 检测器初始化
        self.detector = NanokaDetector()
        
        screenshot = np.array(pyautogui.screenshot())
        self.height, self.width = int(screenshot.shape[0]), int(screenshot.shape[1])
        print("当前屏幕分辨率: {}x{}".format(self.width, self.height))
    
    def run(self):
        # 首先我们先把星盘定位到添加好友
        timeout = 50
        while timeout > 0:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            self.detector.update_image(frame)  # 更新检测器的图像数据
            
            pdata = None
            timeout_detector = 10  # 每次检测最多尝试10次
            while timeout_detector > 0:
                try:
                    pdata, img = self.detector.multi_detector(plot=False)
                    break
                except Exception as e:
                    print(f"检测失败: {e}")
                    timeout_detector -= 1
                   
            if pdata['code'] != 1:
                timeout -= 1
                
                # 不停向左检测图片
                pydirectinput.keyDown('z')
                time.sleep(abs(random.gauss(0.6, 1)))
                pydirectinput.keyUp('z')
                print("未检测到添加好友页，继续向左检测... (剩余尝试次数: {})".format(timeout))
                time.sleep(1)
                continue
            else:
                print("检测到添加好友页，开始执行主程序")
                break
        
        if timeout <= 0:
            print("未检测到添加好友页，退出程序")
            return
        
        timeout = 50  # 最多 50 页好友, 不能比这还多了吧???
        while timeout > 0:
            try:
                # 不停向右检测图片即可
                pydirectinput.keyDown('c')
                time.sleep(abs(random.gauss(0.6, 1)))
                pydirectinput.keyUp('c')
                time.sleep(3) 
                pydirectinput.moveTo(0, 0)
                time.sleep(0.5)

                screenshot = pyautogui.screenshot()
                frame = np.array(screenshot)
                self.detector.update_image(frame)  # 更新检测器的图像数据
                
                '''
                pdata = {
                    "code": 0, 
                    "info": "识别成功", 
                    "hearts_info": hearts_info, 
                    "pre_points": pre_points,
                    "post_points": post_points,
                    "labels": labels,
                }
                ''' 
                pdata, img = None, None
                timeout_detector = 10  # 每次检测最多尝试10次
                while timeout_detector > 0:
                    try:
                        pdata, img = self.detector.multi_detector(plot=False)
                        if pdata['code'] == 1:
                            print("检测到添加好友页，退出程序")
                            pydirectinput.keyDown('esc')
                            time.sleep(abs(random.gauss(0.6, 1)))
                            pydirectinput.keyUp('esc')
                            
                            raise Exception("Over")
                        if pdata['code'] == 0:
                            break
                        time.sleep(0.3) # 等待下一次检测
                    except Exception as e:
                        timeout_detector -= 1
                        if str(e) == "Over":
                            raise e
                        print(f"检测失败: {e}")
                
                
                if img is not None:
                    plt.figure(figsize=(10, 6))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.savefig(os.path.join('runs', 'predict', str(datetime.now()).replace(':','').replace(' ','-')+'.png'))
                    plt.close()
                
    ################################################### 接收爱心 ###################################################                    
                # 先处理需要送心的点，其他的不重要
                hearts_info = pdata['hearts_info']
                
                # 逐个处理检测到的目标
                for i, (x, y, w, h) in enumerate(hearts_info):
                    # 一套收爱心流程
                    pydirectinput.moveTo(int(x*self.width//1920), int(y*self.height//1080))  # 移动鼠标到目标中心
                    time.sleep(0.1)
                    pydirectinput.click()  # 点击目标，没送心火的话进入到了送心的人的星盘页
                    time.sleep(0.6)
                    # 这里有一个分歧, 如果检测出来没有文字了那么我们就不再点一次了
                    screenshot = pyautogui.screenshot()
                    frame = np.array(screenshot)
                    gray = cv2.cvtColor(cv2.resize(frame, (1080, 1920)), cv2.COLOR_RGB2GRAY)  # 转换为BGR格式
                    text = self.detector.ocr_detector(gray=gray)
                    pf = self.detector.re_keyword_detector([text])
                    if bool(pf['星盘页'].values[0]):
                        pydirectinput.click()  # 点击目标， 这次一定进入大屏星盘页
                        time.sleep(0.6)
                    
                    for _ in range(8):
                        pydirectinput.keyDown('up')
                        time.sleep(0.1)
                        pydirectinput.keyUp('up')
                        
                    for _ in range(2):
                        pydirectinput.keyDown('down')
                        time.sleep(0.1)
                        pydirectinput.keyUp('down')
                        
                    # 如果这里意外颠倒了其他按钮, 尤其是删除或拉黑好友, 我们要做出应急响应
                    pydirectinput.keyDown('space')
                    time.sleep(0.2)
                    pydirectinput.keyUp('space')
                    time.sleep(0.5)
                    
                    # 加入我们点错了, 我们这一步不能回到星盘页
                    pydirectinput.keyDown('esc')
                    time.sleep(0.1)
                    pydirectinput.keyUp('esc')
                    time.sleep(2)
                    
                    screenshot = pyautogui.screenshot()
                    frame = np.array(screenshot)
                    gray = cv2.cvtColor(cv2.resize(frame, (1080, 1920)), cv2.COLOR_RGB2GRAY)  # 转换为BGR格式
                    text = self.detector.ocr_detector(gray=gray)
                    pf = self.detector.re_keyword_detector([text])
                    if bool(pf['星盘页'].values[0] == False): # 我们没回到星盘
                        print("检测到我们没回到星盘, 信息:", pf)
                        pydirectinput.keyDown('esc') # 这次肯定回到星盘了
                        time.sleep(0.1)
                        pydirectinput.keyUp('esc')
                        time.sleep(2)
                        
                pydirectinput.moveTo(0, 0)
                time.sleep(0.5)        
                
                # 我们还在该页的送心火循环中
                screenshot = pyautogui.screenshot()
                frame = np.array(screenshot)
                self.detector.update_image(frame)  # 更新检测器的图像数据
                
                '''
                pdata = {
                    "code": 0, 
                    "info": "识别成功", 
                    "hearts_info": hearts_info, 
                    "pre_points": pre_points,
                    "post_points": post_points,
                    "labels": labels,
                }
                ''' 
                pdata, img = None, None
                timeout_detector = 10  # 每次检测最多尝试10次
                while timeout_detector > 0:
                    try:
                        pdata, img = self.detector.multi_detector(plot=False)
                        if pdata['code'] == 0:
                            break
                        time.sleep(0.3) # 等待下一次检测
                    except Exception as e:
                        print(f"检测失败: {e}")
                        timeout_detector -= 1
                        
    ################################################### 接收心火 ###################################################
                # 接下来处理所有需要收心的点, 先处理 post 的点
                post_points = pdata['post_points']
                labels = pdata['labels']
                
                # 逐个处理检测到的目标
                for i, ((x, y, w, h), cls) in enumerate(zip(post_points, labels)):
                    if int(cls) == 0:
                        # 一套赠送心火流程
                        pydirectinput.moveTo(int(x*self.width//1920), int(y*self.height//1080))  # 移动鼠标到目标中心
                        time.sleep(0.1)
                        pydirectinput.click()  # 点击目标，不出意外的话我们应该还在星盘页
                        time.sleep(0.6)
                        
                        # 这里有一个分歧, 如果检测出来没有文字了那就是进去了, 因为星屑识别不准, 但是能抑制鼠标
                        screenshot_2 = pyautogui.screenshot()
                        frame_2 = np.array(screenshot_2)
                        gray_2 = cv2.cvtColor(cv2.resize(frame_2, (1080, 1920)), cv2.COLOR_RGB2GRAY)  # 转换为BGR格式
                        text_2 = self.detector.ocr_detector(gray=gray_2)
                        pf_2 = self.detector.re_keyword_detector([text_2])
                        if not bool(pf_2['星盘页'].values[0]):
                            print("检测到已退出星盘页, 信息:", pf_2)
                            pydirectinput.keyDown('esc') # 退回到星盘
                            time.sleep(0.1)
                            pydirectinput.keyUp('esc')
                            time.sleep(0.6)
                        
                        screenshot = pyautogui.screenshot()
                        frame = np.array(screenshot)
                        gray = cv2.cvtColor(cv2.resize(frame, (1080, 1920)), cv2.COLOR_RGB2GRAY)  # 转换为BGR格式
                        text = self.detector.ocr_detector(gray=gray)
                        pf = self.detector.re_keyword_detector([text])
                        if bool(pf['星盘页'].values[0] == False):
                            pydirectinput.keyDown('g') # 如果还没有的话那就只能是卡退了
                            time.sleep(0.1)
                            pydirectinput.keyUp('g')
                            time.sleep(2.5)
                        
    ################################################### 接收心火 ###################################################
                # 接下来开始送心火，处理 pre 的点
                pre_points = pdata['pre_points']
                
                # 逐个处理检测到的目标
                for i, (x, y, w, h) in enumerate(pre_points):
                    # 一套送心火流程
                    pydirectinput.moveTo(int(x*self.width//1920), int(y*self.height//1080))
                    time.sleep(0.1)
                    pydirectinput.click()  # 点击目标，没送心火的话进入到了送心的人的星盘页
                    time.sleep(0.6)
                    # 这里有一个分歧, 如果检测出来没有文字了那么我们就不再点一次了
                    screenshot_3 = pyautogui.screenshot()
                    frame_3 = np.array(screenshot_3)
                    gray_3 = cv2.cvtColor(cv2.resize(frame_3, (1080, 1920)), cv2.COLOR_RGB2GRAY)  # 转换为BGR格式
                    text_3 = self.detector.ocr_detector(gray=gray_3)
                    pf_3 = self.detector.re_keyword_detector([text_3])
                    if bool(pf_3['星盘页'].values[0]):
                        print("检测没有进入星盘页, 信息:", pf_3)
                        pydirectinput.click()  # 点击目标， 这次一定进入大屏星盘页
                        time.sleep(0.6)
                        
                    pydirectinput.keyDown('f')
                    time.sleep(0.1)
                    pydirectinput.keyUp('f')
                    time.sleep(0.5)
                    
                    pydirectinput.keyDown('esc')
                    time.sleep(0.1)
                    pydirectinput.keyUp('esc')
                    time.sleep(0.5)
                    
                    # 检测是否还在星盘页
                    pydirectinput.moveTo(0, 0)  # 移动鼠标到屏幕左上角
                    time.sleep(0.5)  # 等待鼠标移动完成
                    
                    screenshot = pyautogui.screenshot() 
                    frame = np.array(screenshot)
                    gray = cv2.cvtColor(cv2.resize(frame, (1080, 1920)), cv2.COLOR_RGB2GRAY)  # 转换为BGR格式
                    text = self.detector.ocr_detector(gray=gray)
                    pf = self.detector.re_keyword_detector([text])
                    if bool(pf['星盘页'].values[0] == False):
                        pydirectinput.keyDown('g') # 如果还没有的话那就只能是卡退了
                        time.sleep(0.1)
                        pydirectinput.keyUp('g')
                        time.sleep(2)
            except Exception as e:
                # 如果是 Over 异常，直接退出循环
                if str(e) == "Over":
                    break
                else:
                    print(f"主程序执行异常: {e}")
                    self.finished.emit()
                    raise e
        
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