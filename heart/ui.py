import os
os.environ["QT_SCALE_FACTOR"] = "1.0"
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                            QVBoxLayout, QWidget, QHBoxLayout)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QRect, QPoint
from PySide6.QtGui import (QColor, QPalette, QRegion, QPainterPath, 
                          QCursor, QPainter, QPen)
from deploy import multi_detector, ocr, ocr_detector, re_keyword_detector
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
    
    def run(self):
        # 首先我们先把星盘定位到添加好友
        timeout = 50
        while timeout > 0:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 转换为BGR格式
            
            pdata, _ = multi_detector(gray, frame, plot=False)
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
            # 不停向右检测图片即可
            pydirectinput.keyDown('c')
            time.sleep(abs(random.gauss(0.6, 1)))
            pydirectinput.keyUp('c')
            time.sleep(3) 
            
            # 处理10次最后统计出一个比较可靠的结果
            tfa_targets, reliable_points = [], []
            for _ in range(10):
                screenshot = pyautogui.screenshot()
                frame = np.array(screenshot)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 转换为BGR格式
                
                pdata, img = multi_detector(gray, frame, plot=True)
                if pdata['code'] == 1:
                    print("检测到添加好友页，退出程序")
                    pydirectinput.keyDown('esc')
                    time.sleep(abs(random.gauss(0.6, 1)))
                    pydirectinput.keyUp('esc')
                    break
                
                if img is not None:
                    plt.figure(figsize=(10, 6))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.savefig(os.path.join('runs', 'predict', str(datetime.now()).replace(':','').replace(' ','-')+'.png'))
                    plt.close()
                
                # 这是我们检测到的全部目标
                tfa_targets.append(np.array([(x, y) for ((_, _, _, _), (x, y, _), _) in pdata['tfa_targets']]))
                time.sleep(0.1)
            
            max_num_idx = np.argmax([len(target) for target in tfa_targets])
            base_targets = tfa_targets[max_num_idx]
            # print('base_targets', base_targets, '\n\n')
            all_targets = np.concatenate(tfa_targets)
            # print('all_targets', all_targets, '\n\n')
            distance_threshold = min(frame.shape) * 0.04  # 基于图像尺寸的动态阈值
            # print("距离阈值:", distance_threshold, '\n\n')
            
            # 遍历基准目标中的每个点，筛选可靠点
            for point in base_targets:
                distances = np.sqrt(np.sum((point - all_targets) ** 2, axis=1))
                # print("所有距离参数:", distances, '\n\n')
                close_points = all_targets[distances < distance_threshold]
                # print("检测到的目标数量:", len(close_points), '\n\n')
                # 如果有足够多的点接近当前点，则认为这个点是可靠的
                if len(close_points) >= 6:  # 至少需要3个点支持（包括自身）
                    # 计算这些点的中心点作为最终可靠点
                    center_point = np.mean(close_points, axis=0)
                    reliable_points.append(center_point)
            
            # # 统计五次检测到的目标
            # max_num_tfa_targets = np.argmax([len(target) for target in tfa_targets], axis=0)
            # # 组合所有检测到的目标
            # all_target = np.confccatenate(tfa_targets, axis=0)
            
            # ita = min(frame.shape)*0.02
            # unfit_sample = tfa_targets[max_num_tfa_targets]
            # for points in unfit_sample:
            #     # 形状为 (1, 2), 减去 (total, 2) 之后需要有至少四个重复的数, 否则我们不认为有效
            #     diff = np.sum(np.array(points).reshape(1, 2) - all_target, axis=1)
            #     selector = np.where(np.abs(diff) < ita, 1, 0)
            #     if np.sum(selector) >= 4:
            #         # 说明这个点是有效的
            #         medium_p = np.sum(all_target * selector)/np.sum(selector)
            #         reliable_points.append(medium_p)
            
            # 逐个处理检测到的目标
            for i, (x, y) in enumerate(reliable_points):
                # 一套赠送心火流程
                pyautogui.moveTo(x, y)  # 移动鼠标到目标中心
                time.sleep(0.1)
                pyautogui.click()  # 点击目标
                time.sleep(0.6)
                
                # 这里有一个分歧, 如果检测出来没有文字了那么我们就不再点一次了
                screenshot = pyautogui.screenshot()
                frame = np.array(screenshot)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 转换为BGR格式
                text = ocr_detector(gray)
                pf = re_keyword_detector([text])
                if bool(pf['星盘页'].values[0]):
                    pyautogui.click()  # 点击目标
                    time.sleep(0.6)
                
                
                pydirectinput.keyDown('f')
                time.sleep(0.2)
                pydirectinput.keyUp('f')
                time.sleep(0.5)
                
                pydirectinput.keyDown('esc')
                time.sleep(0.1)
                pydirectinput.keyUp('esc')
                time.sleep(2)
                
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
        
        # 拖动区域
        self.drag_area = QWidget()
        self.drag_area.setMinimumHeight(30)
        self.drag_area.setStyleSheet("background-color: rgba(0, 0, 0, 153);"  # 黑色，60%透明度
                                    "border-radius: 10px;")  # 拖动区域圆角
        self.drag_area.mousePressEvent = self.start_drag
        self.drag_area.mouseMoveEvent = self.drag_move
        self.drag_area.mouseReleaseEvent = self.stop_drag
        
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
        
        top_layout.addWidget(self.drag_area, 1)  # 占据剩余空间
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