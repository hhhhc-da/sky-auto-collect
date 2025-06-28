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
import cv2
import pyautogui

import pydirectinput
pydirectinput.FAILSAFE = False

import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

from main_process import HeartProgramThread, CrawlerProgramThread

def opt_parser():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Web Crawler for Heart Collection")
    parser.add_argument('--yaml', type=str, default="config.yaml", help='需要读取的 yaml 文件位置')
    return parser.parse_args()

class TransparentWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.opt = opt_parser()
        
        # 设置窗口属性
        self.setWindowFlags(
            Qt.FramelessWindowHint |  # 无边框
            Qt.WindowStaysOnTopHint |  # 置顶
            Qt.Tool  # 不在任务栏显示
        )
        self.setAttribute(Qt.WA_TranslucentBackground)  # 背景透明
        
        # 设置窗口尺寸和位置（屏幕顶部）
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
        
        # 固定宽度(250px)的按钮
        self.heart_button = QPushButton("自动收取心火")
        self.heart_button.setFixedSize(250, 40)  # 固定宽度250px，高度40px
        self.heart_button.setStyleSheet("""
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
        self.heart_button.clicked.connect(self.run_heart_program)
        main_layout.addWidget(self.heart_button, alignment=Qt.AlignCenter)
        
        self.request_button = QPushButton("主动请求爱心")
        self.request_button.setFixedSize(250, 40)  # 固定宽度250px，高度40px
        self.request_button.setStyleSheet("""
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
        self.request_button.clicked.connect(self.run_request_program)
        main_layout.addWidget(self.request_button, alignment=Qt.AlignCenter)
        
        # 设置主窗口
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        central_widget.setStyleSheet(f"background-color: rgba(0, 0, 0, 153);"  # 黑色，60%透明度
                                    f"border-radius: {self.border_radius}px;")  # 圆角矩形
        self.setCentralWidget(central_widget)
        
        # 创建各个处理线程
        self.heart_thread = HeartProgramThread()
        self.heart_thread.finished.connect(self.show_window)
        self.request_thread = CrawlerProgramThread(yaml=self.opt.yaml)
        self.request_thread.finished.connect(self.show_window)
        
    
    def run_heart_program(self):
        self.hide()
        time.sleep(1)
        self.heart_thread.start()
        
    def run_request_program(self):
        self.hide()
        time.sleep(1)
        self.request_thread.start()
    
    def show_window(self):
        self.show()
    
    def quit_application(self):
        """安全退出应用程序并结束进程"""
        for thread in [self.heart_thread, self.request_thread]:
            if thread.isRunning():
                thread.quit()
                thread.wait()
        
        QApplication.quit()
    
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
    
    font = app.font()
    font.setFamily("SimHei")
    app.setFont(font)
    
    window = TransparentWindow()
    window.show()
    
    sys.exit(app.exec())