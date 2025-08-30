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

import sys
import random
import math
from PySide6.QtCore import QTimer, QPointF
from PySide6.QtGui import QColor, QPainter, QIcon
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from scipy.spatial import Delaunay

class Particle:
    def __init__(self, x, y, angle, radius, speed, color):
        self.origin_x = x
        self.origin_y = y
        self.angle = angle
        self.radius = radius
        self.speed = speed
        self.color = color
        self.size = random.uniform(1, 3)
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.vx = 0
        self.vy = 0
        self.expanding = False
        self.rebounding = False
        self.expansion_speed = 0
        self.remaining_distance = 0
        self.min_speed_threshold = 0.5
        self.min_rebound_speed = 0.5  # 最低回弹速度

    def update(self):
        if self.expanding:
            if self.remaining_distance > 0 and self.expansion_speed > self.min_speed_threshold:
                move_distance = min(self.expansion_speed, self.remaining_distance)
                self.x += self.vx * move_distance
                self.y += self.vy * move_distance
                self.remaining_distance -= move_distance
                self.expansion_speed *= 0.9  # 扩散速度快速衰减
            else:
                self.expanding = False
                self.rebounding = True
        elif self.rebounding:
            # 动态计算目标位置
            self.target_x = self.origin_x + self.radius * math.cos(self.angle)
            self.target_y = self.origin_y + self.radius * math.sin(self.angle)

            dx = self.target_x - self.x
            dy = self.target_y - self.y
            distance = math.hypot(dx, dy)

            self.vx = dx / 30
            self.vy = dy / 30

            if abs(self.vx) < self.min_rebound_speed and abs(self.vy) < self.min_rebound_speed:
                self.vx = dx / abs(dx) * self.min_rebound_speed if dx != 0 else 0
                self.vy = dy / abs(dy) * self.min_rebound_speed if dy != 0 else 0

            self.x += self.vx
            self.y += self.vy

            if distance < 2.1:  # 到达目标位置
                self.rebounding = False
        else:
            self.angle += self.speed
            self.x = self.origin_x + self.radius * math.cos(self.angle)
            self.y = self.origin_y + self.radius * math.sin(self.angle)

    def trigger_expansion(self, mouse_x, mouse_y):
        self.expanding = True
        self.rebounding = False
        dx, dy = self.x - mouse_x, self.y - mouse_y
        distance = math.hypot(dx, dy)
        if distance == 0:
            self.vx, self.vy = 1, 0
        else:
            self.vx, self.vy = dx / distance, dy / distance
        self.remaining_distance = (150 - distance) / 150 * 500
        self.expansion_speed = self.remaining_distance / 70  # 初始扩散速度与距离相关

class ParticleWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("光遇辅助工具箱")
        self.setWindowIcon(QIcon("ui/icon.png"))  # 设置窗口图标
        self._width = 300
        self._height = 150
        self.resize(self._width, self._height)

        self.opt = opt_parser()

        # 创建主布局
        main_layout = QVBoxLayout(self)

        # 创建新的 widget
        overlay_widget = QWidget(self)
        overlay_widget.setStyleSheet(
            "background-color: rgba(205, 133, 63, 0.3);"  # 淡咖啡色，透明度30%
        )
        overlay_widget.setGeometry(0, 0, int(self._width * 0.9), int(self._height * 0.9))  # 宽占90%，高占90%

        # 创建按钮并添加到 overlay_widget
        button_layout = QVBoxLayout(overlay_widget)

        self.horizontal_spacer = QSpacerItem(
            0,                      # 水平最小宽度（可设为0）
            5,                     # 垂直最小高度
            QSizePolicy.Expanding,  # 水平方向可扩展
            QSizePolicy.Minimum     # 垂直方向固定
        )
        self.horizontal_spacer_2 = QSpacerItem(
            0,                      # 水平最小宽度（可设为0）
            5,                     # 垂直最小高度
            QSizePolicy.Expanding,  # 水平方向可扩展
            QSizePolicy.Minimum     # 垂直方向固定
        )

        button_layout.addSpacerItem(self.horizontal_spacer)

        self.heart_button = QPushButton("自动收取心火")
        self.heart_button.setFixedSize(250, 40)  # 固定宽度250px，高度40px
        self.heart_button.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 20px;  /* 圆角半径为高度的一半，形成圆角矩形 */
                font-size: 14px;
            }
        """)
        self.heart_button.clicked.connect(self.run_heart_program)

        button_layout.addWidget(self.heart_button)

        self.request_button = QPushButton("主动请求爱心")
        self.request_button.setFixedSize(250, 40)  # 固定宽度250px，高度40px
        self.request_button.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 20px;  /* 圆角半径为高度的一半，形成圆角矩形 */
                font-size: 14px;
            }
        """)
        self.request_button.clicked.connect(self.run_request_program)
        main_layout.addWidget(self.request_button, alignment=Qt.AlignCenter)

        button_layout.addWidget(self.request_button)

        self.auto_run_button = QPushButton("全自动跑图")
        self.auto_run_button.setFixedSize(250, 40)  # 固定宽度250px，高度40px
        self.auto_run_button.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 20px;  /* 圆角半径为高度的一半，形成圆角矩形 */
                font-size: 14px;
            }
        """)
        self.auto_run_button.clicked.connect(self.run_request_program)
        button_layout.addWidget(self.auto_run_button, alignment=Qt.AlignCenter)

        button_layout.addSpacerItem(self.horizontal_spacer_2)

        # 设置 overlay_widget 的布局
        overlay_widget.setLayout(button_layout)

        # 将 overlay_widget 添加到主布局
        main_layout.addWidget(overlay_widget)

        self.setLayout(main_layout)

        self.particles = [
            Particle(
                random.randint(0, self._width),
                random.randint(0, self._height),
                random.uniform(0, 2 * math.pi),
                random.randint(6, 30),
                random.uniform(0.01, 0.06),
                QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            )
            for _ in range(random.randint(15, 30))
        ]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_particles)
        self.timer.start(16)  # 每 16 毫秒更新一次（约 60 FPS）

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

    def update_particles(self):
        for particle in self.particles:
            particle.update()
        self.update()

    def mousePressEvent(self, event):
        for particle in self.particles:
            distance = math.sqrt((particle.x - event.x())**2 + (particle.y - event.y())**2)
            if distance < 100:  # 点击范围内的粒子触发扩散
                particle.trigger_expansion(event.x(), event.y())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制粒子
        points = []
        for particle in self.particles:
            painter.setBrush(particle.color)
            painter.drawEllipse(QPointF(particle.x, particle.y), particle.size, particle.size)
            points.append((particle.x, particle.y))

        # 进行三角剖分
        if len(points) > 2:
            tri = Delaunay(points)
            for simplex in tri.simplices:
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]

                # 计算边的长度并仅保留长度小于一定值的边
                edges = [
                    (p1, p2, math.hypot(p1[0] - p2[0], p1[1] - p2[1])),
                    (p2, p3, math.hypot(p2[0] - p3[0], p2[1] - p3[1])),
                    (p3, p1, math.hypot(p3[0] - p1[0], p3[1] - p1[1]))
                ]

                for edge in edges:
                    if edge[2] < 50:  # 仅绘制长度小于一定值的边
                        painter.setPen(QColor(200, 200, 200))
                        painter.drawLine(QPointF(edge[0][0], edge[0][1]), QPointF(edge[1][0], edge[1][1]))

def main():
    app = QApplication(sys.argv)
    widget = ParticleWidget()
    widget.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
