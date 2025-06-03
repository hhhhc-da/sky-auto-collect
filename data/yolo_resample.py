import os
from ultralytics import YOLO
import torch
import argparse
import shutil
import cv2
import numpy as np
import pyautogui
import time

def evaluate_yolov11n_model(model=None, model_path=os.path.abspath(os.path.join('skt.pt')), match_class=['candle', 'candle_important', 'candle_season', 'file_trace']):
    if model is None:
        # 加载训练好的模型
        model = YOLO(model_path)  # 加载训练好的模型
    
    # 确保模型加载成功
    if model is None:
        raise RuntimeError("模型加载失败，请检查路径和文件是否正确。")
    print("模型加载成功，开始运行程序...")
    
    
    # 测试模型 :)
    results = model.predict(os.path.join("..", "data", "source", "scshots"), save=True)
        
    return model