import os
from ultralytics import YOLO
import torch
import argparse
import shutil
import cv2
import numpy as np
import pyautogui
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11 Nano on custom dataset.")
    parser.add_argument('--data', type=str, default=os.path.join("dataset", "sky", "data.yaml"), help='Path to the data configuration file.')
    parser.add_argument('--weights', type=str, default=os.path.join("weights", "yolo11s.pt"), help='Path to the pre-trained weights file.')
    parser.add_argument('--outputs', type=str, default=os.path.join("weights", "sky.pt"), help='Path to save the trained model weights.')
    parser.add_argument('--exp', type=str, default="yolo11n_sky", help='Experiment name for saving results.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--train', action='store_true', help='Flag to train the YOLOv11 Nano model.')
    parser.add_argument('--val', action='store_true', help='Flag to evaluate the YOLOv11 Nano model after training.')
    return parser.parse_args()

def train_yolov11n_model(options=None):
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print(f"CUDA 可用，使用 GPU: {torch.cuda.get_device_name(0)}")
        device = 0  # 使用第一个 GPU (cuda:0)
    else:
        print("CUDA 不可用，将使用 CPU 训练，这可能会很慢！")
        device = "cpu"
    
    path = options.data if options is not None else os.path.join("dataset", "sky", "data.yaml")
    weights = options.weights if options is not None else os.path.join("weights", "yolo11n.pt")
    
    # 确保数据配置文件存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据配置文件不存在: {path}")
    
    # 加载预训练模型
    print("正在加载 YOLOv11 纳米版预训练模型, 模型路径: {}".format(os.path.abspath(weights)))
    model = YOLO(os.path.abspath(weights))  # 加载 YOLOv11 纳米版预训练模型
    
    # 训练模型
    results = model.train(
        data=path,                  # 数据配置文件路径
        imgsz=640,                  # 输入图像尺寸
        epochs=options.epochs,      # 训练轮次
        batch=options.batch_size,   # 批次大小
        device=device,              # 指定设备
        project="sky_detection",    # 项目名称
        name=options.exp,           # 实验名称
        pretrained=True,            # 使用预训练权重
        optimizer="Adam",           # 优化器
        lr0=1e-3,                   # 初始学习率
        workers=8,                  # 数据加载工作线程数
        cache=True                  # 缓存图像以加速训练
    )
    
    # 输出训练结果
    print(f"训练完成，结果保存在: {results.save_dir}")
    return model
    
def evaluate_yolov11n_model(model=None, options=None):
    if model is None:
        # 加载训练好的模型
        model = YOLO(options.outputs)  # 加载训练好的模型
    
    # 确保模型加载成功
    if model is None:
        raise RuntimeError("模型加载失败，请检查路径和文件是否正确。")
    print("模型加载成功，开始评估...")
    
    if options is not None and options.val:
        # 评估模型性能
        metrics = model.val()
        print(f"mAP@0.5: {metrics.box.map50}")
        print(f"mAP@0.5:0.95: {metrics.box.map}")
    
    # 测试模型 :)
    results = model.predict(os.path.join("..", "data", "source", "scshots"), save=True)
        
    return model

def run_yolov11n_model(model, region=None, display_scale=0.5):
    # 获取屏幕尺寸
    screen_width, screen_height = pyautogui.size()
    
    # 如果未指定区域，则使用全屏
    if region is None:
        region = (0, 0, screen_width, screen_height)
    
    # 创建显示窗口
    cv2.namedWindow("YOLOv11 Screen Detector", cv2.WINDOW_NORMAL)
    
    fps_list = []
    start_time = time.time()
    
    while True:
        # 捕获屏幕
        screenshot = pyautogui.screenshot(region=region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 使用模型进行检测
        results = model(frame, verbose=False)
        
        # 获取检测结果
        annotated_frame = results[0].plot()
        
        # 计算 FPS
        current_time = time.time()
        fps = 1.0 / (current_time - start_time)
        fps_list.append(fps)
        avg_fps = sum(fps_list[-30:]) / min(len(fps_list), 30)
        start_time = current_time
        
        # 在画面上显示 FPS
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 调整显示尺寸
        display_width = int(annotated_frame.shape[1] * display_scale)
        display_height = int(annotated_frame.shape[0] * display_scale)
        display_frame = cv2.resize(annotated_frame, (display_width, display_height))
        
        # 显示结果
        cv2.imshow("YOLOv11 Screen Detector", display_frame)
        
        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 关闭所有窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 获取命令行参数
    options = parse_args()
    
    # 定义捕获区域 (left, top, width, height)
    # 例如，捕获屏幕中央 800x600 的区域：capture_region = (400, 300, 800, 600)
    capture_region = None  # 全屏捕获
    
    model = None
    # 是否训练模型
    if options.train:
        model = train_yolov11n_model(options=options)
        num_batch = max([int(dirs[len('yolo11n_sky'):]) if len(dirs[len('yolo11n_sky'):])>0 else 1 for dirs in os.listdir("sky_detection")])
        shutil.copy(os.path.join("sky_detection", "yolo11n_sky{}".format(num_batch if num_batch != 1 else ''), "weights", "best.pt"), options.outputs)
    else:
        print("Skipping training, only evaluating the model.")
    
    # 评估模型
    if model is not None:
        model = evaluate_yolov11n_model(model=model, options=options)
    else:
        model = evaluate_yolov11n_model(options=options)
    
    # 运行屏幕捕获和检测
    run_yolov11n_model(model, region=capture_region, display_scale=0.7)    