import cv2
import os
import numpy as np
from tqdm import tqdm

def hue_detector(hsv_image, kernel_hue=0, hue_threhold=15):
    hue_ranges = []
    
    # 计算色相环, 其他阈值卡死为 30
    if hue_threhold > kernel_hue:
        hue_ranges.append([180 + kernel_hue - hue_threhold, 179])
        hue_ranges.append([0, hue_threhold + kernel_hue - 1])
    else:
        hue_ranges.append([kernel_hue - hue_threhold, hue_threhold + kernel_hue - 1])
    
    height, width = hsv_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if hue_threhold > 5:
        hue_masks = [cv2.inRange(hsv_image, np.array([lower_hue, 128, 100]), np.array([upper_hue, 255, 255]))for (lower_hue, upper_hue) in hue_ranges]
        for hue_mask in hue_masks:
            mask = cv2.bitwise_or(mask, hue_mask)
    return mask

def process_video_to_hsv(input_path, output_path):
    # 打开输入视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_path}")
        return

    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 初始化输出视频
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"正在处理视频: {input_path}")
    with tqdm(total=frame_count, desc="处理进度", unit="帧") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为 HSV 图像
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 应用色相检测
            mask = hue_detector(hsv_frame)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # 写入输出视频
            out.write(mask)

            # 更新进度条
            pbar.update(1)

    # 释放资源
    cap.release()
    out.release()
    print(f"处理完成，输出文件: {output_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.join('navigator', 'output')
    if not os.path.exists(BASE_DIR):
        print(f"目录不存在: {BASE_DIR}")
        exit()

    # 遍历 BASE_DIR 中的所有 MP4 文件
    for filename in os.listdir(BASE_DIR):
        if filename.endswith(".mp4") and filename.startswith("output_"):
            input_path = os.path.join(BASE_DIR, filename)
            # 提取 ID 并生成输出文件名
            file_id = filename.split("_")[1].split(".")[0]
            output_filename = f"hsv_output_{file_id}.mp4"
            output_path = os.path.join(BASE_DIR, output_filename)

            # 处理视频
            process_video_to_hsv(input_path, output_path)