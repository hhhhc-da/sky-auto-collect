#coding=utf-8
import cv2
import os
import shutil
from pathlib import Path

def extract_frames(video_path, output_folder="frames"):
    """
    从视频中提取所有帧并保存到指定文件夹
    
    参数:
        video_path (str): 输入视频的路径
        output_folder (str): 输出帧的文件夹路径，默认为"frames"
    """
    # 创建或清空输出文件夹
    Path(output_folder).mkdir(exist_ok=True)
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'无法删除 {file_path}, 原因: {e}')
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return
    
    # 获取视频帧率和总帧数（可选）
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频帧率: {fps} FPS")
    print(f"总帧数: {total_frames}")
    
    frame_count = 0
    success, frame = cap.read()
    
    while success:
        # 保存当前帧
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        # 读取下一帧
        frame_count += 1
        success, frame = cap.read()
    
    # 释放资源
    cap.release()
    print(f"成功提取 {frame_count} 帧到 {output_folder} 文件夹")

if __name__ == "__main__":
    # 指定要处理的视频文件路径
    filenames = os.listdir('videos')
    filenames = [f for f in filenames if f.endswith(('.mp4'))]  # 只处理视频文件
    
    if not filenames:
        print("错误: 'videos' 文件夹中没有视频文件")
        exit(1)
        
    for i, filename in enumerate(filenames):
        video_file = os.path.join('videos', filename)
        
        # 检查视频文件是否存在
        if not os.path.exists(video_file):
            print(f"错误: 视频文件 {video_file} 不存在")
        else:
            extract_frames(video_file, output_folder=os.path.join('frames', f"frames_{i+1}_{filename[:-4]}"))    