import cv2
import numpy as np
import pyautogui
import time
from pynput import keyboard
import uuid
import os

# 全局变量
key_states = {key: 0 for key in ['w', 'a', 's', 'd', 'f', 'g', 'up', 'down', 'left', 'right', 'space']}
recording = True  # 控制录制状态

# 键盘监听线程
def on_press(key):
    try:
        if key.char in key_states:
            key_states[key.char] = 1
    except AttributeError:
        if key == keyboard.Key.up:
            key_states['up'] = 1
        elif key == keyboard.Key.down:
            key_states['down'] = 1
        elif key == keyboard.Key.left:
            key_states['left'] = 1
        elif key == keyboard.Key.right:
            key_states['right'] = 1
        elif key == keyboard.Key.space:
            key_states['space'] = 1

def on_release(key):
    try:
        if key.char in key_states:
            key_states[key.char] = 0
    except AttributeError:
        if key == keyboard.Key.up:
            key_states['up'] = 0
        elif key == keyboard.Key.down:
            key_states['down'] = 0
        elif key == keyboard.Key.left:
            key_states['left'] = 0
        elif key == keyboard.Key.right:
            key_states['right'] = 0
        elif key == keyboard.Key.space:
            key_states['space'] = 0

    # 停止录制的组合按键：Ctrl + Shift + Q
    if key == keyboard.Key.esc:
        global recording
        recording = False
        return False

# 录制屏幕和键盘数据
def record_screen_and_keys(video_filename, txt_filename):
    global recording

    # 初始化视频写入
    screen_size = pyautogui.size()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, screen_size)

    # 打开键盘数据文件
    with open(txt_filename, 'w') as txt_file:
        txt_file.write("w,a,s,d,f,g,up,down,left,right,space\n")  # 写入表头

        print("录制将在3秒后开始...")
        time.sleep(3)  # 延迟3秒
        print("开始录制！按下 ESC 停止录制。")

        while recording:
            img = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            out.write(frame)

            # 写入键盘状态
            key_data = ','.join(str(key_states[key]) for key in key_states)
            txt_file.write(key_data + '\n')

    # 释放资源
    out.release()
    print("录制结束，文件已保存。")

# 主函数
if __name__ == "__main__":
    BASE_DIR = os.path.join('navigator', 'output')
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    id = uuid.uuid4().hex[:6]
    video_filename = os.path.join(BASE_DIR, f"output_{id}.mp4")
    txt_filename = os.path.join(BASE_DIR, f"key_data_{id}.txt")

    # 启动键盘监听线程
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # 开始录制屏幕和键盘数据
    record_screen_and_keys(video_filename, txt_filename)

    # 等待键盘监听线程结束
    listener.join()