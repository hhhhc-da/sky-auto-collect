# coding=utf-8
import os
os.environ["QT_SCALE_FACTOR"] = "1.0"
os.environ['OMP_NUM_THREADS'] = '1'

import math
import time
from PySide6.QtCore import Signal
import cv2
import pyautogui

import pydirectinput
pydirectinput.FAILSAFE = False

import numpy as np
import matplotlib.pyplot as plt
import pyperclip
import pandas as pd
import yaml
from tqdm import tqdm

from module.base import BaseThread
from module.deploy import NanokaDetector
from module.web import WebCrawler

# 爬虫线程 
class CrawlerProgramThread(BaseThread):
    finished = Signal()
    
    def __init__(self, sigma=0.07, yaml='config.yaml'):
        super().__init__(sigma=sigma)
        
        self.sigma = sigma
        self.yaml_path = yaml
        
        self.detector = NanokaDetector(yaml_path=self.yaml_path, paddle_ocr_on=True, yolo_on=True)
        self.crawler = WebCrawler()
        
    def goto_page(self, page=0) -> bool:
        '''
        跳转到指定的页数, 重写虚函数跳转到指定页面
        '''
        self.mouse_clear()  # 清除鼠标位置，避免干扰检测
        
        timeout = 50
        while timeout > 0:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            self.detector.update_image(frame)  # 更新检测器的图像数据
            
            timeout_detector = 3  # 每次检测最多尝试 3 次
            while timeout_detector > 0:
                try:
                    code = self.check_page()
                    if code in [0, 1]: # 如果是添加好友页或者在星盘内
                        break
                except Exception as e:
                    print("捕捉到 Exception:", e)
                    timeout_detector -= 1
                   
            if code != 1:
                timeout -= 1
                
                # 不停向左检测图片
                self.last_page()
                continue
            
            break
        
        if timeout <= 0:
            print("未检测到添加好友页")
            return False

        # 跳转到指定页面
        for _ in range(page):
            self.faster_next_page()
        
        return True
        
    def check_page(self) -> int:
        '''
        检查当前页面是否是正常星盘页, 纯净版, 这一部分再次被重写, 将来可能会并回主线
        '''
        code = None

        text = self.detector.ocr_detector()
        pf = self.detector.re_keyword_detector([text])
        code = None
    
        if not bool(pf['星盘页'].values[0]):
            code = -1
        elif bool(pf['添加好友'].values[0]):
            code = 1
        elif bool(pf['好友'].values[0]) or bool(pf['挚友'].values[0]):
            code = 0
            
        return code
    
    def check_danger(self) -> bool:
        '''
        检查是否存在危险按钮，这里主要是检测红色区域 (255,63,52) 附近，一般红色是满的 255
        '''
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        
        # 检测红色区域
        red_mask = cv2.inRange(frame, (250,58,48), (255,65,55))
        red_area = cv2.countNonZero(red_mask)
        
        if red_area > 50: # 超过 50 个经典像素点就不行
            return True
        else:
            return False
        
    def check_transfer(self) -> bool:
        '''
        检测我们是否正确的被传送进房间内，我们可能遇到传送不进去的情况，这种情况我们要反复尝试但是要加一个计时器
        成功进去返回 True，失败返回 False
        '''
        # 检测画面是否为几乎全黑色或几乎全白色
        s = time.time()
        
        while time.time() - s < 2:  # 最多等待十秒钟
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算图像的平均亮度
            mean_brightness = np.mean(gray_frame)
            
            if mean_brightness < 10 or mean_brightness > 245:
                return True, 0
            
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        
        # 两秒的时候截取最下面是否存在 "您的朋友就在您身边"
        boarder_height = int(frame.shape[0] * 0.1) # 最下面 10% 的区域
        text_area = frame[-boarder_height:, :, :]
        
        # 检测是否存在 "您的朋友就在您身边" 的提示
        paddle_ocr_result = self.detector.paddle_ocr.predict([text_area])
        text = ''
        if (paddle_ocr_result is not None) and (len(paddle_ocr_result) > 0):
            text = ''.join(paddle_ocr_result[0]['rec_texts'])
        
        if '朋友' in text or '身边' in text:
            print("检测到已传送成功，您的朋友就在您身边")
            return True, 1
            
        while time.time() - s < 10:  # 最多等待十秒钟
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算图像的平均亮度
            mean_brightness = np.mean(gray_frame)
            
            if mean_brightness < 10 or mean_brightness > 245:
                return True, 0
            
        return False, -1  # 超过时间限制仍未检测到有效传送，返回 False
    
    def add_friend(self, friend_code='', friend_name='') -> None:
        '''
        使用好友码加好友，加完好友之后就可以寻找对应位置了
        '''
        self.press_key(key='esc', wait_time=3)
        
        pydirectinput.keyDown('up')
        time.sleep(0.1)
        pydirectinput.keyUp('up')
        self.gauss_sleep(0.1)
        
        for _ in range(8):
            pydirectinput.keyDown('left')
            time.sleep(0.1)
            pydirectinput.keyUp('left')
            self.gauss_sleep(0.1)  # 向左移到最左边选框
            
        self.gauss_sleep(0.3)
        
        for _ in range(3):
            pydirectinput.keyDown('right')
            time.sleep(0.1)
            pydirectinput.keyUp('right')
            self.gauss_sleep(0.1)
            
        self.press_key('space', wait_time=1)  # 进入好友码添加好友页
        
        # 直接将好友码放入复制缓冲区内
        pyperclip.copy(friend_code)
        self.gauss_sleep(0.3)
        pydirectinput.keyDown('ctrl')  # 粘贴好友码
        pydirectinput.keyDown('v')
        self.gauss_sleep(0.1)
        pydirectinput.keyUp('v')
        pydirectinput.keyUp('ctrl')
        self.gauss_sleep(0.5)  # 等待粘贴完成
        
        # 按下回车键确认
        self.press_key('enter', wait_time=1)
        
        # 这是一组好友邀请编号！（右下角接受）
        self.press_key('down', wait_time=0.2)
        self.press_key('right', wait_time=0.2)
        self.press_key('space', wait_time=1)  # 确认添加好友
        
        pyperclip.copy(friend_name)
        self.gauss_sleep(0.3)
        pydirectinput.keyDown('ctrl')  # 粘贴好友码
        pydirectinput.keyDown('v')
        self.gauss_sleep(0.1)
        pydirectinput.keyUp('v')
        pydirectinput.keyUp('ctrl')
        self.gauss_sleep(0.5)  # 等待粘贴完成
        
        # 按下回车键确认添加好友
        self.press_key('enter', wait_time=1)
        
        # 恭喜你有了新的朋友
        self.press_key('down', wait_time=0.2)
        self.press_key('space', wait_time=1)  # 退出添加好友页
    
    def search_friend_name(self, find_target='', plot=False, show=False) -> None:
        '''
        寻找对应名字的好友的星星位置, 这里就开始操控了
        '''
        # 首先对准星盘使用, 按下 g 进入星盘页
        self.search_star_board()
        self.press_key(key='g', wait_time=4)
        
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
            code = self.check_page()
            if code == 1:
                print("检测到添加好友页，退出程序")
                break
            
            try:
                # 开始解析需要传送的好友信息
                pf = self.detector.text_detector()
                
                target, star_pos, text_info = find_target, None, []
                for i, name in enumerate(pf['result'].values):
                    similarity = self.detector.text_simularity(ocr_text=name, target=target, min_gap=0, max_gap=3)
                    print(f"Text: {name}, Similarity to '{target}': {similarity:.2f}")
                    
                    if similarity > 0.95: # 高于这个阈值的都被拿出来考量一下
                        xywh = eval(pf[pf['result'] == name]['position'].values[0]) # xywh
                        text_info.append([name, xywh, similarity])
                  
                if len(text_info) == 0:
                    print("没有找到好友，开始寻找下一页")
                    continue  # 如果没有找到对应的好友就跳过这次循环
                
                # 对文本信息按照相似度进行排序
                sorted(text_info, key=lambda x: x[2], reverse=True)
                print(f"找到相似的名称如下:\n{text_info}")
                        
                dc, img = None, None
                timeout_detector = 3  # 每次检测最多尝试 3 次
                while timeout_detector > 0:
                    try:
                        dc, img = self.detector.multi_detector(plot=plot, show=show)
                        if dc is not None and dc['code'] in [0, 1]:
                            break
                    except Exception as e:
                        print(f"检测失败: {e}")
                        timeout_detector -= 1
                        self.gauss_sleep(0.3) # 等待下一次检测, 每约 0.3s 检测一次
                        continue

                if dc['code'] != 0:
                    print(f"Error: dc = {dc['info']}, 不满足前置条件")
                    continue
                
                points = []
                for dp in [dc["hearts_info"], dc["pre_points"], dc["post_points"]]:
                    points.extend(dp)
                
                closest_star_pos, text_pos = None, None
                min_distance = float('inf')
                
                # 逐个检测星星中心与文本中心的距离, 取出距离最近的那个点
                for i, (name, xywh, similarity) in enumerate(text_info):
                    closest_star_pos = None
                    min_distance = float('inf')
                    
                    for x, y, w, h in points:
                        # 不能只看是不是离得近，还要看是否在文本框内
                        if (xywh[0] < x < xywh[0]+xywh[2]) and (xywh[1] < y < xywh[1]+xywh[3]):
                            print("找到文本框内的无效点, 本点不作为有效点，跳过")
                            continue
                        
                        star_pos = [x, y]
                        distance = math.sqrt((star_pos[0] - xywh[0])**2 + (star_pos[1] - xywh[1])**2)
                        
                        if distance < min_distance:
                            closest_star_pos = star_pos
                            min_distance = distance
                            
                    if min_distance < 2 * math.sqrt(xywh[2]**2 + xywh[3]**2): # 校验
                        print(f"找到好友 {name} 的星星位置: {closest_star_pos}, 文本相似度: {similarity:.2f}, 距离: {min_distance:.2f}")
                
                        # 找到点了之后开始移动即可
                        self.goto_meet(star_pos=closest_star_pos)
                        return 
                
            except Exception as e:
                print(f"捕捉到 Exception: {e}")
                break
             
    
    def goto_meet(self, star_pos=None) -> None:
        '''
        传送到对应房间并接收爱心
        '''        
        timeout = 3
        while timeout > 0:
            if self.check_text(): # 我们现在应该在星盘页才对
                self.move_mouse(x=star_pos[0], y=star_pos[1], wait_time=2)  # 移动鼠标到目标中心, 考虑了屏幕分辨率的影响
                break
            else:
                print("检测到我们不在星盘页需要重新定位") # 由于是自动程序控制所以不用担心在这里卡死
                timeout -= 1
                
                self.press_key('g', wait_time=2)
                self.goto_page(self.page)
                continue
        if timeout == 0:
            raise RuntimeError("无法定位到星盘页，请检查程序运行状态")
        
        # 这里有一个分歧, 如果检测出来没有文字了那么我们就不再点一次了
        if self.check_text():
            pydirectinput.click()  # 点击目标， 这次一定进入大屏星盘页
            self.gauss_sleep(3)
        
        for _ in range(8):
            pydirectinput.keyDown('up')
            time.sleep(0.12)
            pydirectinput.keyUp('up')
            self.gauss_sleep(0.2)  # 向上移动 8 次
            
        self.gauss_sleep(0.4)
        
        for _ in range(3):
            pydirectinput.keyDown('down')
            time.sleep(0.12)
            pydirectinput.keyUp('down')
            self.gauss_sleep(0.2)  # 向上移动 8 次
            
        self.gauss_sleep(0.4)  # 等待响应
            
        # 如果这里意外颠倒了其他按钮, 尤其是删除或拉黑好友, 我们要做出应急响应
        self.press_key('space', wait_time=2)
        
        code = -1
        # 校验是否存在"屏蔽按钮"（实际上校验红色区域即可），校验方法为十分严格的 RGB 检测
        if not self.check_danger():
            # 进入之后只要你不乱动就是没问题的
            self.press_key('space', wait_time=0.1) # 进入房间
            
            s = time.time()
            while True: # 阻塞校验是否进入, 最多循环检测十分钟
                transfer_success, code = self.check_transfer()
                if transfer_success:
                    if code == 1:
                        self.press_key('esc', wait_time=1)
                        self.press_key('esc', wait_time=3)
                        break
                    elif code == 0:
                        break
                
                if not transfer_success and time.time() - s < 600:
                    self.press_key('space', wait_time=0.1)
                
            if time.time() - s >= 600:
                raise RuntimeError("传送异常，可能是传送失败或未进入房间，请检查程序运行状态")
        else:
            self.press_key('esc', wait_time=0.1) # 加入我们点错了, 我们这一步一定不能执行
            raise RuntimeError('检测到有危险行为不能确定, 请立刻查看')
        
        self.gauss_sleep(3)
        
        # 进来了之后开始寻找周围人，由于进来之后不需要其他操作只需要四处转一圈就可以了
        for i in range(60): # 左旋寻找目标，是离散的
            frame = self.screenshot()
            track_result = self.detector.yolov11s_botsort_track(rgb_image=frame)
            
            if len(track_result['cls']) > 1: # 因为我们只检测一个类型甚至不用看 cls
                '''
                这里我们会检测到多个目标，有以下两种方案
                1.直接选择较远的目标进行对齐，因为我们假设整个遇境只有两个人
                2.使用 BoT-SORTS 进行目标跟踪，锁定后出现的目标（当前使用的方案）
                '''
                print("检测到人物出现，已方向确定")
                
                # 算一下是否有多个 id 以及是否离中心很近（离得近的是我们自己）
                closest_id, closest_xy, closest_distance = -1, None, -float('inf')
                for i in range(len(track_result['xywh'])):
                    if track_result['cls'][i] == 0:
                        x, y = track_result['xywh'].numpy()[i][0], track_result['xywh'].numpy()[i][1] # YOLOv11 直出的坐标不用计算偏移
                        distance = np.sqrt((x - self.width//2)**2 + (y - self.height//2)**2)
                        
                        if distance > closest_distance:
                            closest_distance = distance
                            closest_xy = (x, y)
                            closest_id = i
                # 我想导出成 CPU 模型的但是最后也没导出成功
                distance_x = closest_xy[0] - self.width // 2  # 计算 x 轴偏移量, 这里最好是校验一下是不是 GPU 上
                print('distance_x:', distance_x)
                
                '''
                摄像机旋转角度方程为 y = 1.6592x + b
                也就是说转一圈需要 1.6592s, 这个是有数据的
                数据为 [12.29,13.88,15.46,17.16,18.78,20.74,22.14,23.84,25.56,27.09]
                最小二乘法计算出来参数为 y = 1.6592x + 10.5687
                近似一个相机的视角是 120 度，那么我们可以试着每次按下 40 度
                也就是每次按下 0.1884s，相当于每次转 40 度，检测分度值为 5 度
                1920px 像素相当于 120 度，那么我们简单的计算一下 r * sin θ = px * k
                其中，r 是物体在光遇内的距离， k 是像素到角度的转换系数
                那么 θ = arcsin(px * k / r)，而 k / r = sin(θ) / px = sin(π/3) / 960
                也就是 t = 1.6592/2π * (arcsin(px * sin(π/3) / 960))
                最后乘一个系数，相机有一个加速的过程，很奇怪，所以这样对齐一下
                '''
                spin_time = 1.5 * 1.6592 / (2 * math.pi) * math.asin(abs(distance_x) * math.sin(5 * math.pi / 12) / 960)  # 计算旋转时间
                
                print(f"计算出的旋转时间: {spin_time:.2f} 秒")
                if distance_x > 0:  # 如果偏移量大于 0，说明需要向右转
                    pydirectinput.keyDown('right')
                    self.gauss_sleep(spin_time)
                    pydirectinput.keyUp('right')
                elif distance_x < 0:  # 如果偏移量小于 0，说明需要向左转
                    pydirectinput.keyDown('left')
                    self.gauss_sleep(spin_time)
                    pydirectinput.keyUp('left')
                    
                if i < 5:
                    self.gauss_sleep(2) # 也别太快了
                break
            
            pydirectinput.keyDown('left')
            self.gauss_sleep(0.1884) # 随机慢速旋转
            pydirectinput.keyUp('left')
            
        # 使用 SIFT 检测是否存在 
        timeout = 3
        while timeout > 0:
            frame = self.screenshot()
            # sift_matches 匹配结果: [([x, y], cls), ...]
            sift_matches = self.detector.analyze_receive_button_by_sift(frame)
            
            if len(sift_matches) > 0:
                print("检测到送心员行为，开始接收爱心")
                
                if code == 0:
                    # 拍手
                    self.press_key('f', wait_time=0.3)
                    self.press_key('down', wait_time=0.1)
                    self.press_key('right', wait_time=0.1)
                    self.press_key('space', wait_time=7)
                    # 送心
                    self.press_key('f', wait_time=0.3)
                    self.press_key('down', wait_time=0.1)
                    self.press_key('right', wait_time=0.1)
                    self.press_key('space', wait_time=2)
                    
                elif code == 1:
                    # 送他一个击掌
                    self.press_key('f', wait_time=0.3)
                    
                    for _ in range(8):
                        pydirectinput.keyDown('down')
                        time.sleep(0.12)
                        pydirectinput.keyUp('down')
                        self.gauss_sleep(0.1)  # 向下移动 8 次
                        
                    for _ in range(2):
                        pydirectinput.keyDown('up')
                        time.sleep(0.12)
                        pydirectinput.keyUp('up')
                        self.gauss_sleep(0.1)  # 向下移动 8 次
                        
                    self.press_key('space', wait_time=1) # 送他一个击掌
                    
                    self.press_key('down', wait_time=0.1)
                    self.press_key('right', wait_time=0.1)
                    self.press_key('space', wait_time=9)
                    
                    # 送心
                    self.press_key('f', wait_time=0.3)
                    self.press_key('down', wait_time=0.1)
                    self.press_key('right', wait_time=0.1)
                    self.press_key('space', wait_time=2)
                
                pydirectinput.keyDown('down') # 定位回星盘去
                self.gauss_sleep(2)
                pydirectinput.keyUp('down')
                return
            
            else:
                timeout -= 1
                self.gauss_sleep(3) # 等待约 10s 没有就报错就可以了
                continue
        if timeout == 0:
            raise RuntimeError("无法检测到送心员行为，这是未定义的行为")
    
    def run(self):
        '''
        请求爱心主线程，用来从行为上控制程序
        '''
        data = None
        with open(self.yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            file.close()
        
        pf = pd.read_excel(data['file'], sheet_name="Sheet1", names=["url"])
        crawler = WebCrawler()
        
        for epoch in range(data['episode']):
            ts = time.time()
            print(f"第 {epoch + 1} 轮取心")
            for index, row in pf.iterrows():
                target_url = row['url']
                flag = crawler.crawl_main(target_url, headless=False, valid=False) # 记得一定要关闭验证模式
                if not flag:
                    print(f"第 {epoch + 1} 轮取心失败，说明出现了问题")
                    break 
                
                if crawler.code is not None:
                    print(f"识别到好友码: {crawler.code}")
                    
                    friend_name = "送心员{}".format(''.join([self.detector.chinese_map[int(ch)] for ch in str(data['index'])]))
                    print(pd.DataFrame([[friend_name, crawler.code]], columns=['name', 'code']))
                    data['index'] += 1
                    
                    with open(self.yaml_path, 'w+', encoding='utf-8') as file:
                        yaml.dump(data, file, allow_unicode=True, sort_keys=True)
                        file.close()
                        
                    self.add_friend(friend_code=crawler.code, friend_name=friend_name)
                    print(f"添加好友 {friend_name} 成功")
                        
                    self.search_friend_name(find_target=friend_name)
                        
                time.sleep(3)
                
            if epoch != data['episode'] - 1:
                delay_time = 5
                # 这里多加几秒的延迟响应时间, 同时这一轮最少也要等这么久，别响应太快了
                with tqdm(total=data['delay'] + delay_time, desc=f"Time", unit='s') as pbar:
                    diff = time.time() - ts
                    pbar.update(int(diff))  # 更新进度条为当前时间
                    for i in range(int(data['delay'] - int(diff) + delay_time)):
                        time.sleep(1)
                        pbar.update(1)
                        
        print("CrawlerProgramThread 线程已结束")
        self.finished.emit()

# 收集线程
class HeartProgramThread(BaseThread):
    finished = Signal()
    
    def __init__(self, sigma=0.07, yaml='config.yaml'):
        super().__init__(sigma=sigma)
        
        self.sigma = sigma
        self.yaml_path = yaml
        
        self.detector = NanokaDetector(yaml_path=self.yaml_path)
        
    # override
    def goto_page(self, page=0) -> bool:
        '''
        跳转到指定的页数, 重写虚函数
        '''
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
                   
            if pdata is not None and pdata['code'] != 1:
                timeout -= 1
                
                # 不停向左检测图片
                self.last_page()
                continue
            
            break
        
        if timeout <= 0:
            print("未检测到添加好友页")
            return False

        # 跳转到指定页面
        for _ in range(page):
            self.faster_next_page()
        
        return True
        
    def check_page(self, plot=False, show=False) -> tuple:
        '''
        检查当前页面是否是正常星盘页, 但是使用 multi_detector 进行操作
        '''
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
            self.press_key('esc', wait_time=0.1)
            return True, None, None
            
        if img is not None:
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(os.path.join('runs', 'predict', f'page{self.page}.png'))
            plt.close()
            
        return False, pdata, img
    
    def receive_hearts(self, hearts_info) -> None:
        '''
        接收爱心的函数, 完全是操纵行为
        '''
        for i, (x, y, w, h) in enumerate(hearts_info):
            # 移动鼠标到目标中心, 考虑了屏幕分辨率的影响
            timeout = 3
            while timeout > 0:
                if self.check_text(): # 我们现在应该在星盘页才对
                    self.move_mouse(x=x, y=y, wait_time=2)  # 移动鼠标到目标中心
                    break
                else:
                    print("检测到我们不在星盘页需要重新定位") # 由于是自动程序控制所以不用担心在这里卡死
                    timeout -= 1
                    
                    self.press_key('g', wait_time=2)  # 按下 g 键回到星盘页
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
                time.sleep(0.12)
                pydirectinput.keyUp('up')
                self.gauss_sleep(0.1)  # 向上移动 8 次
                
            self.gauss_sleep(0.4)
                
            for _ in range(2):
                pydirectinput.keyDown('down')
                time.sleep(0.12)
                pydirectinput.keyUp('down')
                self.gauss_sleep(0.1) 
                
            self.gauss_sleep(0.4)  # 等待响应
                
            # 如果这里意外颠倒了其他按钮, 尤其是删除或拉黑好友, 我们要做出应急响应
            self.press_key('space', wait_time=0.5) 
            
            # 加入我们点错了, 我们这一步不能回到星盘页
            self.press_key('esc', wait_time=0.1)
            
            wait_responce_sec = 2.0
            self.gauss_sleep(wait_responce_sec) # 等待页面加载完成, 这一步一定要等待充分
            
            if not self.check_text():
                print("检测到我们没回到星盘, 存在按钮误操作的可能性, 进行应急处理")
                self.press_key('esc', wait_time=wait_responce_sec)
                
    def receive_stars(self, post_points, labels) -> None:
        '''
        接收心火的函数, 完全是操纵行为
        '''
        for i, ((x, y, w, h), cls) in enumerate(zip(post_points, labels)):
            # 星屑检测结果, 同时过滤鼠标效果
            if int(cls) == 0:
                # 第一轮判断
                timeout = 3
                while timeout > 0:
                    if self.check_text(): # 我们现在应该在星盘页才对
                        self.move_mouse(x=x, y=y, wait_time=0.8) # 进入到了详情页
                        break
                    else:
                        print("检测到我们不在星盘页需要重新定位") # 由于是自动程序控制所以不用担心在这里卡死
                        timeout -= 1
                        
                        self.press_key('g', wait_time=2)  # 按下 g 键回到星盘页
                        self.goto_page(self.page)
                        continue
                if timeout == 0:
                    raise RuntimeError("无法定位到星盘页，请检查程序运行状态")
                
                # 第二轮判断, 如果进去了就出来
                if not self.check_text(): # 我们现在应该在星盘页才对, 如果不在那就是在详情页里, 那我们就回退
                    self.press_key('esc', wait_time=0.8)
                if not self.check_text():
                    self.press_key('g', wait_time=2.5)
                    self.goto_page(self.page)  # 恢复到当前页
                if not self.check_page():
                    raise RuntimeError("无法定位到星盘页，请检查程序运行状态")
    
    def give_stars(self, pre_points) -> None:
        '''
        送出心火的函数, 完全是操纵行为
        '''
        # 逐个处理检测到的目标
        for i, (x, y, w, h) in enumerate(pre_points):
            # 第一轮判断
            timeout = 3
            while timeout > 0:
                if self.check_text(): # 我们现在应该在星盘页才对
                    self.move_mouse(x=x, y=y, wait_time=1.5)  # 移动鼠标到目标中心, 考虑了屏幕分辨率的影响
                    break
                else:
                    print("检测到我们不在星盘页需要重新定位") # 由于是自动程序控制所以不用担心在这里卡死
                    timeout -= 1
                    
                    self.press_key('g', wait_time=2)  # 按下 g 键回到星盘页
                    self.goto_page(self.page)
                    continue
            if timeout == 0:
                raise RuntimeError("无法定位到星盘页，请检查程序运行状态")
            
            if self.check_text(): # 没进去再点一下
                self.press_key('space', wait_time=1.5)
            if not self.check_text(): 
                self.press_key('f', wait_time=0.5)
                self.press_key('esc', wait_time=0.5)
    
    def run(self):
        '''
        赠送心火主线程，用来从行为上控制程序
        '''
        # 首先对准星盘使用, 按下 g 进入星盘页
        self.search_star_board()
        self.press_key('g', wait_time=4)
        
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
        
        print("HeartProgramThread 线程已结束")
        self.finished.emit()
    