import os
os.environ['OMP_NUM_THREADS'] = '1'

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
from tqdm import tqdm
from torch import nn
import math
import traceback

from utils import MoENavigationModel

# 重新制作数据集
class NavigatorDataset(torch.utils.data.Dataset):
    def __init__(self, path=os.path.join("output")):
        super(NavigatorDataset, self).__init__()

        self.root_dir = path
        self.code = [(i.split('_')[-1]).split('.')[0] for i in os.listdir(self.root_dir) if i.startswith("output_")][0]

        c = self.code # 只处理一个视频

        video_path = os.path.join(self.root_dir, "output_{}.mp4".format(c))
        feature_path = os.path.join(self.root_dir, "hsv_output_{}.mp4".format(c))
        key_path = os.path.join(self.root_dir, "key_data_{}.txt".format(c))

        self.cap = cv2.VideoCapture(video_path)
        self.fcap = cv2.VideoCapture(feature_path)
        self.fp = open(key_path, 'r', encoding='utf-8')

        self.feature = []
        self.labels = []
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.batch = self.length // 200 + 1
        print("需要重复装载 {} 次".format(self.batch))

        self.idx = 0

        if not self.cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            raise RuntimeError("无法打开视频文件")
        if not self.fcap.isOpened():
            print(f"无法打开视频文件: {feature_path}")
            raise RuntimeError("无法打开视频文件")
        if not self.fp.readable():
            print(f"无法打开标签文件: {key_path}")
            raise RuntimeError("无法打开标签文件")

        _ = self.fp.readline() # 跳过表头

        cache_line = []
        line = self.fp.readline().strip().split(',')
        line = list(map(int, line))
        cache_line.append(line)

        with tqdm(total=200, desc="读取视频帧 {}".format(c), unit='帧') as tbar:
            idx = 0
            while idx < 200:   # 避免一次装载过多
                ret, frame = self.cap.read()
                if not ret:
                    print(f"视频读取完成: {video_path}")
                    break

                fret, fframe = self.fcap.read()
                if not fret:
                    print(f"视频读取完成: {feature_path}")
                    break

                line = self.fp.readline().strip().split(',')
                if line and len(line) == 11:
                    line = list(map(int, line))

                    cache_line.append(line)
                    cache_line = cache_line[-3:] # 只保留最近的三行

                # 随机归类
                if len(cache_line) <= 2:
                    self.feature.append([frame, fframe, [0 for _ in range(11)]])
                    self.feature.append([frame, fframe, [0 for _ in range(11)]])
                    self.feature.append([frame, fframe, cache_line[0]])

                    self.labels.append([0 for _ in range(11)])
                    self.labels.append(cache_line[0])
                    self.labels.append(cache_line[1])
                else:
                    self.feature.append([frame, fframe, [0 for _ in range(11)]])
                    self.feature.append([frame, fframe, cache_line[0]])
                    self.feature.append([frame, fframe, cache_line[1]])

                    self.labels.append(cache_line[0])
                    self.labels.append(cache_line[1])
                    self.labels.append(cache_line[2])
                    
                idx += 1
                tbar.update(1)

    def reset_batch(self):
        c = self.code # 只处理一个视频

        video_path = os.path.join(self.root_dir, "output_{}.mp4".format(c))
        feature_path = os.path.join(self.root_dir, "hsv_output_{}.mp4".format(c))
        key_path = os.path.join(self.root_dir, "key_data_{}.txt".format(c))

        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'fcap') and self.fcap.isOpened():
            self.fcap.release()
        if hasattr(self, 'fp'):
            self.fp.close()

        self.cap = cv2.VideoCapture(video_path)
        self.fcap = cv2.VideoCapture(feature_path)
        self.fp = open(key_path, 'r', encoding='utf-8')

        self.feature = []
        self.labels = []

        self.idx = 0

        cache_line = []
        line = self.fp.readline().strip().split(',')
        line = list(map(int, line))
        cache_line.append(line)

        with tqdm(total=200, desc="读取视频帧 {}".format(c), unit='帧') as tbar:
            idx = 0
            while idx < 200:   # 避免一次装载过多
                ret, frame = self.cap.read()
                if not ret:
                    print(f"视频读取完成: {video_path}")
                    break

                fret, fframe = self.fcap.read()
                if not fret:
                    print(f"视频读取完成: {feature_path}")
                    break

                line = self.fp.readline().strip().split(',')
                if line and len(line) == 11:
                    line = list(map(int, line))

                    cache_line.append(line)
                    cache_line = cache_line[-3:] # 只保留最近的三行

                # 随机归类
                if len(cache_line) <= 2:
                    self.feature.append([frame, fframe, [0 for _ in range(11)]])
                    self.feature.append([frame, fframe, [0 for _ in range(11)]])
                    self.feature.append([frame, fframe, cache_line[0]])

                    self.labels.append([0 for _ in range(11)])
                    self.labels.append(cache_line[0])
                    self.labels.append(cache_line[1])
                else:
                    self.feature.append([frame, fframe, [0 for _ in range(11)]])
                    self.feature.append([frame, fframe, cache_line[0]])
                    self.feature.append([frame, fframe, cache_line[1]])

                    self.labels.append(cache_line[0])
                    self.labels.append(cache_line[1])
                    self.labels.append(cache_line[2])
                    
                idx += 1
                tbar.update(1)

    def next_batch(self):
        c = self.code # 只处理一个视频

        video_path = os.path.join(self.root_dir, "output_{}.mp4".format(c))
        feature_path = os.path.join(self.root_dir, "hsv_output_{}.mp4".format(c))
        key_path = os.path.join(self.root_dir, "key_data_{}.txt".format(c))

        self.feature = []
        self.labels = []

        self.idx += 1
        if self.idx >= self.batch:
            return False
        
        cache_line = []
        line = self.fp.readline().strip().split(',')
        line = list(map(int, line))
        cache_line.append(line)

        with tqdm(total=200, desc="读取视频帧 {}".format(c), unit='帧') as tbar:
            idx = 0
            while idx < 200:   # 避免一次装载过多
                ret, frame = self.cap.read()
                if not ret:
                    print(f"视频读取完成: {video_path}")
                    break

                fret, fframe = self.fcap.read()
                if not fret:
                    print(f"视频读取完成: {feature_path}")
                    break

                line = self.fp.readline().strip().split(',')
                if line and len(line) == 11:
                    line = list(map(int, line))

                    cache_line.append(line)
                    cache_line = cache_line[-3:] # 只保留最近的三行

                # 随机归类
                if len(cache_line) <= 2:
                    self.feature.append([frame, fframe, [0 for _ in range(11)]])
                    self.feature.append([frame, fframe, [0 for _ in range(11)]])
                    self.feature.append([frame, fframe, cache_line[0]])

                    self.labels.append([0 for _ in range(11)])
                    self.labels.append(cache_line[0])
                    self.labels.append(cache_line[1])
                else:
                    self.feature.append([frame, fframe, [0 for _ in range(11)]])
                    self.feature.append([frame, fframe, cache_line[0]])
                    self.feature.append([frame, fframe, cache_line[1]])

                    self.labels.append(cache_line[0])
                    self.labels.append(cache_line[1])
                    self.labels.append(cache_line[2])
                    
                idx += 1
                tbar.update(1)

        return True

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'fcap') and self.fcap.isOpened():
            self.fcap.release()
        if hasattr(self, 'fp'):
            self.fp.close()

        
    def __getitem__(self, idx):
        return {
            'frame': torch.tensor(self.feature[idx][0], dtype=torch.float32).permute(2, 0, 1), 
            'fframe': torch.tensor(self.feature[idx][1], dtype=torch.float32).permute(2, 0, 1), 
            'fp': torch.tensor(self.feature[idx][2], dtype=torch.float32), 
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
        
    def __len__(self):
        return 200 if self.idx != self.batch - 1 else self.length - (self.batch - 1) * 200
    
# 训练模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MoENavigationModel(
    map_channels=3,    # HSV 通道图（检测蜡烛的红色）
    state_dim=11,      # 运动状态维度（上一时刻 Keyboard）
    image_channels=3,  # RGB 图像
    num_experts=8,     # 专家数量
    top_k=2,           # Top-K 门控
    balance_loss_weight=0.1  # 负载均衡损失权重
).to(device)

MODEL_FOLDER = os.path.join("models")
 
# 为什么要写这个是因为多线程 dataloader 需要这个
if __name__ == "__main__":
    episode = 20
    train_dataset = NavigatorDataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    max_early_stopping, early_stopping = 20, 0
    best_loss = math.inf # 最小化损失所以初始化最大
    batch_losses = []
    
    try:
        for epoch in range(episode):
            optimizer.zero_grad()
            losses = []
            for b in range(train_dataset.batch):
                with tqdm(train_dataloader, desc="train batch{} - epoch{}".format(b, epoch)) as tbar:
                    for i, data in enumerate(tbar):
                        frame = data['frame'].to(device)
                        fframe = data['fframe'].to(device)
                        fp = data['fp'].to(device)

                        label = data['label'].to(device)
                        # print("Label:", label.shape)
                        
                        actions, moe_loss = model(fframe, fp, frame)
                        # print("Actions:", actions.shape)
                        loss = criterion(actions, label) + moe_loss
            
                        loss.backward()
                        optimizer.step()
        
                        losses.append(loss.cpu().item())
                        tbar.set_postfix(loss=np.average(losses))
                
            
                sts = train_dataset.next_batch()
                if not sts:
                    raise RuntimeError("滚动切换下一数据集失败")
                
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
                
            batch_avg_loss = np.average(losses)
            batch_losses.append(batch_avg_loss)

            if batch_avg_loss < best_loss:
                early_stopping = 0
                best_loss = batch_avg_loss
                torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, "navigator_valley_best.pth"))
            else:
                early_stopping += 1
                if early_stopping >= max_early_stopping:
                    raise Exception("达到 EarlyStopping 基准, 训练提前停止")
                    
    except Exception as e:
        traceback.print_exc() # 输出详细异常信息
        if str(e) !=  "达到 EarlyStopping 基准, 训练提前停止":
            print("捕获到异常:", e)
    finally:
        torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, "navigator_valley_last.pth"))
        plt.figure(figsize=(12,3))
        plt.plot(batch_losses)
        plt.title("Train Loss Figure")
        plt.xlabel("epoch")
        plt.ylabel("avg_loss")
        plt.show()