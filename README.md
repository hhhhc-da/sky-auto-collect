# __光遇全自动跑图程序__

本程序基于计算机视觉与人工智能算法开发，高度类似于人类，仅适配电脑端

### __软件安装__

```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

如果不用 Anaconda 的用户考虑使用下面的命令，同时 PyTorch 已经不需要再手动安装 CUDA 和 cuDNN 了

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

之后安装 requirements.txt 安装好依赖，之后就可以直接运行 Python 程序

```
pip install -r requirements.txt
```

目前版本送心火使用的是 heart\ui.py 我们直接打开它就行，没有其他要求，屏幕分辨率最好是 1920x1080 的

__一定要注意，由于这类自动化程序需要很高的执行权限，所以务必在使用时保持管理员权限__

可以类似于我的启动器（heart\A启动器.bat）编写启动程序，先请求管理员权限后再执行就不会有问题了

```bat
@echo off

net session >nul 2>&1

if %errorLevel% neq 0 (
    powershell -Command "Start-Process '%~dpnx0' -Verb RunAs"
    exit /b
)


cd /d "%~dp0"

conda activate data && python ui.py
```

最后的那个 conda activate data && python ui.py 换成自己的启动语句就可以了

### __功能详情__

软件当前的 UI 是这样的

![image](./images/ui.png)

UI 仍然在开发中...

每次使用一个功能时，菜单会收起来，如果你想中断那只能打开任务管理器搜索 python 之后强制中断

好消息是因为屏幕只有一个进程只有一个所以我们不用考虑多线程导致的资源竞争，使用 QThread 也只是为了方便组织

在进程执行期间请不要控制任何 USB HID 输入设备, 否则可能出现问题

|功能|目录|
|-----|-----|
|README静态文件|images|
|获取图像模态数据|data|
|光遇行为测试|sky-api|
|YOLOv11目标检测|yolov11|
|主程序|main|

功能仍在开发中，请耐心等待

作者不定时会更新一下，剩下的时间都在鸽hhhh

### __自动收发心火__

使用计算机视觉方案获取屏幕上的亮点特征后进行分析

走传统计算机视觉路线了, 开始抓取图片中心的旋转不变特性, 还能顺便清理一下文字的那些杂色

之后我们再根据相似的特征来进行筛选, 直接绘制出八个方向的折线图

目前这一版是我比较满意的一版，可以看出来已经能识别出爱心、还有其他需要收、送心火的点

目前还可以识别文字在图片中的位置

![image](./images/machine_vision.png)

HSV 色相检测爱心，之后用四角星检测是否需要送心火，八角星不需要送

最后加上了星屑检测，这个不是很准，所以可能会有多点的时候，霉逝的这个不难判断

我们使用 Space 来控制进入，这样就省略了很多麻烦的东西

### __主动取心__

由于本人技术有限所以只能做到这种地步，下面这种的想必大家都比较眼熟

![image](./images/web.png)

可以取这种形式的心，同时该类网站一般没有反爬措施，所以请注意爬取频率

按钮识别使用正则表达式, 有一定的通用性

因为它也是基于图形学的所以有一定的偏差，同时 OCR 使用两个

ddddocr 快速识别信息、paddleocr 准确识别内容，所以包体会很大是必然的

```python
# 星盘上必须是这些字，修改之后识别不到的，目前不打算完善这部分， 请务必注意
patterns = [
    r'添[\x00-\x7F]{0,3}加[\x00-\x7F]{0,3}好[\x00-\x7F]{0,3}友',
    r'好[\x00-\x7F]{0,3}友',
    r'挚[\x00-\x7F]{0,3}友'
]

# 识别确定、取爱心、复制编码、间隔领取按钮，这是该网站的文字特征
patterns = [
    r'确[\x00-\x7F]{0,5}定',
    r'取.*?爱心',
    r'复[\x00-\x7F]{0,2}制[\x00-\x7F]{0,2}编[\x00-\x7F]{0,2}码',
    r'请[\x00-\x7F]{0,8}秒后领取'
]

# 识别好友码
code = pyperclip.paste() # 举例 APS5-S0EQ-DH2B, 下面的这个是好友码的文字特征
if re.search(r'^[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$', code):
    print("识别到好友码:", code)
```

因为今天的链接都领完了所以这次没办法再领了

我们首先需要调整 yaml 文件中的配置

```yaml
delay: 10 # 这个是个轮次之间需要等待的时间数
episode: 2 # 这个是我们需要取心的轮次
file: E:\pandownload1\ML\links.xlsx # 这个是我们取心链接的 Excel 表的绝对地址, 路径不要有空格
index: 0 # 起名编号, 取名为 "AAA送心员{index}"
```

之后正常运行就可以了，可以在后面确定参数 --yaml config.yaml 指定需要读取的 yaml 文件

![image](./images/purchase_output.png)

有进度条不用担心卡死，其实是为了等待取心网页的后端服务器，根据自己的需求灵活调整

### __自动游走跑图__

使用目标检测算法确定物体位置并使用强化学习算法控制任务游走

使用的算法为 YOLOv11，因为 YOLOv12 引入了注意力机制

RTX 3050 LapTop 所以不支持 flash_attn 所以运行速度较慢, 使用 CUDA 加速

之后打算使用深度强化学习进行学习, 所以我只能说路漫漫...