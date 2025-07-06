# coding=utf-8
import helium
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
import random
import time
import pandas as pd
import yaml
import argparse
from tqdm import tqdm
import pyperclip
import re

class WebCrawler:
    def __init__(self, sigma:float=0.07):
        """初始化爬虫，设置高斯分布的标准差"""
        self.sigma = sigma
        self.browser_started = False  # 跟踪浏览器是否成功启动
        self.code = None
        
    def gauss_sleep(self, seconds:float=0.6, min_seconds:float=0.1) -> None:
        """暂停指定的秒数"""
        time.sleep(max(min_seconds, random.gauss(seconds, self.sigma)))
        
    def re_keyword_detector(self, texts):
        """
        ASCII字符清洗检测子串, 最大程度保证我们能点到正确的内容
        """
        wait_time = 0
        patterns = [
            r'确[\x00-\x7F]{0,5}定', 
            r'取.*?爱心', 
            r'复[\x00-\x7F]{0,2}制[\x00-\x7F]{0,2}编[\x00-\x7F]{0,2}码', 
            r'请[\x00-\x7F]{0,8}秒后领取'
        ]
        
        dataframes = {
            '确定': [],
            '取爱心': [],
            '复制':[],
            '等待': [],
            '文本': []
        }
    
        for text in texts:
            matchs = list([bool(re.search(pattern, text)) for pattern in patterns])
            dataframes['确定'].append(matchs[0] == True)
            dataframes['取爱心'].append(matchs[1] == True)
            dataframes['复制'].append(matchs[2] == True)
            dataframes['等待'].append(matchs[3] == True)
            
            if matchs[3]:
                wait_time = int(re.findall(r'\d+', text)[0]) if re.findall(r'\d+', text) else 0
            
            dataframes['文本'].append(text)
            
        return pd.DataFrame(dataframes), wait_time

    def crawl_main(self, url="https://www.baidu.com/", headless=True, valid=False):
        """单次请求爬虫主函数, 不要太频繁地请求网页就可以"""
        self.code = None  # 重置 code
        
        if valid:
            print("您当前正咱运行在 VALID 验证环境中...请勿正式上线此版本...")
            
            code = 'APS5-S0EQ-DH2B' # 举例 APS5-S0EQ-DH2B
            if re.search(r'^[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$', code):
                self.code = code
            return True
           
        exp_flag = True 
        try:
            # 智能生成 User-Agent（基于真实统计数据）
            ua = UserAgent()
            user_agent = ua.chrome  # 固定使用 Chrome 的 UA
            chrome_options = Options()
            user_options = [
                f'--user-agent={user_agent}',
                '--disable-blink-features=AutomationControlled',
                '--start-maximized'
            ]
            for option in user_options:
                chrome_options.add_argument(option)
            
            helium.start_chrome(url, headless=headless, options=chrome_options)
            self.browser_started = True
            print("正在加载页面: {}".format(url))
            self.gauss_sleep(10) # 使用随机数用于逃过验证
            
            texts = []
            buttons = helium.find_all(helium.S("button"))
            for index, button in enumerate(buttons):
                texts.append(button.web_element.text)
            
            button_text = None
            pf, wait_time = self.re_keyword_detector(texts)
            if wait_time > 0:
                print(f"识别到等待时间: {wait_time} 秒")
                with tqdm(total=wait_time+5, desc=f"等待", unit='s') as pbar: # 正宗延时等待
                    for _ in range(wait_time+5):
                        time.sleep(1)
                        pbar.update(1)
                
                timeout = 30
                while True:
                    pf, _ = self.re_keyword_detector(texts)
                    if len(pf[pf['取爱心']]['文本'].values) > 0:
                        button_text = pf[pf['取爱心']]['文本'].values[0]
                        print("识别到取爱心按钮: {}".format(button_text))
                        break
                    else:
                        print("没有找到取爱心按钮, 1s 秒后重试...")
                        timeout -= 1
                        self.gauss_sleep(1)
            
            while True: # 如果繁忙那我们就一直扫就可以了，反正别的链接也会是繁忙的
                button_text, timeout = None, 10
                while timeout > 0:
                    pf, _ = self.re_keyword_detector(texts)
                    if len(pf[pf['取爱心']]['文本'].values) > 0:
                        button_text = pf[pf['取爱心']]['文本'].values[0]
                        break
                    
                    print("没有找到取爱心按钮, 等待 2 秒后重试...")
                    timeout -= 1
                    self.gauss_sleep(2)
                
                heart_button = helium.Button(button_text)
                helium.wait_until(heart_button.exists, timeout_secs=10)
                helium.click(heart_button)
                
                self.gauss_sleep(5)
                texts = []
                buttons = helium.find_all(helium.S("button"))
                for index, button in enumerate(buttons):
                    texts.append(button.web_element.text)
                pf, _ = self.re_keyword_detector(texts)
                    
                # 双重验证, 如果出现这个或者没有出现确定那么我们就这样处理
                if helium.Text("今日取心已到上限了,请明天再来哦！").exists() or len(pf[pf['确定']]['文本'].values) == 0:
                    raise Exception("本链接已经达到上限")
                if helium.Text("送心员繁忙！").exists() or len(pf[pf['确定']]['文本'].values) == 0:
                    print("送心员繁忙, 请稍后再试...")
                    self.gauss_sleep(10)
                    continue
            
                ok_text = pf[pf['确定']]['文本'].values[0]
                ok_button = helium.Button(ok_text)
                helium.wait_until(ok_button.exists, timeout_secs=10)
                helium.click(ok_button)
                break
            
            self.gauss_sleep(3)
            
            texts = []
            buttons = helium.find_all(helium.S("button"))
            for index, button in enumerate(buttons):
                texts.append(button.web_element.text)
            pf, _ = self.re_keyword_detector(texts)
            code_text = pf[pf['复制']]['文本'].values[0]
            
            code_button = helium.Button(code_text)
            helium.wait_until(code_button.exists, timeout_secs=10)
            helium.click(code_button)
            self.gauss_sleep(1)
            
            # 将剪切板内容复制出来
            code = pyperclip.paste() # 举例 APS5-S0EQ-DH2B
            if re.search(r'^[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$', code):
                self.code = code
            
            self.gauss_sleep(5)
            
        except Exception as e:
            print(f"(WEB_CRAWLER) 捕捉到 Exception: {e}")
            exp_flag = False
        finally:
            if self.browser_started:
                helium.kill_browser()
                self.browser_started = False
                
            return exp_flag

if __name__ == "__main__":
    def opt_parser():
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description="Web Crawler for Heart Collection")
        parser.add_argument('--yaml', type=str, default="config.yaml", help='需要读取的 yaml 文件位置')
        return parser.parse_args()
    
    # 取心地址, Excel 中要有一个表头为 url 在 (1, A) 位置
    opt = opt_parser()
    
    data = None
    with open(opt.yaml, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        file.close()
    
    pf = pd.read_excel(data['file'], sheet_name="Sheet1", names=["url"])
    crawler = WebCrawler()
    
    for epoch in range(data['episode']):
        print(f"第 {epoch + 1} 轮取心")
        for index, row in pf.iterrows():
            target_url = row['url']
            crawler.crawl_main(target_url)
            
        if epoch != data['episode'] - 1:
            with tqdm(total=data['delay'], desc=f"Time", unit='s') as pbar:
                for i in range(data['delay']):
                    time.sleep(1)
                    pbar.update(1)
        
    print("全部链接已经处理完毕")