import helium
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
import random
import time
import pandas as pd
import os

class WebCrawler:
    def __init__(self, sigma:float=0.07):
        """初始化爬虫，设置高斯分布的标准差"""
        self.sigma = sigma
        self.browser_started = False  # 跟踪浏览器是否成功启动
        
    def gauss_sleep(self, seconds:float=0.6, min_seconds:float=0.1) -> None:
        """暂停指定的秒数"""
        time.sleep(max(min_seconds, random.gauss(seconds, self.sigma)))

    def crawl_main(self, url="https://www.baidu.com/"):
        """爬虫主函数, 不要太频繁地请求网页就可以"""
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
            
            helium.start_chrome(url, headless=False, options=chrome_options)
            self.browser_started = True
            print("正在加载页面: {}".format(url))
            self.gauss_sleep(8) # 使用随机数用于逃过验证
            
            heart_button = helium.Button("点击取一颗爱心")
            helium.wait_until(heart_button.exists, timeout_secs=10)
            helium.click(heart_button)
            
            self.gauss_sleep(2)
            if helium.Text("今日取心已到上限了,请明天再来哦！").exists():
                raise Exception("本链接已经达到上限")
            
            code_button = helium.Button("复制编码")
            helium.wait_until(code_button.exists, timeout_secs=10)
            helium.click(code_button)
            
            with open("code.txt", "w+", encoding='utf-8') as f:
                f.write(helium.get_clipboard())
                f.close()
            
            self.gauss_sleep(5)
            
        except Exception as e:
            print(f"捕捉到 Exception: {e}")
        finally:
            if self.browser_started:
                helium.kill_browser()
                self.browser_started = False

if __name__ == "__main__":
    # 取心地址, Excel 中要有一个表头为 url 在 (1, A) 位置
    data = pd.read_excel(os.path.join("source", "links.xlsx"), sheet_name="Sheet1", names=["url"])
    crawler = WebCrawler()
    
    for index, row in data.iterrows():
        target_url = row['url']
        crawler.crawl_main(target_url)
        
    print("全部链接已经处理完毕")