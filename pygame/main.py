import pygame
import sys
import random
import math
from datetime import datetime

# 初始化pygame
pygame.init()
pygame.font.init()

# 确保中文显示正常
try:
    font = pygame.font.Font("simhei.ttf", 16)  # 尝试加载系统中的黑体字
except:
    # 如果找不到指定字体，使用默认字体
    font = pygame.font.SysFont(["SimHei", "WenQuanYi Micro Hei", "Heiti TC"], 16)

# 屏幕设置
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("基于机器视觉的自动控制软件")
pygame.display.set_icon(pygame.image.load("icon2.png"))  # 确保有一个icon.png文件在同一目录下

# 颜色定义
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
BLUE = (0, 120, 215)
GREEN = (52, 199, 89)
BLACK = (0, 0, 0)

# 聊天消息类
class Message:
    def __init__(self, text, is_self=True):
        self.text = text
        self.is_self = is_self
        self.time = datetime.now().strftime("%H:%M")
        self.rect = None
        
    def render(self, surface, x, y, max_width):
        # 渲染消息气泡
        text_surface = font.render(self.text, True, BLACK)
        text_width = text_surface.get_width() + 20
        text_height = text_surface.get_height() + 15
        
        # 计算气泡位置和大小
        if self.is_self:
            bubble_rect = pygame.Rect(x - text_width, y, text_width, text_height)
            pygame.draw.rect(surface, GREEN, bubble_rect, border_radius=10)
            # 绘制气泡小三角
            pygame.draw.polygon(surface, GREEN, [(x, y + 10), (x - 10, y + 5), (x - 10, y + 15)])
        else:
            bubble_rect = pygame.Rect(x, y, text_width, text_height)
            pygame.draw.rect(surface, WHITE, bubble_rect, border_radius=10)
            # 绘制气泡小三角
            pygame.draw.polygon(surface, WHITE, [(x, y + 10), (x + 10, y + 5), (x + 10, y + 15)])
        
        # 渲染消息文本和时间
        if self.is_self:
            surface.blit(text_surface, (bubble_rect.x + 10, bubble_rect.y + 7))
            time_surface = font.render(self.time, True, DARK_GRAY)
            surface.blit(time_surface, (bubble_rect.x - time_surface.get_width() - 5, bubble_rect.y + 5))
        else:
            surface.blit(text_surface, (bubble_rect.x + 10, bubble_rect.y + 7))
            time_surface = font.render(self.time, True, DARK_GRAY)
            surface.blit(time_surface, (bubble_rect.x + bubble_rect.width + 5, bubble_rect.y + 5))
        
        self.rect = bubble_rect
        return bubble_rect.height + 15

class Particle:
    def __init__(self, x, y):
        self.original_x = x                  # 原始位置
        self.original_y = y
        self.x = x                           # 当前位置
        self.y = y
        self.radius = random.randint(1, 3)
        self.color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
        self.speed = random.uniform(1.0, 2.0)  # 固定移动速度
        self.distance = random.uniform(20, 50) # 旋转半径
        self.angle = random.uniform(0, 2 * math.pi)  # 当前旋转角度
        self.base_speed = random.uniform(0.002, 0.03) # 基础旋转速度
        self.clicked = False                   # 是否被点击
        self.return_progress = 0               # 返回进度 (0-1)
        self.direction_x = 0                   # 扩散方向
        self.direction_y = 0
        self.diffusion_distance = 0            # 扩散距离
        self.state = "idle"                    # 状态: idle, diffusing, returning
        
    def update(self, mouse_pos, mouse_drag, clicked, click_pos):
        # 处理点击事件
        if clicked and self.state == "idle":
            dx = self.x - click_pos[0]
            dy = self.y - click_pos[1]
            distance_to_click = math.sqrt(dx*dx + dy*dy)
            
            if distance_to_click < 150:
                # 计算扩散方向和距离
                self.direction_x = dx / distance_to_click
                self.direction_y = dy / distance_to_click
                
                # 距离越近，扩散距离越大
                self.diffusion_distance = (150 - distance_to_click) / 150 * 500
                self.return_progress = 0
                self.state = "diffusing"
        
        # 更新粒子状态
        if self.state == "diffusing":
            # 扩散阶段
            move_distance = min(self.speed, self.diffusion_distance)
            self.x += self.direction_x * move_distance
            self.y += self.direction_y * move_distance
            self.diffusion_distance -= move_distance
            
            if self.diffusion_distance <= 0:
                self.state = "returning"
                # 计算返回方向
                dx_return = self.original_x - self.x
                dy_return = self.original_y - self.y
                distance_to_origin = math.sqrt(dx_return*dx_return + dy_return*dy_return)
                self.direction_x = dx_return / distance_to_origin if distance_to_origin > 0 else 0
                self.direction_y = dy_return / distance_to_origin if distance_to_origin > 0 else 0
                # 距离越远，返回时间越长
                self.return_distance = distance_to_origin
                self.return_progress = 0
                
        elif self.state == "returning":
            # 返回阶段
            move_distance = min(self.speed, self.return_distance)
            self.x += self.direction_x * move_distance
            self.y += self.direction_y * move_distance
            self.return_distance -= move_distance
            self.return_progress += move_distance / (self.return_distance + move_distance)
            
            if self.return_distance <= 0:
                self.x = self.original_x
                self.y = self.original_y
                self.state = "idle"
                self.angle = random.uniform(0, 2 * math.pi)  # 重置角度
        else:
            # 正常旋转
            self.angle += self.base_speed + (mouse_drag * 0.03)
            self.x = self.original_x + math.cos(self.angle) * self.distance
            self.y = self.original_y + math.sin(self.angle) * self.distance
    
    def draw(self, surface):
        # 绘制粒子
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        
        # 可选：绘制返回进度指示器
        if self.state == "returning":
            progress_radius = max(1, self.radius * (1 - self.return_progress))
            pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), int(progress_radius), 1)

# 聊天界面类
class ChatInterface:
    def __init__(self):
        self.messages = []
        self.input_text = ""
        self.input_rect = pygame.Rect(50, HEIGHT - 80, WIDTH - 100, 40)
        self.send_button = pygame.Rect(WIDTH - 120, HEIGHT - 130, 100, 30)
        self.scroll_offset = 0
        self.max_scroll = 0
        self.particles = []
        self.mouse_dragging = False
        self.drag_start_angle = 0
        self.current_drag = 0
        self.clicked = False
        self.click_pos = (0, 0)
        self.num_particles = random.randint(50, 80)         # 粒子数量
        self.contact_distance = random.uniform(50, 100)    # 粒子链接距离
        
        # 初始化粒子
        self.initialize_particles()
        
        # 添加欢迎消息
        self.add_message("欢迎使用微信风格聊天软件！", False)
        self.add_message("点击屏幕，背景粒子会向外扩散并部分回弹。", False)
        self.add_message("拖动鼠标可以影响线条的旋转速度和方向。", False)
    
    def initialize_particles(self):
        # 创建粒子
        for _ in range(self.num_particles):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT - 150)  # 避开输入区域
            self.particles.append(Particle(x, y))
    
    def add_message(self, text, is_self=True):
        self.messages.append(Message(text, is_self))
        self.update_scroll()
    
    def update_scroll(self):
        # 计算最大滚动量
        total_height = 20  # 顶部间距
        for message in self.messages:
            total_height += message.render(pygame.Surface((WIDTH, HEIGHT)), WIDTH - 100 if message.is_self else 100, 0, WIDTH - 200) + 10
        
        self.max_scroll = max(0, total_height - (HEIGHT - 150))
    
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if self.input_text.strip() != "":
                    self.add_message(self.input_text)
                    # 添加自动回复
                    if random.random() < 0.8:
                        replies = [
                            "我收到了你的消息: " + self.input_text,
                            "是的，你说得对。",
                            "很有趣的观点！",
                            "让我考虑一下...",
                            "我稍后回复你。"
                        ]
                        pygame.time.delay(500)  # 模拟思考时间
                        self.add_message(random.choice(replies), False)
                    self.input_text = ""
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                self.input_text += event.unicode
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 左键
                if self.send_button.collidepoint(event.pos):
                    if self.input_text.strip() != "":
                        self.add_message(self.input_text)
                        self.input_text = ""
                else:
                    self.mouse_dragging = True
                    self.drag_start_angle = math.atan2(event.pos[1] - HEIGHT/2, event.pos[0] - WIDTH/2)
                    self.clicked = True
                    self.click_pos = event.pos
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # 左键
                self.mouse_dragging = False
                self.current_drag = 0
        
        elif event.type == pygame.MOUSEMOTION and self.mouse_dragging:
            current_angle = math.atan2(event.pos[1] - HEIGHT/2, event.pos[0] - WIDTH/2)
            angle_diff = current_angle - self.drag_start_angle
            
            # 处理角度跳跃（超过π或-π）
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            self.current_drag = angle_diff
            self.drag_start_angle = current_angle
        
        elif event.type == pygame.MOUSEWHEEL:
            # 处理滚动
            self.scroll_offset = max(0, min(self.max_scroll, self.scroll_offset - event.y * 30))
    
    def update(self, mouse_pos):
        # 更新粒子
        for particle in self.particles:
            particle.update(mouse_pos, self.current_drag, self.clicked, self.click_pos)
        
        # 重置点击状态
        self.clicked = False
    
    def draw(self, surface):
        # 绘制背景
        surface.fill((240, 240, 240))
        
        # 绘制粒子和连接线
        for i, particle in enumerate(self.particles):
            particle.draw(surface)
            
            # 绘制与其他粒子的连接线
            for j in range(i + 1, len(self.particles)):
                other = self.particles[j]
                dx = particle.x - other.x
                dy = particle.y - other.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < self.contact_distance:
                    # 距离越近，线条越明显
                    alpha = int(255 * (1 - distance / self.contact_distance))
                    color = (min(200, particle.color[0] + other.color[0] - self.contact_distance),
                             min(200, particle.color[1] + other.color[1] - self.contact_distance),
                             min(200, particle.color[2] + other.color[2] - self.contact_distance),
                             alpha)
                    
                    pygame.draw.line(surface, color, (particle.x, particle.y), (other.x, other.y), 1)
        
        # # 绘制聊天区域
        # chat_area = pygame.Rect(50, 50, WIDTH - 100, HEIGHT - 200)
        # pygame.draw.rect(surface, WHITE, chat_area)
        # pygame.draw.rect(surface, GRAY, chat_area, 1)
        
        # # 绘制消息
        # y_offset = 20 - self.scroll_offset
        # for message in self.messages:
        #     height = message.render(surface, WIDTH - 100 if message.is_self else 100, y_offset, WIDTH - 200)
        #     y_offset += height + 10
        
        # # 绘制输入框
        # pygame.draw.rect(surface, WHITE, self.input_rect)
        # pygame.draw.rect(surface, GRAY, self.input_rect, 2)
        
        # # 绘制输入文本
        # text_surface = font.render(self.input_text, True, BLACK)
        # surface.blit(text_surface, (self.input_rect.x + 10, self.input_rect.y + 10))
        
        # # 绘制光标
        # if pygame.time.get_ticks() % 1000 < 500:
        #     cursor_x = self.input_rect.x + text_surface.get_width() + 10
        #     pygame.draw.line(surface, BLACK, (cursor_x, self.input_rect.y + 10), 
        #                     (cursor_x, self.input_rect.y + self.input_rect.height - 10), 2)
        
        # # 绘制发送按钮
        # pygame.draw.rect(surface, BLUE, self.send_button, border_radius=5)
        # button_text = font.render("发送", True, WHITE)
        # text_rect = button_text.get_rect(center=self.send_button.center)
        # surface.blit(button_text, text_rect)
        
        # # 绘制标题
        # title_text = font.render("微信风格聊天", True, BLACK)
        # title_rect = title_text.get_rect(center=(WIDTH//2, 25))
        # surface.blit(title_text, title_rect)
        
        # # 绘制提示
        # hint_text = font.render("点击屏幕让粒子扩散，拖动鼠标影响旋转", True, DARK_GRAY)
        # surface.blit(hint_text, (WIDTH - hint_text.get_width() - 20, HEIGHT - 30))

def main():
    clock = pygame.time.Clock()
    chat_interface = ChatInterface()
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            chat_interface.handle_event(event)
        
        # 更新界面
        chat_interface.update(mouse_pos)
        
        # 绘制界面
        chat_interface.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()    