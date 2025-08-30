import torch
import torch.nn as nn
import torch.nn.functional as F

'''
感谢 豆包 的倾力协助, 让我瞅瞅是怎么个事
我第一次做混合专家模型出现了一大堆的问题...
尤其是配置分布式环境的时候，可是我特么只有单卡
我从来没配过多卡环境，而且最终落实到用户都不一定能用 GPU
所以让豆包输出了一个本地 MoE 的实现
'''

class MapEncoder(nn.Module):
    '''
    地图编码器: 处理801x801的地图探索记录
    '''
    def __init__(self, input_channels=3, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

class StateEncoder(nn.Module):
    '''
    状态编码器: 处理运动方向和速度
    '''
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ImageEncoder(nn.Module):
    '''
    图像编码器: 处理屏幕截图
    '''
    def __init__(self, input_channels=3, hidden_dim=256):
        super().__init__()
        # 使用简化的ResNet结构
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.res_block1 = self._make_res_block(64, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.res_block2 = self._make_res_block(128, 128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, hidden_dim)
        
    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(x + self.res_block1(x))  # 残差连接
        x = F.relu(self.conv3(x))
        x = F.relu(x + self.res_block2(x))  # 残差连接
        x = F.relu(self.conv4(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

class Expert(nn.Module):
    '''
    专家网络: 处理多模态融合后的特征
    '''
    def __init__(self, input_dim=576, hidden_dim=512, output_dim=256):
        super().__init__()
        self.output_dim = output_dim  # 添加output_dim属性
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Top2Gate(nn.Module):
    '''
    Top-2 门控机制, 为每个输入选择两个最佳专家
    '''
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.num_experts = num_experts
        
    def forward(self, x):
        # 计算门控分数
        logits = self.gate(x)
        
        # 找到Top-2专家
        top_logits, top_indices = torch.topk(logits, k=2, dim=-1)
        
        # 计算专家权重（softmax归一化）
        top_weights = F.softmax(top_logits, dim=-1)
        
        # 构建稀疏路由矩阵
        batch_size = x.shape[0]
        gates = torch.zeros(batch_size, self.num_experts, device=x.device)
        
        # 设置Top-2专家的权重
        for i in range(batch_size):
            gates[i, top_indices[i, 0]] = top_weights[i, 0]
            gates[i, top_indices[i, 1]] = top_weights[i, 1]
            
        return gates

class LocalMoE(nn.Module):
    '''
    本地MoE实现, 不依赖分布式环境
    '''
    def __init__(self, num_experts, expert_factory, gate, balance_loss_coef=0.1):
        super().__init__()
        self.experts = nn.ModuleList([expert_factory() for _ in range(num_experts)])
        self.gate = gate
        self.balance_loss_coef = balance_loss_coef
        self.num_experts = num_experts
        
    def forward(self, x):
        # 计算门控输出
        gates = self.gate(x)
        
        # 计算负载均衡损失
        expert_usage = gates.sum(0)
        ideal_usage = torch.ones_like(expert_usage) * (x.shape[0] / self.num_experts)
        l_aux = F.mse_loss(expert_usage, ideal_usage) * self.balance_loss_coef
        
        # 应用每个专家并聚合结果
        batch_size = x.shape[0]
        output_dim = self.experts[0].output_dim  # 获取正确的输出维度
        output = torch.zeros(batch_size, output_dim, device=x.device)
        
        for i, expert in enumerate(self.experts):
            # 获取选择当前专家的样本
            mask = gates[:, i] > 0
            if mask.sum() > 0:
                expert_input = x[mask]
                expert_output = expert(expert_input)
                output[mask] += gates[mask, i].unsqueeze(1) * expert_output
                
        return output, l_aux

class MoENavigationModel(nn.Module):
    '''
    于MoE的导航模型: 整合地图、状态和图像信息
    '''
    def __init__(self, 
                 map_channels=3, 
                 state_dim=2, 
                 image_channels=3,
                 num_experts=8, 
                 top_k=2, 
                 balance_loss_weight=0.1):
        super().__init__()
        
        # 编码器
        self.map_encoder = MapEncoder(input_channels=map_channels)
        self.state_encoder = StateEncoder(input_dim=state_dim)
        self.image_encoder = ImageEncoder(input_channels=image_channels)
        
        # 特征融合维度
        self.feature_dim = 256 + 64 + 256  # 地图 + 状态 + 图像
        
        # 本地MoE层（无分布式依赖）
        self.moe = LocalMoE(
            num_experts=num_experts,
            expert_factory=lambda: Expert(input_dim=self.feature_dim),
            gate=Top2Gate(self.feature_dim, num_experts),
            balance_loss_coef=balance_loss_weight
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),  # 256是Expert的输出维度
            nn.ReLU(),
            nn.Linear(128, 5),  # 输出5个动作
            nn.Sigmoid()  # 将输出归一化到[0,1]区间
        )
        
    def forward(self, map_input, state_input, image_input):
        # 提取特征
        map_features = self.map_encoder(map_input)
        state_features = self.state_encoder(state_input)
        image_features = self.image_encoder(image_input)
        
        # 特征融合
        combined_features = torch.cat([map_features, state_features, image_features], dim=1)
        
        # 通过MoE层
        moe_output, l_aux = self.moe(combined_features)
        
        # 生成动作输出
        actions = self.output_layer(moe_output)
        
        return actions, l_aux

# 示例使用
if __name__ == "__main__":
    # 创建模型并移至GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoENavigationModel(
        map_channels=3,    # 地图通道数（走过/未走过/不能走）
        state_dim=2,       # 运动状态维度（方向、速度）
        image_channels=3,  # RGB图像
        num_experts=8,     # 专家数量
        top_k=2,           # Top-K门控
        balance_loss_weight=0.1  # 负载均衡损失权重
    ).to(device)
    
    # 模拟输入
    map_input = torch.randn(1, 3, 801, 801).to(device)
    state_input = torch.randn(1, 2).to(device)
    image_input = torch.randn(1, 3, 224, 224).to(device)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        actions, moe_loss = model(map_input, state_input, image_input)
    print(f"动作输出: {actions}")
    print(f"MoE负载均衡损失: {moe_loss}")    