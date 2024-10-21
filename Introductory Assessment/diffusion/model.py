import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np


class UNet(nn.Module):
    """ U-Net 模型，用于图像去噪和重建 """

    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        # 编码器部分
        self.enc1 = self.encoder_block(in_channels, 64)
        self.enc2 = self.encoder_block(64, 128)
        self.enc3 = self.encoder_block(128, 256)
        self.enc4 = self.encoder_block(256, 512)

        # Bottleneck
        self.bottleneck = self.encoder_block(512, 1024)

        # 解码器部分
        self.dec4 = self.decoder_block(1024, 512)
        self.dec3 = self.decoder_block(512, 256)
        self.dec2 = self.decoder_block(256, 128)
        self.dec1 = self.decoder_block(128, 64)

        # 输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def encoder_block(self, in_channels, out_channels):
        """ 定义编码器中的卷积块 """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 卷积层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 卷积层
            nn.ReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        )

    def decoder_block(self, in_channels, out_channels):
        """ 定义解码器中的卷积块 """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),  # 上采样
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 卷积层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # 卷积层
        )

    def forward(self, x):
        """ 前向传播过程 """
        enc1 = self.enc1(x)  # 编码器第一层
        enc2 = self.enc2(enc1)  # 编码器第二层
        enc3 = self.enc3(enc2)  # 编码器第三层
        enc4 = self.enc4(enc3)  # 编码器第四层

        bottleneck = self.bottleneck(enc4)  # Bottleneck层

        dec4 = self.dec4(bottleneck)  # 解码器第四层
        dec4 = torch.cat((dec4, enc4), dim=1)  # 跳跃连接
        dec3 = self.dec3(dec4)  # 解码器第三层
        dec3 = torch.cat((dec3, enc3), dim=1)  # 跳跃连接
        dec2 = self.dec2(dec3)  # 解码器第二层
        dec2 = torch.cat((dec2, enc2), dim=1)  # 跳跃连接
        dec1 = self.dec1(dec2)  # 解码器第一层
        dec1 = torch.cat((dec1, enc1), dim=1)  # 跳跃连接

        return self.final_conv(dec1)  # 最终输出


class DiffusionModel:
    """ 扩散模型的实现 """

    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        # 线性噪声调度
        self.beta = np.linspace(0.0001, 0.02, num_timesteps)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = np.cumprod(self.alpha)

    def get_noisy_data(self, x_0, t, noise):
        """ 根据时间步 t 添加噪声 """
        return (
                torch.sqrt(self.alpha_cumprod[t]) * x_0 +
                torch.sqrt(1 - self.alpha_cumprod[t]) * noise
        )

    def loss(self, model, x_0, t):
        """ 计算模型的损失 """
        # 1. 生成与 x_0 相同形状的噪声
        noise = torch.randn_like(x_0)  # 生成随机噪声

        # 2. 获取带噪声的样本
        x_t = self.get_noisy_data(x_0, t, noise)  # 将噪声传递给该函数

        # 3. 预测噪声
        noise_pred = model(x_t)  # 使用模型预测噪声

        # 4. 返回均方误差损失
        return nn.MSELoss()(noise_pred, noise)  # 计算预测噪声与真实噪声之间的差异


def train(model, diffusion_model, dataloader, num_epochs=10, lr=1e-3):
    """ 训练扩散模型 """
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器

    for epoch in range(num_epochs):
        for images, _ in dataloader:
            images = images.to(device)  # 将数据移到GPU
            t = torch.randint(0, diffusion_model.num_timesteps, (images.size(0),)).to(device)  # 随机选择时间步
            optimizer.zero_grad()  # 清空梯度
            loss = diffusion_model.loss(model, images, t)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')  # 打印损失


# 超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
batch_size = 64
num_epochs = 10

# 数据集和数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 数据标准化
])
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)  # CIFAR-10数据集
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 创建数据加载器

# 模型初始化
unet_model = UNet(in_channels=3, out_channels=3).to(device)  # 创建U-Net模型
diffusion_model = DiffusionModel(num_timesteps=1000)  # 创建扩散模型

# 训练模型
train(unet_model, diffusion_model, dataloader, num_epochs=num_epochs)