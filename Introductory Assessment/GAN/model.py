import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision

# 设置设备为 GPU（如果可用），否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 生成器模型
class Generator(nn.Module):
    """ 生成器模型，用于生成假图像 """

    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        # 定义生成器的结构
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),  # 输入层：从潜在空间（z_dim）到 128 个神经元
            nn.ReLU(),               # 使用 ReLU 激活函数
            nn.Linear(128, 256),     # 隐藏层：128 到 256 个神经元
            nn.ReLU(),               # 使用 ReLU 激活函数
            nn.Linear(256, img_dim), # 输出层：256 到图像维度（img_dim），例如 28*28=784
            nn.Tanh()                # 使用 Tanh 激活函数，将输出范围限制在 [-1, 1]
        )

    def forward(self, z):
        """ 前向传播方法，生成假图像 """
        return self.model(z)  # 将潜在向量 z 输入生成器模型

# 判别器模型
class Discriminator(nn.Module):
    """ 判别器模型，用于判断图像是真实还是假 """

    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        # 定义判别器的结构
        self.model = nn.Sequential(
            nn.Linear(img_dim, 256),  # 输入层：从图像维度（img_dim）到 256 个神经元
            nn.LeakyReLU(0.2),        # 使用 Leaky ReLU 激活函数，避免神经元死亡
            nn.Linear(256, 128),      # 隐藏层：256 到 128 个神经元
            nn.LeakyReLU(0.2),        # 使用 Leaky ReLU 激活函数
            nn.Linear(128, 1),        # 输出层：128 到 1 个神经元，输出判别结果
            nn.Sigmoid()              # 使用 Sigmoid 激活函数，输出范围在 [0, 1] 之间
        )

    def forward(self, img):
        """ 前向传播方法，判断图像真假 """
        return self.model(img)  # 将输入图像传递给判别器模型

# 超参数设置
z_dim = 100  # 潜在空间的维度
img_dim = 784  # 图像的维度（28*28=784）
batch_size = 64  # 批量大小
lr = 0.0002  # 学习率
num_epochs = 10  # 训练轮数

# 数据集加载
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化图像到 [-1, 1]
])

# 使用 MNIST 数据集
dataset = dsets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化生成器和判别器
generator = Generator(z_dim, img_dim).to(device)  # 将生成器移动到设备（GPU/CPU）
discriminator = Discriminator(img_dim).to(device)  # 将判别器移动到设备

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer_g = optim.Adam(generator.parameters(), lr=lr)  # 生成器优化器
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)  # 判别器优化器

# 训练 GAN
for epoch in range(num_epochs):
    for idx, (imgs, _) in enumerate(dataloader):
        # 获取真实图像并将其转换为设备
        imgs = imgs.view(-1, img_dim).to(device)  # 展平图像

        # 训练判别器
        optimizer_d.zero_grad()  # 清空梯度
        real_labels = torch.ones(imgs.size(0), 1).to(device)  # 真实标签为 1
        real_loss = criterion(discriminator(imgs), real_labels)  # 计算真实图像的损失

        z = torch.randn(imgs.size(0), z_dim).to(device)  # 随机生成潜在向量
        fake_imgs = generator(z)  # 使用生成器生成假图像
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)  # 假标签为 0
        fake_loss = criterion(discriminator(fake_imgs), fake_labels)  # 计算假图像的损失

        d_loss = real_loss + fake_loss  # 总判别器损失
        d_loss.backward()  # 反向传播
        optimizer_d.step()  # 更新判别器参数

        # 训练生成器
        optimizer_g.zero_grad()  # 清空梯度
        z = torch.randn(imgs.size(0), z_dim).to(device)  # 生成新的潜在向量
        fake_imgs = generator(z)  # 生成假图像
        real_labels = torch.ones(imgs.size(0), 1).to(device)  # 生成器希望假图像被判别为真实
        g_loss = criterion(discriminator(fake_imgs), real_labels)  # 计算生成器损失

        g_loss.backward()  # 反向传播
        optimizer_g.step()  # 更新生成器参数

    # 每个 epoch 输出损失
    print(f'Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}')

# 生成样本并可视化
with torch.no_grad():
    test_z = torch.randn(16, z_dim).to(device)  # 生成随机潜在向量
    generated_imgs = generator(test_z).view(-1, 1, 28, 28)  # 生成假图像并调整形状

    # 可视化生成的图像
    grid_img = torchvision.utils.make_grid(generated_imgs, nrow=4, normalize=True)  # 创建网格图像
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())  # 转换为 NumPy 数组并可视化
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示图像