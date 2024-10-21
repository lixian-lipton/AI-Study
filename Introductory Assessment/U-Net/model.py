import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder部分：逐步提取特征，降低空间维度
        self.enc1 = self.encoder_block(in_channels, 64)  # 第一层，输入通道数为in_channels，输出64个通道
        self.enc2 = self.encoder_block(64, 128)  # 第二层，输入64个通道，输出128个通道
        self.enc3 = self.encoder_block(128, 256)  # 第三层，输入128个通道，输出256个通道
        self.enc4 = self.encoder_block(256, 512)  # 第四层，输入256个通道，输出512个通道

        # Bottleneck部分：连接编码器和解码器，提取更高层次的特征
        self.bottleneck = self.encoder_block(512, 1024)

        # Decoder部分：逐步恢复空间维度
        self.dec4 = self.decoder_block(1024, 512)  # 第四层解码
        self.dec3 = self.decoder_block(512, 256)  # 第三层解码
        self.dec2 = self.decoder_block(256, 128)  # 第二层解码
        self.dec1 = self.decoder_block(128, 64)  # 第一层解码

        # 最终卷积层：将输出压缩到所需的通道数
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def encoder_block(self, in_channels, out_channels):
        """ 构建编码器块，包括卷积和池化层 """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3卷积
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 3x3卷积
            nn.ReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化，降低空间维度
        )

    def decoder_block(self, in_channels, out_channels):
        """ 构建解码器块，包括转置卷积和卷积层 """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),  # 上采样
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 3x3卷积
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 3x3卷积
            nn.ReLU(inplace=True)  # 激活函数
        )

    def forward(self, x):
        """ 前向传播 """
        # 编码过程
        enc1 = self.enc1(x)  # 通过第一层编码
        enc2 = self.enc2(enc1)  # 通过第二层编码
        enc3 = self.enc3(enc2)  # 通过第三层编码
        enc4 = self.enc4(enc3)  # 通过第四层编码

        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # 通过瓶颈层

        # 解码过程
        dec4 = self.dec4(bottleneck)  # 通过第四层解码
        dec4 = torch.cat((dec4, enc4), dim=1)  # 跳跃连接，将编码器的输出与解码器的输出连接
        dec3 = self.dec3(dec4)  # 通过第三层解码
        dec3 = torch.cat((dec3, enc3), dim=1)  # 跳跃连接
        dec2 = self.dec2(dec3)  # 通过第二层解码
        dec2 = torch.cat((dec2, enc2), dim=1)  # 跳跃连接
        dec1 = self.dec1(dec2)  # 通过第一层解码
        dec1 = torch.cat((dec1, enc1), dim=1)  # 跳跃连接

        # 最终输出
        return self.final_conv(dec1)  # 应用最后的卷积层，得到最终输出


# 示例用法
if __name__ == "__main__":
    # 创建U-Net模型
    model = UNet(in_channels=3, out_channels=1)

    # 打印模型架构
    print(model)

    # 创建随机输入张量，形状为(batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 256, 256)  # 示例为256x256的RGB图像

    # 前向传播
    output = model(input_tensor)
    print(f'Output shape: {output.shape}')  # 应该为(1, 1, 256, 256)