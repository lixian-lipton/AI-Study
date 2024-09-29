# AI-Study
初学者的一些尝试。


## Introductory Assessment：

### Resnet：

#### 1、模型
resnet-50   (参数存储于model_state_dict.pth)
#### 2、数据集
链接：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/overview 

（注：由于测试集缺少 labels ，因此实际操作时将原训练集的 25000 张图片切分成了 20000 张训练图片和 5000 张测试图片，没有使用原测试集）

#### 3、测试记录

main_easy：（使用 torchvision 预训练模型）
- 初始参数：预训练参数
- 首次训练 (lr=0.001, epoch=10) 效果较好，测试准确率 95.56%
- 修改学习率等超参，严重 overfitting ，训练集准确率 99.99% ，测试集准确率低至 92%
- 重新训练，epoch=50，并在每个 epoch 结束后都进行了测试，绘制了随训练次数增加，loss 和 Accuracy 的变化图像 （见 easy_result。xlsx）
- 图像显示，后期结果在 95%-96% 之间波动，最高准确率 96.82%

main_hard：（自行编写，但事先阅读 models.resnet50 的源码学习参考）
- 初始参数：随机初始化
- 使用 Bottleneck 而非 Basic block
- 训练一段时间后不再收敛，93-94% 之间震荡长达 60 个 epoch；调整 batch-size 和 lr，效果不佳
- 引入L2正则化（进行中）

### VGG：

#### 1、模型
vgg-11   (参数存储于model_state_dict.pth)
#### 2、数据集
同上
#### 3、测试记录

main_easy：（使用 torchvision 预训练模型）