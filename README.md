# AI-Study
初学者的一些尝试。


#Introductory Assessment：

1、Resnet：
模型：resnet-50
数据集链接：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/overview
（注：由于测试集缺少 labels ，因此实际操作时将原训练集的 25000 张图片切分成了 20000 张训练图片和 5000 张测试图片）
测试结果：
main_easy 中使用 torchvision 库预训练的 models.resnet50 进行训练，batch size=32，epoch=10，测试准确率为 95.56% 。
main_diff 中继承 torch 的 Modules 基类进行编写，