import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import shutil
from PIL import Image


# 定义数据集路径
data_dir = r"C:\Users\lipton\Downloads\dogs-vs-cats-redux-kernels-edition\test"

# 创建子文件夹
os.makedirs(os.path.join(data_dir, 'cats'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'dogs'), exist_ok=True)

# 移动文件到相应的子文件夹
for i in range(2500):
    cat_file = os.path.join(data_dir, f'cat.test.{i}.jpg')
    dog_file = os.path.join(data_dir, f'dog.test.{i}.jpg')

    if os.path.exists(cat_file):
        shutil.move(cat_file, os.path.join(data_dir, 'cats', f'cat.test.{i}.jpg'))

    if os.path.exists(dog_file):
        shutil.move(dog_file, os.path.join(data_dir, 'dogs', f'dog.test.{i}.jpg'))