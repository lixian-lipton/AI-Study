import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import shutil
from PIL import Image


# 定义数据集路径
data_dir = r"C:\Users\24800\Desktop\AI\cats-vs-dogs"

# 创建子文件夹
os.makedirs(os.path.join(data_dir, 'test'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'test', 'cats'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'test', 'dogs'), exist_ok=True)

# 移动文件到相应的子文件夹
for i in range(10000, 15000):
    cat_file = os.path.join(data_dir, 'train', 'cats', f'cat.{i}.jpg')
    dog_file = os.path.join(data_dir, 'train', 'dogs', f'dog.{i}.jpg')

    if os.path.exists(cat_file):
        shutil.move(cat_file, os.path.join(data_dir, 'test', 'cats', f'cat.{i-10000}.jpg'))

    if os.path.exists(dog_file):
        shutil.move(dog_file, os.path.join(data_dir, 'test', 'dogs', f'dog.{i-10000}.jpg'))