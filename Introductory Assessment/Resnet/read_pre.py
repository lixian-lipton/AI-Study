import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# 重新定义模型结构
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
device = torch.device('cuda')
model.load_state_dict(torch.load('model_state_dict.pth'))
model = model.to(device)

# 准备测试数据
test_dir = r"C:\Users\lipton\Downloads\dogs-vs-cats-redux-kernels-edition\test"

# 测试
# 数据预处理
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc=f'Test: ', leave=False):
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 统计正确预测的样本数

accuracy = 100 * correct / total
print(f'测试集准确率: {accuracy:.2f}%')