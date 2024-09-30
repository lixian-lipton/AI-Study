import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


data_dir = r"C:\Users\lipton\Downloads\dogs-vs-cats-redux-kernels-edition\train"
test_dir = r"C:\Users\lipton\Downloads\dogs-vs-cats-redux-kernels-edition\test"

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
device = torch.device('cuda')

# 对自带的 ResNet50 进行微调
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
# model.load_state_dict(torch.load('res-easy_model_state_dict.pth')) #####################
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("模型已加载！")

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
    torch.save(model.state_dict(), 'res-easy_model_state_dict.pth')
    print("本次参数已更新！")

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


