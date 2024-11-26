import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np

# 自定义数据集类
class SurgeryDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 定义数据转换
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 从pkl文件中加载数据
with open('data/lc_lpd_recog_image_paths_and_labels.pkl', 'rb') as f:
    data = pickle.load(f)

# 划分训练集和验证集
np.random.shuffle(data)
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_data = data[:train_size]
val_data = data[train_size:]

train_dataset = SurgeryDataset(train_data, transform=data_transforms['train'])
val_dataset = SurgeryDataset(val_data, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=40)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=40)

# 定义模型架构
class OpenSetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(OpenSetModel, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(resnet.fc.in_features, num_classes)
        self.open_set_layer = nn.Linear(resnet.fc.in_features, 1)  # For open set detection

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_outputs = self.classifier(features)
        open_set_output = self.open_set_layer(features)
        return class_outputs, open_set_output

# 初始化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = OpenSetModel(num_classes=2)
model = nn.DataParallel(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def adjust_loss_weights(class_loss, open_set_loss):
    # 根据具体情况调整权重
    class_weight = 1.0
    open_set_weight = 1.0

    # 动态调整权重
    if class_loss > open_set_loss:
        class_weight = open_set_loss / class_loss
    else:
        open_set_weight = class_loss / open_set_loss

    return class_weight, open_set_weight

# 训练过程
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, open_set_threshold=0.5):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 验证模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in (train_loader if phase == 'train' else val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    class_outputs, open_set_output = model(inputs)
                    _, preds = torch.max(class_outputs, 1)
                    class_loss = criterion(class_outputs, labels)

                    # Open set loss
                    open_set_labels = torch.zeros_like(open_set_output)
                    open_set_loss = F.binary_cross_entropy_with_logits(open_set_output, open_set_labels)
                    print("class_loss: ", class_loss.item(), "open_set_loss: ", open_set_loss.item())

                    # 动态调整损失权重
                    class_weight, open_set_weight = adjust_loss_weights(class_loss.item(), open_set_loss.item())
                    loss = class_weight * class_loss + open_set_weight * open_set_loss

                    # 反向传播
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset) if phase == 'train' else len(val_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset) if phase == 'train' else len(val_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)

# 保存模型
torch.save(model.state_dict(), 'model/lc_lpd_recon_model.pth')
