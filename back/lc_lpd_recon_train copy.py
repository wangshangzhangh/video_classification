import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import random
import torch.nn.functional as F

# 自定义数据集类
class ImageDataset(Dataset):
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

# 从pkl文件中加载数据
with open('data/lc_lpd_recog_image_paths_and_labels.pkl', 'rb') as f:
    data = pickle.load(f)

# 划分训练集和验证集
random.shuffle(data)
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_data, val_data = random_split(data, [train_size, val_size])

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

# 创建数据集和数据加载器
train_dataset = ImageDataset(train_data, transform=data_transforms['train'])
val_dataset = ImageDataset(val_data, transform=data_transforms['val'])

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=20),
    'val': DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=20)
}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = OpenSetModel(num_classes=2)
model = nn.DataParallel(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, open_set_output = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Open set loss
                    open_set_labels = torch.zeros_like(open_set_output)
                    open_set_loss = F.binary_cross_entropy_with_logits(open_set_output, open_set_labels)

                    loss = loss + open_set_loss
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

model = train_model(model, criterion, optimizer, num_epochs=25)

# 保存模型
torch.save(model.state_dict(), 'data/lc_lpd_recon_model.pth')
