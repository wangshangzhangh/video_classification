import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2

# 自定义视频帧数据集类
class VideoFrameDataset(Dataset):
    def __init__(self, frames, transform=None):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
        return frame

# 视频帧提取函数，每秒取一帧，最多处理前10分钟
def extract_frames_in_batches(video_path, batch_size, max_minutes=10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    frame_interval = int(fps)  # 每秒一帧的间隔
    max_frames = int(fps * 60 * max_minutes)  # 最多处理的帧数
    batch = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:  # 只保留每秒的一帧
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            batch.append(frame)
            if len(batch) == batch_size:
                yield batch
                batch = []
        frame_count += 1
    if batch:
        yield batch
    cap.release()

# 定义数据转换
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载训练好的模型
model_path = 'model/endo_recon_model.pth'
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = nn.DataParallel(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 视频路径
video_path = '/home/hp/ProcessingData/VideoClassification/endo_original/merged/merged_20240713_161441_part2.mp4'
batch_size = 128
workers = 40

# 对视频帧进行推理
all_preds = []
all_probs = []
for frames in extract_frames_in_batches(video_path, batch_size, max_minutes=10):
    dataset = VideoFrameDataset(frames, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

# 统计每个类的数量和比例
class_counts = [0, 0]
total_count = len(all_preds)
for pred in all_preds:
    class_counts[pred] += 1

class_ratios = [count / total_count for count in class_counts]

print(f'Class 0 count: {class_counts[0]}, Ratio: {class_ratios[0]:.4f}')
print(f'Class 1 count: {class_counts[1]}, Ratio: {class_ratios[1]:.4f}')

# 将每一类的概率值输出到文件
output_file_path = 'data/predicts.txt'
with open(output_file_path, 'w') as f:
    for pred, prob in zip(all_preds, all_probs):
        f.write(f'Class: {pred}, Probabilities: {prob.tolist()}\n')

print(f'Probabilities saved to {output_file_path}')
