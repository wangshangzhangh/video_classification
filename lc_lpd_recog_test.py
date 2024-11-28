import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import logging
import os
from model import OpenSetModel
from dataset import VideoFrameDataset
import torch.nn.functional as F

# 设置日志文件
# 定义日志目录和文件路径
log_dir = '/home/hp/project/VideoClassification/logs'
log_file = os.path.join(log_dir, 'lc_lpd_recon_test.log')
if os.path.exists(log_file):
    os.remove(log_file)

# 如果目录不存在，则创建
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, filename='logs/lc_lpd_recon_test.log')

# 视频帧提取函数，最多处理前10分钟，每秒取一帧
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
model_path = 'saved_model/lc_lpd_recon_model_full.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = OpenSetModel(num_classes=2)
model = nn.DataParallel(model)
model = torch.load(model_path)
model = model.to(device)
model.eval()

# 视频路径
# video_path = '/home/hp/ProcessingData/VideoClassification/endo_original/Colonoscopic/00001.mp4'
video_path = '/home/hp/ProcessingData/VideoClassification/non_endo_video/3197622-hd_1920_1080_25fps.mp4'
# video_path = '/home/hp/data/xufeng/公开数据集/cholec80/videos/video22.mp4'
# video_path = '/home/hp/data/xufeng/北医三院普外/LPD/merged/2023-03-27_224123/2023-03-27_224123.mp4'
batch_size = 128
workers = 20

# 对视频帧进行推理
all_preds = []
all_probs = []
open_set_threshold = 0.5  # 定义开放集检测阈值
open_set_flags = []

for frames in extract_frames_in_batches(video_path, batch_size, max_minutes=10):
    dataset = VideoFrameDataset(frames, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs, open_set_output = model(inputs)
            logging.info(f'Outputs: {outputs}, Open set output: {open_set_output}')
            open_set_labels = torch.zeros_like(open_set_output)
            open_set_loss = F.binary_cross_entropy_with_logits(open_set_output, open_set_labels)
            logging.info(f'Open set loss: {open_set_loss.item()}')
            probs = nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            open_set_flags.extend((torch.sigmoid(open_set_output) > open_set_threshold).cpu().numpy())

# 统计每个类的数量和比例
class_counts = [0, 0]
open_set_count = 0
total_count = len(all_preds)

for pred, open_set_flag in zip(all_preds, open_set_flags):
    if open_set_flag:
        open_set_count += 1
    else:
        class_counts[pred] += 1

class_ratios = [count / total_count for count in class_counts]
open_set_ratio = open_set_count / total_count

logging.info(f'Class 0 count: {class_counts[0]}, Ratio: {class_ratios[0]:.4f}')
logging.info(f'Class 1 count: {class_counts[1]}, Ratio: {class_ratios[1]:.4f}')
logging.info(f'Open set (unknown) count: {open_set_count}, Ratio: {open_set_ratio:.4f}')

# 将每一类的概率值输出到文件
output_file_path = 'data/predicts.txt'
with open(output_file_path, 'w') as f:
    for pred, prob, open_set_flag in zip(all_preds, all_probs, open_set_flags):
        f.write(f'Class: {pred}, Probabilities: {prob.tolist()}, Open set flag: {open_set_flag}\n')

logging.info(f'Probabilities saved to {output_file_path}')
