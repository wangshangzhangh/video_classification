import os
from PIL import Image
import cv2

# 视频帧提取并保存成图片的函数，每秒取一帧，最多处理前10分钟
def extract_and_save_frames(video_path, output_dir, max_minutes=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    frame_interval = int(fps)  # 每秒一帧的间隔
    max_frames = int(fps * 60 * max_minutes)  # 最多处理的帧数
    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:  # 只保留每秒的一帧
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB颜色空间
            frame = Image.fromarray(frame)
            frame.save(os.path.join(output_dir, f"frame_{saved_frame_count:04d}.png"))
            saved_frame_count += 1
        frame_count += 1

    cap.release()

# 视频路径
video_path = '/home/hp/data/xufeng/公开数据集/cholec80/videos/video80.mp4'
# 输出图片的文件夹路径
output_dir = '/home/hp/ProcessingData/test'

# 提取并保存视频帧
extract_and_save_frames(video_path, output_dir)
