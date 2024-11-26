import os
import pickle

# 定义文件夹路径
endo_folder = '/home/hp/ProcessingData/VideoClassification/lc'
non_endo_folder = '/home/hp/ProcessingData/VideoClassification/lpd'

# 准备数据列表
data = []

# 处理endo文件夹
for filename in os.listdir(endo_folder):
    file_path = os.path.join(endo_folder, filename)
    if os.path.isfile(file_path):
        data.append((file_path, 1))

# 处理non-endo文件夹
for filename in os.listdir(non_endo_folder):
    file_path = os.path.join(non_endo_folder, filename)
    if os.path.isfile(file_path):
        data.append((file_path, 0))

# 保存数据到pkl文件
output_pkl = 'data/lc_lpd_recon_image_paths_and_labels.pkl'
with open(output_pkl, 'wb') as f:
    pickle.dump(data, f)

print(f'Data saved to {output_pkl}')
