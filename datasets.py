import torchvision
from torchvision import transforms
import os
import torch
import numpy as np
from PIL import Image

#使用文件操作将所有文件按比例分为测试集和训练集
data_dir="data/flower_photos"
classes=['daisy','dandelion','roses','sunflowers','tulips']
train_dir=os.path.join(data_dir,'train')
test_dir=os.path.join(data_dir,'test')

#创建train文件下的classer文件(test等同)  重复执行会导致图片再被拷贝一次
# for cls in classes:
#     os.makedirs(os.path.join(train_dir,cls),exist_ok=True)
#     os.makedirs(os.path.join(test_dir,cls),exist_ok=True)
#
# for cls in classes:
#     src_dir=os.path.join(data_dir,cls)
#     files=os.listdir(src_dir)
#     np.random.shuffle(files)
#     train_files=files[:int(len(files)*0.75)]
#     test_files=files[int(len(files)*0.75):]
#     for files in train_files:
#         src_path=os.path.join(src_dir,files)
#         dst_path=os.path.join(train_dir,cls,files)
#         print(src_path)
#         img=Image.open(src_path)
#         img.save(dst_path)
#     for files in test_files:
#         src_path=os.path.join(src_dir,files)
#         dst_path=os.path.join(test_dir,cls,files)
#         print(src_path)
#         img=Image.open(src_path)
#         img.save(dst_path)

# 将图片数据用transfroms转化为张量数据。
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])
# train_data=torchvision.datasets.ImageFolder('data/flower_photos/train',transform=transform)
# test_data=torchvision.datasets.ImageFolder('data/flower_photos/test',transform=transform)
train_data=torchvision.datasets.ImageFolder('data/train',transform=transform)
test_data=torchvision.datasets.ImageFolder('data/test',transform=transform)
# print(train_data.targets)     # 输出标签
print(train_data.class_to_idx) # 输出数据和索引标签