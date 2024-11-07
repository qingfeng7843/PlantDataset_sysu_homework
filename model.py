import torchvision.models as models
#!pip install albumentations==0.4.6
#!pip install -U git+https://github.com/albu/albumentations > /dev/null
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Blur, MotionBlur, HorizontalFlip, VerticalFlip, GaussNoise, IAASharpen, RandomBrightnessContrast
from torchvision import transforms as T
import gc
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd

from focal_loss import FocalLoss
from network import SwinTransformer
from model_save_load import save_model, load_model
from dataset import Plant_Dataset
from train import train, train_val, validation

gc.collect() 
torch.cuda.empty_cache()


train_folder_path = "/media/tiankanghui/plant_mymodel/plant_dataset/train/images"
train_path = os.listdir(train_folder_path)
train_csv_path = "/media/tiankanghui/plant_mymodel/plant_dataset/train/labels.csv"
train_df = pd.read_csv(train_csv_path)
train_df_cp = train_df.copy()

val_folder_path = "/media/tiankanghui/plant_mymodel/plant_dataset/val/images"
val_path = os.listdir(val_folder_path)
val_csv_path = "/media/tiankanghui/plant_mymodel/plant_dataset/val/labels.csv"
val_df = pd.read_csv(val_csv_path)
val_df_cp = val_df.copy()

BS = 64
EPOCHS = 100
train_num = 2
transforms = A.Compose([Blur(),MotionBlur(), HorizontalFlip(), VerticalFlip(), GaussNoise(),
                        IAASharpen(),RandomBrightnessContrast(),
                        A.Normalize(),ToTensorV2()])
val_transforms = A.Compose([A.Normalize(),ToTensorV2()])

train_data = Plant_Dataset(train_folder_path,train_path, train_df_cp, transforms=transforms,train=True)
loader_train = DataLoader(train_data, batch_size=BS, shuffle=True,num_workers=4)
val_data = Plant_Dataset(val_folder_path,val_path, val_df_cp, transforms=transforms,train=True) # !!!!!
loader_val = DataLoader(val_data,batch_size=BS, shuffle=False,num_workers=4)

# model = models.swin_b(weights=None)
# model.load_state_dict(torch.load("/media/tiankanghui/plant_mymodel/pretrained/swin_b-68c6b09e.pth"))
# model.head = nn.Linear(1024, 6)
# 加载预训练模型。但是，这个加载的权重文件 head 层的权重为[1000, 1024]，我们期望的为[6, 1024]，因此在模型中手动修改了 head 层。

# 加载预训练权重
pretrained_weights = torch.load('/media/tiankanghui/PlantDataset_sysu_homework-main/pretrained/swin_b-68c6b09e.pth')

# 初始化当前模型
model = SwinTransformer()  # 将其替换为您的模型初始化代码
model_dict = model.state_dict()

# 创建一个新的字典，用于保存匹配的权重
new_state_dict = {}

# 遍历预训练权重
for name, param in pretrained_weights.items():
    if name in model_dict and model_dict[name].shape == param.shape:
        # 如果名称和形状匹配，则使用预训练权重
        new_state_dict[name] = param
    else:
        print(f"Skipping layer: {name} - shape mismatch or not in current model.")

# 更新模型权重
model_dict.update(new_state_dict)
model.load_state_dict(model_dict)

#model = SwinTransformer()
#model.load_state_dict(torch.load("/media/tiankanghui/plant_mymodel/pretrained/swin_b-68c6b09e.pth"))

# 指定要使用的GPU设备，例如使用第0和第1张显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 使用0和1号显卡

# 将模型移动到GPU并启用多卡支持
if torch.cuda.is_available():
    model = nn.DataParallel(model)  # 启用多卡训练
    model = model.cuda()  # 将模型移动到GPU

#model = load_model("/home/jack/Project/zdf/ComputerVision/plant_kaggle/Torch/pretrained/swin_b_train_num0.pth", model, optimi)
#model.train()
model.cuda()
criterion = FocalLoss()
criterion.cuda()
optimi = optim.AdamW(model.parameters(), lr=1e-4)
print(f"Model is using device: {next(model.parameters()).device}")
csv = f"/media/tiankanghui/PlantDataset_sysu_homework-main/train_log/train_{train_num}.csv"
model,train_loss,train_acc,val_loss,val_acc = train_val(model,loader_train,loader_val,optimi,criterion,EPOCHS,csv)

model_path = f"/media/tiankanghui/PlantDataset_sysu_homework-main/pretrained/swin_b_train_num{train_num}.pth"
save_model(model, optimi, EPOCHS, train_loss, val_loss, train_acc, val_acc, model_path)
