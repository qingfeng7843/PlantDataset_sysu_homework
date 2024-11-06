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

from model_save_load import save_model, load_model
from dataset import Plant_Dataset
from train import train, train_val, validation
from test import model_test
gc.collect() 
torch.cuda.empty_cache()

test_folder_path = "/media/tiankanghui/plant_mymodel/plant_dataset/test/images"
test_path = os.listdir(test_folder_path)
test_csv_path = "/media/tiankanghui/plant_mymodel/plant_dataset/test/labels.csv"
test_df = pd.read_csv(test_csv_path)
test_df_cp = test_df.copy()

test_transforms = A.Compose([A.Normalize(),ToTensorV2()])
test_data = Plant_Dataset(test_folder_path,test_path,test_df_cp, transforms=test_transforms,train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False,num_workers=0)


model = models.swin_b(weights=None)
model.head = nn.Linear(1024, 6)
model = model.cuda()  # 先将模型移动到GPU
model = nn.DataParallel(model)
checkpoint = torch.load("/media/tiankanghui/plant_mymodel/pretrained/swin_b_train_num0.pth")
model.load_state_dict(checkpoint['model_state_dict'])



images,predictions, acc = model_test(model,test_loader)

print("Model Swin_b Test Acc:" ,acc)


