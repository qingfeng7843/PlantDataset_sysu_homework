#!pip install albumentations==0.4.6
#!pip install -U git+https://github.com/albu/albumentations > /dev/null
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gc
import torch
from torch.utils.data import  DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import argparse

from focal_loss import FocalLoss
from network import SwinTransformer
from model_save_load import save_model
from dataset import Plant_Dataset
from train import train_val



def main(args):
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

    # 定义训练和验证数据增强
    transforms = A.Compose([
        A.Blur(), A.MotionBlur(), A.HorizontalFlip(), A.VerticalFlip(), A.GaussNoise(),
        A.IAASharpen(), A.RandomBrightnessContrast(), A.Normalize(), ToTensorV2()
    ])
    val_transforms = A.Compose([A.Normalize(), ToTensorV2()])

    # 加载数据集,train=True意义请见相关函数
    train_data = Plant_Dataset(train_folder_path, train_path, train_df_cp, transforms=transforms,
                               train=True)
    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_data = Plant_Dataset(val_folder_path, val_path, val_df_cp, transforms=val_transforms, train=True)
    loader_val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 初始化当前模型
    model = SwinTransformer()  # 将其替换为您的模型初始化代码

    # 如果提供了预训练权重路径，则加载权重
    if args.pretrained_weights:
        print(f"Loading pretrained weights from {args.pretrained_weights}")
        pretrained_weights = torch.load(args.pretrained_weights)
        model_dict = model.state_dict()

        # 创建一个新的字典，用于保存匹配的权重
        new_state_dict = {}
        for name, param in pretrained_weights.items():
            if name in model_dict and model_dict[name].shape == param.shape:
                new_state_dict[name] = param
            else:
                print(f"Skipping layer: {name} - shape mismatch or not in current model.")

        # 更新模型权重
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
    else:
        print("No pretrained weights provided, initializing model from scratch.")

    # 设置设备和多卡支持
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()

    # 定义损失函数和优化器
    criterion = FocalLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 开始训练
    print(f"Model is using device: {next(model.parameters()).device}")
    csv_path = args.csv_path
    model, train_loss, train_acc, val_loss, val_acc = train_val(
        model, loader_train, loader_val, optimizer, criterion, args.epochs, csv_path
    )

    # 保存模型
    model_path = args.model_save_path
    save_model(model, optimizer, args.epochs, train_loss, val_loss, train_acc, val_acc, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--train_num", type=int, default=2, help="Training identifier number")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Path to pretrained weights (optional)")
    parser.add_argument("--csv_path", type=str, default=None, help="Directory to save CSV logs")
    parser.add_argument("--model_save_path", type=str, default=None, help="Directory to save trained models")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU IDs to use")
    
    args = parser.parse_args()

    # 设置默认的 csv_path 和 model_save_path
    if args.csv_path is None:
        args.csv_path = f"./train_log/train_{args.train_num}.csv"
    if args.model_save_path is None:
        args.model_save_path = f"./pretrained/swin_b_train_num{args.train_num}.pth"
    
    # 指定要使用的 GPU 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # 使用解析后的参数训练模型
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Training identifier number: {args.train_num}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Pretrained weights path: {args.pretrained_weights}")
    print(f"CSV log path: {args.csv_path}")
    print(f"Model save path: {args.model_save_path}")
    print(f"Using GPUs: {args.gpus}")

    # 继续加载数据、定义模型和训练流程
    main(args)
