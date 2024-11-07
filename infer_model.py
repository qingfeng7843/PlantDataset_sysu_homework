import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T
import gc
import torch 
from torch.utils.data import  DataLoader
import torch.nn as nn
import os
import pandas as pd
from dataset import Plant_Dataset
from test import model_test
from network import SwinTransformer
import argparse

# Clear memory cache
gc.collect()
torch.cuda.empty_cache()

def main(args):
    # Load test data
    test_folder_path = "/media/tiankanghui/plant_mymodel/plant_dataset/test/images"
    test_csv_path = "/media/tiankanghui/plant_mymodel/plant_dataset/test/labels.csv"
    test_df = pd.read_csv(test_csv_path)
    
    test_transforms = A.Compose([A.Normalize(), ToTensorV2()])
    test_data = Plant_Dataset(test_folder_path, os.listdir(test_folder_path), test_df, transforms=test_transforms, train=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    # Load model and checkpoint
    model = SwinTransformer().cuda()
    model = nn.DataParallel(model)
    
    # Load pretrained weights from specified path
    checkpoint = torch.load(args.pretrained_weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run test and save results
    images, predictions, acc = model_test(model, test_loader, args.output_csv)
    print("Model Swin_b Test Accuracy:", acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model with specified parameters.")
    parser.add_argument("--pretrained_weights", type=str, required=True, help="Path to the pretrained weights")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the predictions CSV file")
    
    args = parser.parse_args()
    
    main(args)
