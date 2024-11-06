# PlantDataset_sysu_homework

![License](https://github.com/qingfeng7843/PlantDataset_sysu_homework) ![Issues](https://github.com/qingfeng7843/PlantDataset_sysu_homework)

本项目为中山大学计算机视觉课程作业，旨在记录学习过程，总结经验与教训。

## 目录
- [简介](#简介)
- [特性](#特性)
- [安装](#安装)
- [用法示例](#用法示例)
- [测试](#测试)

## 简介
在这里详细介绍项目的背景、目标和使用场景。

## 特性
- 📌 主要功能 1
- 🔍 主要功能 2
- ⚙️ 主要功能 3

## 安装

### 先决条件
列出需要的依赖项，如：
- Python 3.8+
- torch
- torchvision
- Pillow
- pandas
- tqdm
- scikit-learn
- typing
- argparse
- matplotlib

### 安装步骤
克隆仓库并安装依赖项。

```bash
git clone https://github.com/qingfeng7843/PlantDataset_sysu_homework.git
cd PlantDataset_sysu_homework
pip install -r requirements.txt
```

1.第一步，将数据集进行处理，使用代码one_hot.py，进行独热编码，将类别标签缩减为6个基本种类。数据集格式处理完后格式如下：

```
plant_dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels.csv
├── val/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── ...
│   └── labels.csv
└── test/
    ├── images/
    │   ├── image1.jpg
    │   └── ...
    └── labels.csv
```

2.第二步，为了解决batch_size较大时显存溢出的问题，修改trian.py，使用多卡训练，直接在model.py中设置，示例如下：
```bash
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用0和1号显卡
```
3.目前的predictions.csv是模型使用torch.nn.BCELoss()时训练所得到的结果，正确率已经达到了87.78%。后续打算优化损失函数，如Focal_Loss与ArcFace_Loss。

4.目前已经从torchvision官网中[Related Link](https://pytorch.org/vision/stable//_modules/torchvision/models/swin_transformer.html#Swin_B_Weights)调取swin_transformer的源码，并进行修改，成功搭建swin_b的网络架构，详细内容请见network.py。

