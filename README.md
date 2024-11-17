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

#### 1.数据集处理

第一步，将数据集进行处理，使用代码one_hot.py，进行独热编码，将类别标签缩减为6个基本种类。数据集格式处理完后格式如下：

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

#### 2.多卡训练设置

第二步，为了解决batch_size较大时显存溢出的问题，修改trian.py，使用多卡训练，直接在model.py中设置，示例如下：
```bash
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用0和1号显卡
```
#### 3.损失函数优化

目前的predictions.csv是模型使用torch.nn.BCELoss()时训练所得到的结果，正确率已经达到了87.78%。后续打算优化损失函数，如Focal_Loss与ArcFace_Loss。

#### 4.backbone网络搭建

目前已经从[torchvision官网中](https://pytorch.org/vision/stable//_modules/torchvision/models/swin_transformer.html#Swin_B_Weights)调取swin_transformer的源码，并进行修改，成功搭建swin_b的网络架构，详细内容请见network.py。

#### 5.ArcFaceLoss引入

成功引入[ArcFaceLoss](https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10)，由于我们的任务是多标签分类，因此最后线性分类头的输出不再经过softmax输出概率，而是使用sigmoid函数。同理，对于ArcFaceLoss的计算也是如此，在计算出对应的[cosx1, cosx2, ...,cos(x6+m)]后，经过sigmoid将其转化为概率值，然后与对应gt label计算二元交叉损失。

#### 6.选择超参数

接下来的任务是确定合适的超参数，主要包括FocalLoss与ArcFaceLoss的占比，固定FocalLoss系数为1，ArcFaceLoss的系数由0.1开始逐步增加，测试结果；此外，还有ArcFaceLoss计算中的尺度缩放因子s，与角度裕度m。

#### 7.预训练模型设置

对模型从头开始训练了一百轮，效果不是很好，因此决定还是要采用预训练模型，由于我们只是对网络架构在head处进行了修改，因此，backbone的预训练模型权重是可以使用的，后续的训练，都是要在预训练模型的基础上进行的。

#### 8.分辨率设置

之前以为分辨率越高越好，所以设置的是512*512，但读论文发现，对于图像分类而言，主流的分辨率还是384*384,详细请见[深度学习图像大小选择：为什么384x384成为主流尺寸？](https://developer.baidu.com/article/details/1890241)。
