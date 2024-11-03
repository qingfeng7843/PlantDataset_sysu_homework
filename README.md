# PlantDataset_sysu_homework

![License](https://github.com/qingfeng7843/PlantDataset_sysu_homework) ![Issues](https://github.com/qingfeng7843/PlantDataset_sysu_homework)

本项目为中山大学计算机视觉课程作业，旨在记录学习过程，总结经验与教训。

## 目录
- [简介](#简介)
- [特性](#特性)
- [安装](#安装)
- [用法示例](#用法示例)
- [API 文档](#api-文档)
- [测试](#测试)
- [贡献](#贡献)
- [许可](#许可)
- [联系我们](#联系我们)

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
- [其他依赖项](链接)

### 安装步骤
克隆仓库并安装依赖项。

```bash
git clone https://github.com/你的用户名/项目名称.git
cd 项目名称
pip install -r requirements.txt
```

1.第一步，将数据集进行处理，使用代码one_hot.py，进行独热编码，将类别标签缩减为6个基本种类。

2.第二步，为了解决batch_size=32时显存溢出的问题，修改trian.py，使用多卡训练，训练指令示例如下：
```bash
python train.py --gpu_ids "1,2" --data-path /media/tiankanghui/Plant-Pathology-2021-master/plant_dataset
```
