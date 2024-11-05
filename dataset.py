from torch.utils.data import Dataset
import cv2
import numpy as np
import os
#import torch_optimizer as optimizer

class Plant_Dataset(Dataset):
    # 初始化数据集
    def __init__(self, folder_path, data_paths, data_df, size=512, transforms=None, train=True):
        # 数据集的文件夹路径
        self.folder_path = folder_path
        # 图像文件路径列表
        self.data_paths = data_paths
        # 存储图像标签信息的 DataFrame
        self.data_df = data_df
        # 数据预处理变换函数
        self.transforms = transforms
        # 是否为训练集的标志
        self.train = train
        # 图像的目标大小（长和宽）
        self.size = size

    # 根据索引获取数据项
    def __getitem__(self, idx):
        # 组合文件夹路径和图像文件名，获取图像的完整路径
        img_path = os.path.join(self.folder_path, self.data_paths[idx])
        # 使用 OpenCV 读取图像
        image = cv2.imread(img_path)
        # 将图像从 BGR 转换为 RGB 格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 将图像调整为指定大小
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        # 将图像转换为 NumPy 数组
        image = np.asarray(image)

        # 下面的代码用于获取标签
        j = 0
        vector = [0] * 6  # 初始化标签向量，长度为6
        # 从 DataFrame 中获取与当前图像路径对应的标签值
        values = self.data_df.loc[self.data_df['images'] == self.data_paths[idx]].values
        # 从 DataFrame 中提取特定列的标签值（假设列索引3到8包含标签）
        for i in range(1, 7):
            num = values[0][i]  # 获取标签值
            vector[j] = num  # 将标签值放入向量中
            j = j + 1  # 移动到下一个位置
        
        # 将标签向量转换为 NumPy 数组
        vector = np.asarray(vector)

        # 如果有数据变换，则应用于图像
        if self.transforms:
            # 检查图像是否为 NumPy 数组，如果不是则转换
            if not isinstance(image, np.ndarray):
                image = image.numpy()

            # 应用预处理变换，并获取处理后的图像
            image = self.transforms(image=image)['image']

        # 返回训练或验证数据
        if self.train:
            return image, vector  # 返回图像和标签向量
        else:
            return image, vector, self.data_paths[idx]  # 返回图像、标签向量和图像路径（用于测试集）

    # 返回数据集的大小
    def __len__(self):
        return len(self.data_paths)  # 返回图像文件路径的数量
