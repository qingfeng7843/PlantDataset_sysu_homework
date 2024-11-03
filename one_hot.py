
import pandas as pd

# 加载数据集
df = pd.read_csv('D:/study/thirdup/cv/plant_dataset/val/val_label.csv')

# 定义类别映射，每个原始标签映射到一个包含目标类别的列表
label_mapping = {
    'complex': ['complex'],
    'frog_eye_leaf_spot': ['frog_eye_leaf_spot'],
    'frog_eye_leaf_spot complex': ['frog_eye_leaf_spot', 'complex'],
    'healthy': ['healthy'],
    'powdery_mildew': ['powdery_mildew'],
    'rust': ['rust'],
    'rust complex': ['rust', 'complex'],
    'rust frog_eye_leaf_spot': ['rust', 'frog_eye_leaf_spot'],
    'scab': ['scab'],
    'scab frog_eye_leaf_spot': ['scab', 'frog_eye_leaf_spot'],
    'scab frog_eye_leaf_spot complex': ['scab', 'frog_eye_leaf_spot', 'complex'],
    'powdery_mildew complex': ['powdery_mildew', 'complex']
}

# 定义目标类别
target_categories = ['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew']
# 将每个标签映射到其对应的目标类别列表
df['mapped_labels'] = df['labels'].map(label_mapping)

# 初始化一个新的DataFrame来存放独热编码结果
encoded_labels = pd.DataFrame(0, index=df.index, columns=target_categories)

# 遍历每个标签组合，将相应位置设为1
for idx, categories in enumerate(df['mapped_labels']):
    for category in categories:
        encoded_labels.loc[idx, category] = 1

# 合并独热编码后的标签和原始数据集（去除原始和映射标签列）
df_encoded = pd.concat([df.drop(['labels', 'mapped_labels'], axis=1), encoded_labels], axis=1)

# 将编码后的结果保存为新的 CSV 文件
df_encoded.to_csv('D:/study/thirdup/cv/plant_dataset/val/encoded_val_dataset.csv', index=False)

print("Encoded data with multi-label has been saved successfully!")
