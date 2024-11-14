import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.preprocessing import label_binarize

def plot_pr_curves_and_find_best_thresholds(y_true, y_pred, labels, save_path):
    plt.figure(figsize=(10, 8))

    best_thresholds = []
    
    # 绘制每个类别的PR曲线
    for i, label in enumerate(labels):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
        
        # 计算 F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall)
        
        # 找到 F1 score 最大的阈值
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_thresholds.append(best_threshold)
        
        # 绘制PR曲线
        plt.plot(recall, precision, label=f'PR curve for {label}')
    
    # 绘制图表
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_path}/pr_curve.png')
    plt.show()

    return best_thresholds

def main(gt_path, pred_path, save_path):
    # 加载数据
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    # 获取标签列
    labels = gt_df.columns[1:].tolist()

    # 使用图像名称匹配真实标签和预测标签
    gt_df.set_index('images', inplace=True)
    pred_df.set_index('images', inplace=True)

    # 获取图像名称列表
    image_names = gt_df.index.tolist()

    # 获取真实标签和预测标签
    gt_labels = np.array([gt_df.loc[image_name].values for image_name in image_names])
    pred_labels = np.array([pred_df.loc[image_name].values for image_name in image_names])

    # 绘制PR曲线并找到每个类别的最佳阈值
    best_thresholds = plot_pr_curves_and_find_best_thresholds(gt_labels, pred_labels, labels, save_path)

    # 打印每个类别的最佳阈值
    for i, label in enumerate(labels):
        print(f'Best threshold for {label}: {best_thresholds[i]}')

if __name__ == "__main__":
    gt_path = '/media/tiankanghui/plant_mymodel/plant_dataset/val/labels.csv'  # 真实标签文件路径
    pred_path = '/media/tiankanghui/PlantDataset_sysu_homework-main/results/predict_probability/val.csv'  # 预测结果文件路径
    save_path = '/media/tiankanghui/PlantDataset_sysu_homework-main/results/PR_curves'  # 保存PR曲线的路径
    main(gt_path, pred_path, save_path)

