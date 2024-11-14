import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(cm, label, save_path, normalize=False):
    # Normalize the confusion matrix if needed
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = f'Normalized Confusion Matrix for {label}'
    else:
        title = f'Confusion Matrix for {label}'

    # Plotting the heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}.png')
    plt.show()

def main(gt_path, pred_path, save_path):
    # Load data from CSV
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    # Get class labels
    labels = gt_df.columns[1:].tolist()

    # Set images as index
    gt_df.set_index('images', inplace=True)
    pred_df.set_index('images', inplace=True)

    # Get image names
    image_names = gt_df.index.tolist()

    # Initialize counter for correct predictions
    correct_predictions = 0

    # Loop over each image and check prediction accuracy
    for image_name in image_names:
        gt_labels = gt_df.loc[image_name].values
        pred_labels = pred_df.loc[image_name].values

        if np.array_equal(gt_labels, pred_labels):
            correct_predictions += 1

    accuracy = correct_predictions / len(image_names)
    print(f'Accuracy: {accuracy:.4f}')

    # For each label, calculate confusion matrix
    for label in labels:
        cm = np.zeros((2, 2), dtype=int)  # Confusion matrix for binary classification

        for image_name in image_names:
            gt_labels = gt_df.loc[image_name].values
            pred_labels = pred_df.loc[image_name].values

            true = int(gt_labels[labels.index(label)] > 0.5)  # Ensure the value is either 0 or 1
            pred = int(pred_labels[labels.index(label)] > 0.5)  # Ensure the value is either 0 or 1
            cm[true, pred] += 1  # Use a tuple for multi-dimensional indexing

        # Plot confusion matrix for the current label
        plot_confusion_matrix(cm, label, save_path, normalize=False)
        plot_confusion_matrix(cm, label, save_path, normalize=True)


if __name__ == "__main__":
    gt_path = '/media/tiankanghui/plant_mymodel/plant_dataset/test/labels.csv'  # Ground truth labels file path
    pred_path = '/media/tiankanghui/PlantDataset_sysu_homework-main/results/predict_results/predictions_10_10_05.csv'  # Prediction results file path
    save_path = '/media/tiankanghui/PlantDataset_sysu_homework-main/results/confusion_matrix/pre_10_10_05'  # Path to save confusion matrices
    main(gt_path, pred_path, save_path)
