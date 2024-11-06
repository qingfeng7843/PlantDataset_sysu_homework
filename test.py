import gc
import torch
from tqdm import tqdm
import pandas as pd  # 引入 pandas 库

def model_test(model, test_loader, output_csv_path="/media/tiankanghui/plant_mymodel/predictions_results/predictions.csv"):
    model.eval()
    images = []
    predictions = []
    threshold = 0.5
    correct = 0
    target_sum = 0

    # 类别标签列名
    class_labels = ["scab", "healthy", "frog_eye_leaf_spot", "rust", "complex", "powdery_mildew"]
    results = []

    for i, (img, target, img_name) in enumerate(tqdm(test_loader, desc="Inference Progress")):
        img = img.float().cuda()
        target = target.float().cuda()
        
        with torch.no_grad():
            output = torch.sigmoid(model(img)).float()

        # 应用阈值
        output = torch.where(output > threshold, 1.0, 0.0)

        # 转换为 numpy 格式并存储结果
        output_np = output.cpu().numpy()
        for idx, name in enumerate(img_name):
            # 构建每一行的内容：图像名和对应的类别预测结果
            row = [name] + output_np[idx].tolist()
            results.append(row)

        res = output == target
        correct += sum(1 for tensor in res if not (False in tensor))
        
        target_sum += len(target)
        avg_acc = correct / target_sum

        del img, output
        gc.collect()
        torch.cuda.empty_cache()

    # 创建 DataFrame 并保存为 CSV
    df = pd.DataFrame(results, columns=["images"] + class_labels)
    df.to_csv(output_csv_path, index=False)

    return images, predictions, avg_acc
