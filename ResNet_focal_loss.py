# --- ResNet_focal_loss.py (Model V2.5 - 完整 K-Fold 训练版) ---

import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler
# --- 评估指标库 ---
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from matplotlib import font_manager, rcParams
from sklearn.utils import resample
from typing import Dict, Any, List, Tuple

# 导入自定义模块 (假设这些文件在同一目录下)
# 🚨 确保您的 dataloader.py, cbam_model.py, resnet_fusion_v25.py 文件存在
from dataloader import imgDataset
from resnet_fusion_v25 import ResNetSAM_TabularFusion

# ----------------------------------------------------------------------
# 0. 文件路径和参数定义
# ----------------------------------------------------------------------
# 🚨 请根据您的实际路径和参数进行检查和修改
CHECKPOINT_FILE = "kfold_checkpoint_V25_fusion_AllData.pt"
BEST_MODEL_WTS_FILE = "best_kfold_model_V25_fusion_AllData.pth"
KFOLD_SUMMARY_FILE = "kfold_summary_V25_fusion_AllData.txt"
# 🚨 【新增】DeLong 检验数据保存文件
DELONG_DATA_FILE = "patient_level_fusion_V25_training_run_delong.csv"
# 表格特征数量 (将从 dataloader 中获取)
NUM_TABULAR_FEATURES = 20


# ----------------------------------------------------------------------
# 1. Focal Loss 定义 (损失函数)
# ----------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        labels = labels.view(-1, 1)
        device = labels.device
        alpha = self.alpha.to(device)

        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)

        preds_softmax = preds_softmax.gather(1, labels)
        preds_logsoft = preds_logsoft.gather(1, labels)
        alpha = alpha.gather(0, labels.view(-1))

        focal_weight = torch.pow((1 - preds_softmax), self.gamma)
        focal_loss = -torch.mul(focal_weight, preds_logsoft)
        loss = torch.mul(alpha, focal_loss.t())

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# ----------------------------------------------------------------------
# 2. 训练和测试函数
# ----------------------------------------------------------------------
def train_model_fold(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=30):
    """K-Fold 内部训练函数"""
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e9

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0

        # dataloader: img, info, masks(忽略), target, patient_id
        for img, info, _, target, _ in dataloaders['train']:
            inputs = img.to(device)
            info = info.to(device).float()
            labels = target.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, info)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        epoch_loss = running_loss / dataset_sizes['train']

        # 验证阶段 (只记录最佳模型，不影响训练)
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for img, info, _, target, _ in dataloaders['val']:
                inputs = img.to(device)
                info = info.to(device).float()
                labels = target.to(device)
                outputs = model(inputs, info)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

        val_loss = val_running_loss / dataset_sizes['val']

        print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_loss:.4f} Val Loss: {val_loss:.4f}')

        # 记录最佳权重
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    return best_model_wts, best_loss


def test_model_fold(model, dataloader, num_classes, device) -> Tuple[torch.Tensor, pd.DataFrame]:
    """测试模型性能并返回图像级指标以及用于患者级聚合的 DataFrame。"""
    model.eval()

    fold_data_list = []
    running_corrects = 0
    dataset_size = len(dataloader.dataset)

    with torch.no_grad():
        for img, info, _, target, patient_ids in dataloader:
            inputs = img.to(device)
            info = info.to(device).float()
            labels = target.to(device)

            outputs = model(inputs, info)
            probas = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

            probas_np = probas.cpu().numpy()
            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # 准备患者级聚合数据
            for i in range(len(patient_ids)):
                fold_data_list.append({
                    'PatientID': patient_ids[i],
                    'TrueLabel': labels_np[i],
                    'PredLabel_Image': preds_np[i],
                    'Prob_0': probas_np[i, 0],
                    'Prob_1': probas_np[i, 1],
                    'Prob_2': probas_np[i, 2],
                })

    image_acc = running_corrects.double() / dataset_size
    fold_df = pd.DataFrame(fold_data_list)

    return image_acc, fold_df


# ----------------------------------------------------------------------
# 3. 结果聚合和指标计算函数 (保持不变)
# ----------------------------------------------------------------------
def aggregate_patient_results(df, num_classes=3, aggregation_method='max_prob'):
    # ... (此函数实现与原文件一致) ...
    if aggregation_method == 'max_prob':
        patient_agg_df = df.groupby('PatientID')[['Prob_0', 'Prob_1', 'Prob_2']].max().reset_index()
    else:
        patient_agg_df = df.groupby('PatientID')[['Prob_0', 'Prob_1', 'Prob_2']].mean().reset_index()

    true_labels_df = df.groupby('PatientID')['TrueLabel'].first().reset_index()
    patient_agg_df = pd.merge(patient_agg_df, true_labels_df, on='PatientID')

    prob_cols = ['Prob_0', 'Prob_1', 'Prob_2']
    patient_agg_df['PredLabel'] = patient_agg_df[prob_cols].values.argmax(axis=1)

    true_labels = patient_agg_df['TrueLabel'].tolist()
    pred_labels = patient_agg_df['PredLabel'].tolist()
    probas = patient_agg_df[prob_cols].values

    patient_acc = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
    report = classification_report(true_labels, pred_labels, digits=4, output_dict=True, zero_division=0)

    return patient_acc, cm, report, true_labels, probas, patient_agg_df


def bootstrap_ci(y_true, y_pred, n_iterations=1000, alpha=0.05):
    # ... (此函数实现与原文件一致) ...
    accuracies = []
    data = np.array(list(zip(y_true, y_pred)))

    if len(data) < 20:
        return None, None

    for _ in range(n_iterations):
        sample = resample(data, replace=True, n_samples=len(data))
        y_true_sample = sample[:, 0]
        y_pred_sample = sample[:, 1]

        acc = accuracy_score(y_true_sample, y_pred_sample)
        accuracies.append(acc)

    p = ((alpha / 2.0) * 100)
    lower = np.percentile(accuracies, p)
    p = (100 - (alpha / 2.0) * 100)
    upper = np.percentile(accuracies, p)

    return lower, upper


def print_metrics(name, image_acc, patient_acc, cm, report, probas, labels, num_classes, class_names_short,
                  output_file=None, ci_bounds=None):
    # ... (此函数实现与原文件一致) ...
    output = []

    def log_print(msg):
        print(msg)
        output.append(msg)

    log_print(f"\n==================== {name} 性能报告 ====================")
    log_print(f"图像级准确率: {image_acc:.4f}")

    ci_str = f"({ci_bounds[0]:.4f}, {ci_bounds[1]:.4f})" if ci_bounds and ci_bounds[0] is not None else ""
    log_print(f"患者级准确率: {patient_acc:.4f} (总计 {len(labels)} 例) {ci_str}")

    y_true_binarized = label_binarize(labels, classes=list(range(num_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        log_print(f"  {class_names_short[i]} AUC: {roc_auc[i]:.4f}")

    log_print("\n患者级详细分类报告 (Precision, Recall, F1-Score):")
    report_str = pd.DataFrame(report).transpose().to_string(float_format='%.4f')
    log_print(report_str)

    log_print("\n患者级混淆矩阵 (Confusion Matrix):")
    log_print(str(pd.DataFrame(cm, index=class_names_short, columns=class_names_short)))
    log_print("========================================================\n")

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output))

        # 绘制 ROC 曲线并保存
        plt.figure(figsize=(10, 7))
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{class_names_short[i]} (AUC = {roc_auc[i]:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测 (AUC = 0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title(f'{name} 患者级 ROC 曲线 (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_file.replace('.txt', '_ROC.png'))
        plt.close()
        log_print(f"--- {name} ROC 曲线图已保存为 {output_file.replace('.txt', '_ROC.png')} ---")


# ----------------------------------------------------------------------
# 4. 主程序块 (Main)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 🚨 请检查这些参数是否与您训练时的设置一致
    data_dir = 'D://' # Please change this path to your local data directory
    batch_size = 16
    num_classes = 3
    num_epochs = 30  # K-Fold 每个折叠的训练轮数
    K = 5
    class_names_short = ['慢性阑尾炎', '慢发急', '急性阑尾炎']

    # 定义数据转换 (保持不变)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Matplotlib 中文显示修复 (保持不变) ---
    # ... (Matplotlib 中文配置代码) ...
    try:
        font_path = None
        for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
            if 'simhei' in font.lower():
                font_path = font
                break
        if font_path:
            font_manager.fontManager.addfont(font_path)
            rcParams['font.family'] = 'SimHei'
            rcParams['axes.unicode_minus'] = False
        # ... (省略打印信息) ...
    except Exception as e:
        # ... (省略错误信息) ...
        pass
    # ------------------------------------------------

    # 1. 加载所有数据
    full_dataset = imgDataset(split='full', transform=data_transforms['train'])
    NUM_TABULAR_FEATURES = full_dataset.NUM_TABULAR_FEATURES  # 从 dataloader 获取特征数

    print(f"所有数据 (K-Fold 总池) 图像数: {len(full_dataset.fnames)}")

    all_indices = list(range(len(full_dataset.fnames)))
    all_labels = np.array([item[2] for item in full_dataset.fnames])

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    splits = list(skf.split(all_indices, all_labels))

    fold_image_accuracies: List[float] = []
    all_fold_df = pd.DataFrame()
    best_val_loss_global = 1e9
    best_model_wts_global = None

    # Focal Loss 参数
    alpha = [0.28, 0.58, 0.14]  # 假设您使用这些权重
    criterion = FocalLoss(alpha=alpha, gamma=2)

    # 2. K-Fold 循环
    for fold in range(K):
        train_index, val_index = splits[fold]

        print(f"\n==================== Fold {fold + 1}/{K} 训练 ====================")

        # 准备数据集和 DataLoader
        train_dataset = Subset(full_dataset, train_index)
        val_dataset = Subset(full_dataset, val_index)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        }
        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

        # 实例化新的融合模型
        model = ResNetSAM_TabularFusion(num_classes, num_tabular=NUM_TABULAR_FEATURES)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.00005)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # 训练该折叠
        best_wts, best_loss_fold = train_model_fold(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
                                                    device,
                                                    num_epochs=num_epochs)

        # 测试该折叠的最佳模型
        model.load_state_dict(best_wts)
        fold_acc_tensor, fold_df = test_model_fold(
            model, dataloaders['val'], num_classes, device
        )

        # 记录结果
        fold_acc = fold_acc_tensor.item()
        fold_image_accuracies.append(fold_acc)
        if not fold_df.empty:
            all_fold_df = pd.concat([all_fold_df, fold_df], ignore_index=True)

        print(f"Fold {fold + 1} 验证集图像级准确率: {fold_acc:.4f}")

        # 记录全局最佳权重
        if best_loss_fold < best_val_loss_global:
            best_val_loss_global = best_loss_fold
            best_model_wts_global = best_wts
            torch.save(best_model_wts_global, BEST_MODEL_WTS_FILE)
            print(f"--- 发现新的全局最佳模型权重，已保存到 {BEST_MODEL_WTS_FILE} ---")

        # 检查点保存（可选）
        torch.save({
            'fold': fold,
            'best_val_loss_global': best_val_loss_global,
            'best_model_wts_global': best_model_wts_global,
            'fold_image_accuracies': fold_image_accuracies,
            'all_fold_df': all_fold_df,
        }, CHECKPOINT_FILE)

    # ----------------------------------------------------
    # 3. 汇总 K-Fold 结果 (最终结果)
    # ----------------------------------------------------
    if not all_fold_df.empty:
        print("\n\n=======================================================")
        print("--- ALL DATA 5-Fold 交叉验证汇总结果 (Max Prob Patient Aggregation) ---")
        mean_image_accuracy = np.mean(fold_image_accuracies)
        std_image_accuracy = np.std(fold_image_accuracies)
        print(f"平均图像级准确率 (Mean Image Accuracy): {mean_image_accuracy:.4f} ± {std_image_accuracy:.4f}")

        patient_acc_kfold, cm_kfold, report_kfold, labels_kfold, probas_kfold, agg_df_kfold = aggregate_patient_results(
            all_fold_df, aggregation_method='max_prob'
        )

        # 计算 CI
        ci_lower, ci_upper = bootstrap_ci(agg_df_kfold['TrueLabel'].tolist(), agg_df_kfold['PredLabel'].tolist())
        ci_bounds = (ci_lower, ci_upper) if ci_lower is not None else None

        # 🚨 【核心修改：保存 DeLong 检验数据】
        try:
            # 只保存进行 DeLong 检验必需的列
            agg_df_kfold[['PatientID', 'TrueLabel', 'Prob_0', 'Prob_1', 'Prob_2']].to_csv(DELONG_DATA_FILE, index=False,
                                                                                          encoding='utf-8')
            print(f"\n--- 🥳 患者级概率数据已保存到 {DELONG_DATA_FILE}，可用于 DeLong 检验。 ---")
        except Exception as e:
            print(f"🚨 警告: 保存患者级概率数据失败: {e}")

        # 打印并保存性能报告和 ROC 曲线图
        print_metrics("K-Fold 交叉验证汇总 (Model V2.5 Fusion, All Data)",
                      mean_image_accuracy, patient_acc_kfold, cm_kfold, report_kfold, probas_kfold, labels_kfold,
                      num_classes, class_names_short, output_file=KFOLD_SUMMARY_FILE, ci_bounds=ci_bounds)

    # 训练结束后，删除检查点文件
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"--- 训练检查点文件 {CHECKPOINT_FILE} 已删除 ---")


    print("\n程序执行完毕。")
