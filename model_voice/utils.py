from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import Dataset


def print_classification_report(data, target_names=None, labels=None):
    """
    In classification_report cho từng fold trong K-Fold
    """
    for i, (labels_fold, preds_fold) in enumerate(data):
        print(f"\nFold {i}:")
        print(classification_report(
            labels_fold,
            preds_fold,
            labels=labels,                # ép dùng toàn bộ nhãn
            target_names=target_names,
            zero_division=0               # tránh warning chia 0
        ))


def average_classification_report(data, target_names=None, labels=None):
    """
    In classification_report trung bình gộp tất cả các fold
    """
    all_labels = []
    all_preds = []
    for labels_fold, preds_fold in data:
        all_labels.extend(labels_fold)
        all_preds.extend(preds_fold)

    print("\n=== Average classification report (all folds) ===")
    print(classification_report(
        all_labels,
        all_preds,
        labels=labels,
        target_names=target_names,
        zero_division=0
    ))


def plot_metrics(data_kfold):
    """
    Vẽ Accuracy / Recall / F1 / Precision theo epoch cho từng fold
    """
    num_folds = len(data_kfold)
    plt.figure(figsize=(20, 5))

    for idx, data in enumerate(data_kfold):
        acc, f1, recall, precision = data
        plt.subplot(1, num_folds, idx + 1)
        plt.plot(acc, label='Accuracy')
        plt.plot(recall, label='Recall')
        plt.plot(f1, label='F1')
        plt.plot(precision, label='Precision')
        plt.title(f"K-Fold {idx + 1}")
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(labels_preds_kfold, labels=None, target_names=None):
    """
    Vẽ confusion matrix gộp tất cả các fold.
    - labels_preds_kfold: danh sách chứa [labels, preds] cho từng fold.
    - labels: danh sách nhãn số (ví dụ [0,1,2,...]) — nếu None, sẽ tự suy ra.
    - target_names: danh sách tên lớp để hiển thị trên trục.
    """

    # Gộp toàn bộ label và predict từ các fold
    all_labels = []
    all_preds = []
    for data in labels_preds_kfold:
        if len(data) >= 2:   # đảm bảo đúng cấu trúc
            all_labels.extend(data[0])
            all_preds.extend(data[1])

    # Nếu labels chưa có -> tự sinh ra theo unique giá trị
    if labels is None:
        labels = np.unique(all_labels)

    # Tính confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=labels)

    # Nếu không có tên lớp, tự tạo tên số
    if target_names is None:
        target_names = [str(l) for l in labels]

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (All folds)')
    plt.tight_layout()
    plt.show()



def compute_ua_wa(labels_preds_kfold):
    """
    Tính UA (Unweighted Accuracy) và WA (Weighted Accuracy)
    """
    true_labels = np.array([])
    predicted_labels = np.array([])

    # Gom tất cả true/pred từ các fold
    for data in labels_preds_kfold:
        true_labels = np.append(true_labels, data[0])
        predicted_labels = np.append(predicted_labels, data[1])

    # UA = accuracy tổng thể
    ua = accuracy_score(true_labels, predicted_labels)

    # WA = accuracy trung bình từng lớp
    wa = []
    for label in np.unique(true_labels):
        i_true = true_labels[true_labels == label]
        i_pred = predicted_labels[true_labels == label]
        wa.append(np.sum(i_true == i_pred) / len(i_true))
    wa = np.mean(wa)

    return ua, wa


# =============================
# Dataset class dùng cho train/test
# =============================
class EmoDataset(Dataset):
    def __init__(self, mfccs, labels):
        self.mfccs = mfccs
        self.labels = labels

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, idx):
        mfcc = torch.tensor(self.mfccs[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mfcc, label
