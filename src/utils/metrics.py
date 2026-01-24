import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report


def save_confusion_matrix(y_true, y_pred, class_names, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_classification_report(y_true, y_pred, class_names, out_path: str):
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(out_path, 'w') as f:
        f.write(report)
    return report
