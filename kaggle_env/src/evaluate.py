from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(y_true, y_pred, labels, title="Model Evaluation"):
    print(f"--- {title} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels[:len(np.unique(y_true))]))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels[:len(np.unique(y_true))], 
                yticklabels=labels[:len(np.unique(y_true))])
    plt.title(f"Confusion Matrix: {title}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
