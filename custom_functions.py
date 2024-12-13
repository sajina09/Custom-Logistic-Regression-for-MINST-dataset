import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def custom_accuracy_score(y_true, y_pred):
    """
    Compute accuracy manually.
    
    Parameters:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        
    Returns:
        Accuracy score as a float
    """
    correct_predictions = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    accuracy = correct_predictions / len(y_true)
    return accuracy



def custom_confusion_matrix(y_true, y_pred, num_classes, class_names=None):
    """
    Compute the confusion matrix manually and plot it with a stylish design.
    
    Parameters:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        num_classes: Number of unique classes
        class_names: List of class names corresponding to the classes (optional)
        
    Returns:
        Confusion matrix as a 2D numpy array
    """
    
    # Initialize confusion matrix
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1  # Increment the cell corresponding to (true, predicted)
    
    # Plotting the styled confusion matrix
    plt.figure(figsize=(12,10))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names if class_names else range(num_classes),
                yticklabels=class_names if class_names else range(num_classes),
                linewidths=0.5, linecolor='black')
    
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.show()
    
    return matrix


def custom_classification_report(y_true, y_pred, num_classes, class_names=None):
    """
    Compute and display the classification report in a styled table format.
    
    Parameters:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        num_classes: Number of unique classes
        class_names: List of class names (optional)
        
    Returns:
        A pandas DataFrame containing precision, recall, F1-score, and support.
    """
    import pandas as pd
    import numpy as np

    # Compute confusion matrix
    cm = custom_confusion_matrix(y_true, y_pred, num_classes)

    # Initialize report dictionary
    report = {}
    for cls in range(num_classes):
        true_positive = cm[cls, cls]
        false_positive = sum(cm[:, cls]) - true_positive
        false_negative = sum(cm[cls, :]) - true_positive

        # Handle cases with no positive or negative samples for this class
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(y_true == cls)

        report[cls] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "support": support
        }

    # Convert the report dictionary to a DataFrame
    class_names = class_names if class_names else [f"Class {i}" for i in range(num_classes)]
    df_report = pd.DataFrame.from_dict(report, orient='index')
    df_report.index = class_names

    # Add average metrics
    macro_avg = df_report[["precision", "recall", "f1-score"]].mean()
    weighted_avg = df_report[["precision", "recall", "f1-score"]].multiply(df_report["support"], axis=0).sum() / df_report["support"].sum()

    df_report.loc["macro avg"] = macro_avg
    df_report.loc["weighted avg"] = weighted_avg

    return df_report
