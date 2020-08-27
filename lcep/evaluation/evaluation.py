import mlflow

from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, recall_score, accuracy_score
from rich import print


def calculate_log_metrics(y_test, y_pred):
    print('[bold blue]Calculating and logging metrics')
    mlflow.log_metric('Accuracy', accuracy_score(y_test, y_pred))
    mlflow.log_metric('Roc AUC', roc_auc_score(y_test, y_pred))
    mlflow.log_metric('Mathews Correlation', matthews_corrcoef(y_test, y_pred))
    mlflow.log_metric('Specificity', specificity_score(y_test, y_pred))
    mlflow.log_metric('Sensitivity', recall_score(y_test, y_pred))
    mlflow.log_metric('F1 Score', f1_score(y_test, y_pred))


def specificity_score(labels, predictions):
    tp = fp = tn = fn = 0
    for x in range(len(labels)):
        if (predictions[x] == labels[x]) and (labels[x] == 1):
            tp += 1
        elif (predictions[x] != labels[x]) and (labels[x] == 1):
            fn += 1
        elif (predictions[x] == labels[x]) and (labels[x] == 0):
            tn += 1
        elif (predictions[x] != labels[x]) and (labels[x] == 0):
            fp += 1

    score = tn / (tn + fp)

    return score
