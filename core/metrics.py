import numpy as np

def compute_accuracy(y_true, y_pred):
    """
    Вычисляет точность предсказаний.
    
    Args:
        y_true: истинные метки в one-hot формате
        y_pred: предсказанные вероятности
    
    Returns:
        float: точность (accuracy)
    """
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

def compute_f1_score(y_true, y_pred, num_classes=10):
    """
    Вычисляет F1-score для мультиклассовой классификации.
    
    Args:
        y_true: истинные метки в one-hot формате
        y_pred: предсказанные вероятности
        num_classes: количество классов
    
    Returns:
        float: macro F1-score
    """
    # Преобразуем в индексы классов
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    # Вычисляем F1-score для каждого класса
    f1_scores = []
    for class_idx in range(num_classes):
        # True Positives
        tp = np.sum((y_true == class_idx) & (y_pred == class_idx))
        # False Positives
        fp = np.sum((y_true != class_idx) & (y_pred == class_idx))
        # False Negatives
        fn = np.sum((y_true == class_idx) & (y_pred != class_idx))
        
        # Precision и Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1-score для текущего класса
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    # Возвращаем macro F1-score (среднее по всем классам)
    return np.mean(f1_scores)

def confusion_matrix(y_true, y_pred):
    """
    Вычисление матрицы неточностей без использования sklearn
    
    Args:
        y_true: истинные метки в one-hot формате
        y_pred: предсказанные вероятности
        
    Returns:
        confusion_matrix: матрица неточностей размера (n_classes, n_classes)
    """
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    n_classes = y_true.shape[1]
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for true, pred in zip(y_true_classes, y_pred_classes):
        confusion_matrix[true][pred] += 1
        
    return confusion_matrix

def precision_recall_f1(conf_matrix):
    """
    Вычисление precision, recall и F1 на основе матрицы неточностей

    Args:
        conf_matrix: матрица неточностей
    """
    n_classes = conf_matrix.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_scores = np.zeros(n_classes)

    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    return precision, recall, f1_scores


def roc_curve(y_true_binary, y_pred_proba):
    """
    Вычисление ROC-кривой без использования sklearn

    Args:
        y_true_binary: бинарные метки (0 или 1)
        y_pred_proba: предсказанные вероятности

    Returns:
        fpr: False Positive Rate
        tpr: True Positive Rate
    """
    # Сортируем предсказания и соответствующие метки
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_binary = y_true_binary[sorted_indices]

    # Вычисляем количество положительных и отрицательных примеров
    P = np.sum(y_true_binary)
    N = len(y_true_binary) - P

    # Инициализируем массивы для ROC-кривой
    fpr = [0]
    tpr = [0]

    # Вычисляем точки ROC-кривой
    fp = 0
    tp = 0

    for label in y_true_binary:
        if label == 1:
            tp += 1
        else:
            fp += 1

        fpr.append(fp / N)
        tpr.append(tp / P)

    return np.array(fpr), np.array(tpr)

def compute_auc(fpr, tpr):
    """
    Вычисление площади под ROC-кривой (AUC)
    """
    # Используем метод трапеций
    width = np.diff(fpr)
    height = (tpr[1:] + tpr[:-1]) / 2
    return np.sum(width * height)

def compute_roc_auc(y_true, y_pred):
    """Вычисление ROC и AUC для каждого класса"""
    n_classes = y_pred.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i]= roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = compute_auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc