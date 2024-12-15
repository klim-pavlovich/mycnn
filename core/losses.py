import numpy as np

def softmax_loss(y_true, y_pred, lambda_reg=0.0001):
    log_p = np.log(y_pred + 1e-8)
    loss = -np.sum(y_true * log_p) / y_true.shape[0] # Делим на размер батча
    # Добавление L2 регуляризации
    loss += l2_regularization(lambda_reg)

    return loss

def l2_regularization(filters, W_fc, lambda_reg):
    # Для весов сверток
    l2_loss = np.sum(np.square(filters))
    # Для весов полносвязанных слоев
    l2_loss += np.sum(np.square(W_fc))

    return lambda_reg * l2_loss


def cross_entropy_loss(y_pred, y_true):
    """
    Вычисляет многоклассовую кросс-энтропию.
    
    y_pred: np.ndarray, shape (n_samples, n_classes) - предсказанные вероятности
    y_true: np.ndarray, shape (n_samples, n_classes) - истинные метки в формате one-hot
    """
    # Избегаем логарифма от нуля
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Вычисляем кросс-энтропию
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

def cross_entropy_gradient(y_pred, y_true):
    """
    Вычисляет градиент многоклассовой кросс-энтропии.
    
    y_pred: np.ndarray, shape (n_samples, n_classes) - предсказанные вероятности
    y_true: np.ndarray, shape (n_samples, n_classes) - истинные метки в формате one-hot
    """
    return (y_pred - y_true) / y_true.shape[0]