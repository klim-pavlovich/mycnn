import numpy as np

# def softmax_loss(y_true, y_pred, lambda_reg=0.0001):
#     log_p = np.log(y_pred + 1e-8)
#     loss = -np.sum(y_true * log_p) / y_true.shape[0] # Делим на размер батча
#     # Добавление L2 регуляризации
#     loss += l2_regularization(lambda_reg)

#     return loss


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


# def cross_entropy_loss_with_l2(y_pred, y_true, model, lambda_reg=0.0001):
#     """
#     Вычисляет многоклассовую кросс-энтропию с L2 регуляризацией.
    
#     y_pred: np.ndarray, shape (n_samples, n_classes) - предсказанные вероятности
#     y_true: np.ndarray, shape (n_samples, n_classes) - истинные метки в формате one-hot
#     model: объект модели, содержащий параметры (веса) для регуляризации
#     lambda_reg: коэффициент регуляризации
#     """
#     # Избегаем логарифма от нуля
#     epsilon = 1e-15
#     y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
#     # Вычисляем кросс-энтропию
#     loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
#     # Добавляем L2 регуляризацию
#     l2_loss = 0
#     for layer in model.layers:
#         if hasattr(layer, 'weights'):
#             l2_loss += np.sum(np.square(layer.weights))
    
#     loss += lambda_reg * l2_loss
#     return loss

# def cross_entropy_gradient_with_l2(y_pred, y_true, model, lambda_reg=0.0001):
    """
    Вычисляет градиент многоклассовой кросс-энтропии с учетом L2 регуляризации.
    
    y_pred: np.ndarray, shape (n_samples, n_classes) - предсказанные вероятности
    y_true: np.ndarray, shape (n_samples, n_classes) - истинные метки в формате one-hot
    model: объект модели, содержащий параметры (веса) для регуляризации
    lambda_reg: коэффициент регуляризации
    """
    # Градиент кросс-энтропии
    grad = (y_pred - y_true) / y_true.shape[0]
    
    # Добавляем градиент L2 регуляризации
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            grad += 2 * lambda_reg * layer.weights
    
    return grad