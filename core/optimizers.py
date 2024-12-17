import numpy as np

# Нормализации-коректировки
def adjust_learning_rate(self, epoch, decay_rate=0.96, decay_epoch=10, warmup_epochs=5):
    """Корректировка learning rate с warm-up"""
    if epoch < warmup_epochs:
        # Постепенно увеличиваем learning rate
        self.learning_rate = self.learning_rate * (1 + 0.1 * epoch)  # Примерно увеличиваем в 1.5 раза

    elif epoch % decay_epoch == 0:
        # После warm-up уменьшаем learning rate
        self.learning_rate *= decay_rate

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # Первый момент
        self.v = {}  # Второй момент
        self.t = 0   # Временной шаг

    def update(self, params, grads, layer_id):
        """
        Обновление параметров с помощью Adam.

        Args:
            params: словарь параметров (weights, bias)
            grads: словарь градиентов
            layer_id: идентификатор слоя
        """
        if layer_id not in self.m:
            self.m[layer_id] = {key: np.zeros_like(value) for key, value in params.items()}
            self.v[layer_id] = {key: np.zeros_like(value) for key, value in params.items()}

        self.t += 1

        for key in params:
            # Обновляем моменты
            self.m[layer_id][key] = self.beta1 * self.m[layer_id][key] + (1 - self.beta1) * grads[key]
            self.v[layer_id][key] = self.beta2 * self.v[layer_id][key] + (1 - self.beta2) * (grads[key]**2)

            # Корректируем смещение
            m_hat = self.m[layer_id][key] / (1 - self.beta1**self.t)
            v_hat = self.v[layer_id][key] / (1 - self.beta2**self.t)

            # Обновляем параметры
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)