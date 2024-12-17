import numpy as np

class BatchNormalization:
    """
    Слой батч-нормализации для сверточных нейронных сетей.
    Нормализует активации по батчу для каждого канала независимо.
    """
    def __init__(self, num_channels):
        """
        Инициализация слоя батч-нормализации.

        Args:
            num_channels (int): Количество каналов во входных данных
        """
        self.gamma = np.ones((1, 1, 1, num_channels))
        self.beta = np.zeros((1, 1, 1, num_channels))
        self.dgamma = np.zeros((1, 1, 1, num_channels))
        self.dbeta = np.zeros((1, 1, 1, num_channels))
        # Добавляем running statistics
        self.running_mean = np.zeros((1, 1, 1, num_channels))
        self.running_var = np.ones((1, 1, 1, num_channels))
        self.momentum = 0.9
        
        # Параметр для экспоненциального сглаживания
        self.momentum = 0.9
        self.epsilon = 1e-5
        self.cache = None

    def forward(self, x, training=True):
        """
        Прямой проход батч-нормализации.

        Args:
            x (np.array): Входные данные shape (batch_size, height, width, channels)
            training (bool): Режим работы (обучение/тестирование)

        Returns:
            np.array: Нормализованные данные того же размера
        """
        
        if training:
            # Вычисляем статистики по текущему батчу
            mu = np.mean(x, axis=(0,1,2), keepdims=True)
            var = np.var(x, axis=(0,1,2), keepdims=True)
            
            # Обновляем running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # В режиме инференса используем running statistics
            mu = self.running_mean
            var = self.running_var

        # Вычисляем среднее по батчу для каждого канала
        mu = np.mean(x, axis=(0,1,2), keepdims=True)  # shape: (1,1,1,C)
        # Вычисляем дисперсию
        var = np.var(x, axis=(0,1,2), keepdims=True)  # shape: (1,1,1,C)
        # Нормализуем
        x_normalized = (x - mu) / np.sqrt(var + self.epsilon)  # shape: (N,H,W,C)
        # Масштабируем и сдвигаем
        out = self.gamma * x_normalized + self.beta  # shape: (N,H,W,C)
        # Сохраняем значения для обратного прохода
        self.cache = {
            'x': x,
            'mu': mu,
            'var': var,
            'x_normalized': x_normalized,
            'gamma': self.gamma
        }

        return out

    def backward(self, dout):
        """
        Обратный проход батч-нормализации.

        Args:
            dout (np.array): Градиент по выходу слоя shape (batch_size, height, width, channels)

        Returns:
            tuple: (dx, dgamma, dbeta) - градиенты по входу и параметрам
        """
        # Достаем сохраненные значения
        x = self.cache['x']
        mu = self.cache['mu']
        var = self.cache['var']
        x_normalized = self.cache['x_normalized']
        gamma = self.cache['gamma']

        N, H, W, C = dout.shape
        M = N * H * W  # количество элементов для усреднения

        # Градиенты для gamma и beta
        dgamma = np.sum(dout * x_normalized, axis=(0,1,2), keepdims=True)
        dbeta = np.sum(dout, axis=(0,1,2), keepdims=True)

        # Градиент по нормализованному входу
        dx_normalized = dout * gamma

        # Градиент по дисперсии
        dvar = np.sum(dx_normalized * (x - mu) * -0.5 * (var + self.epsilon)**(-1.5), axis=(0,1,2), keepdims=True)

        # Градиент по среднему
        dmu = np.sum(dx_normalized * -1/np.sqrt(var + self.epsilon), axis=(0,1,2), keepdims=True)
        dmu += dvar * np.sum(-2 * (x - mu), axis=(0,1,2), keepdims=True) / M

        # Градиент по входу
        dx = dx_normalized / np.sqrt(var + self.epsilon)
        dx += 2 * dvar * (x - mu) / M
        dx += dmu / M

        return dx, dgamma, dbeta
