import numpy as np

class BatchNorm:
    def __init__(self, num_filters, epsilon=1e-5):
        # Initialization of the Batch Normalization parameters
        self.num_filters = num_filters
        self.epsilon = epsilon
        self.gamma = np.ones((1, 1, 1, num_filters))  # scale parameter
        self.beta = np.zeros((1, 1, 1, num_filters))  # shift parameter
        self.m_gamma = np.zeros_like(self.gamma)  # momentums for Adam
        self.v_gamma = np.zeros_like(self.gamma)
        self.m_beta = np.zeros_like(self.beta)  # momentums for Adam
        self.v_beta = np.zeros_like(self.beta)

    def forward(self, x):
        """
        Forward pass for Batch Normalization.
        
        :param x: Input data with shape (batch_size, height, width, num_filters)
        :return: Normalized output and cache for backward pass
        """
        mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
        var = np.var(x, axis=(0, 1, 2), keepdims=True)
        
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        out = self.gamma * x_norm + self.beta  # Apply scale and shift
        
        cache = (x, mean, var, x_norm, self.epsilon)  # Cache for backward pass
        return out, cache

    def backward(self, dout, cache):
        """
        Backward pass for Batch Normalization.
        
        :param dout: The gradient of the loss with respect to the output
        :param cache: The cached data from the forward pass
        :return: Gradients for the inputs, gamma, and beta
        """
        x, mean, var, x_norm, epsilon = cache
        N, H, W, C = dout.shape
        
        grad_gamma = np.sum(dout * x_norm, axis=(0, 1, 2), keepdims=True)
        grad_beta = np.sum(dout, axis=(0, 1, 2), keepdims=True)
        
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * (-0.5) * (var + epsilon) ** (-1.5), axis=(0, 1, 2), keepdims=True)
        dmean = np.sum(dx_norm * (-1) / np.sqrt(var + epsilon), axis=(0, 1, 2), keepdims=True) \
                + dvar * np.sum(-2 * (x - mean), axis=(0, 1, 2), keepdims=True) / N
        
        dx = dx_norm / np.sqrt(var + epsilon) + 2 * dvar * (x - mean) / N + dmean / N

        return dx, grad_gamma, grad_beta


# import numpy as np

# def batch_norm_forward(self, x, epsilon=1e-5):
#     """ Выполняет операцию нормализации по батчу на входных данных x.
#     :param x: Входные данные (batch_size, height, width, num_filters).
#     :param epsilon: Маленькое число для избежания деления на ноль при нормализации.
#     :return: (out, cache), где
#     out - нормализованные и масштабированные данные,
#     cache - сохраненные данные для использования при обратном распространении ошибки.
#     """
#     mean = np.mean(x, axis=(0, 1, 2), keepdims=True) # Вычисляем среднее
#     var = np.var(x, axis=(0, 1, 2), keepdims=True) # Вычисляем диспресию

#     x_norm = (x - mean) / np.sqrt(var + epsilon) # Нормализуем данные
#     # Применяем параметры машстабирования и сдвига
#     out = self.gamma.reshape(1, 1, 1, self.num_filters) * x_norm + self.beta.reshape(1, 1, 1, self.num_filters)
#     # Кешируем данные для обратного распространения ошибки для расчета градиентов при обратном проходе
#     cache = (x, mean, var, x_norm, epsilon)
#     return out, cache


# def batch_norm_backward(self, dout, cache):
#     x, mean, var, x_norm, epsilon = cache

#     N, H, W, C = dout.shape
#     grad_gamma = np.sum(dout * x_norm, axis=(0, 1, 2), keepdims=True)
#     grad_beta = np.sum(dout, axis=(0, 1, 2), keepdims=True)

#     dx_norm = dout * self.gamma
#     dvar = np.sum(dx_norm * (x - mean) * (-0.5) * (var + epsilon)**(-1.5), axis=(0, 1, 2), keepdims=True)
#     dmean = np.sum(dx_norm * (-1) / np.sqrt(var + epsilon), axis=(0, 1, 2), keepdims=True) + dvar * np.sum(-2 * (x - mean), axis=(0, 1, 2), keepdims=True) / N
#     dx = dx_norm / np.sqrt(var + epsilon) + 2 * dvar * (x - mean) / N + dmean / N

#     return dx, grad_gamma, grad_beta
