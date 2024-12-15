import sys
import os

import numpy as np
from .im2col import Im2Col
from .layer import Layer


class Conv2D(Layer):
    """
    Класс для операции свертки с использованием im2col и матричного умножения.
    """

    def __init__(self, num_filters: int, kernel_size: tuple | int, stride=1, padding=0, input_channels: int = 1):
        """
        Инициализация слоя свертки (Conv2D).

        :param num_filters: Количество фильтров.
        :param stride: Шаг свёртки.
        :param padding: Паддинг.
        :param bias: Смещение для каждого фильтра (K).
        """
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels

        # Размеры фильтра
        if isinstance(kernel_size, int):
            self.kernel_height = self.kernel_width = kernel_size
        else:
            self.kernel_height, self.kernel_width = kernel_size

        # Инициализация для сверточных фильтров с использованием He-инициализации
        self.weights = np.random.randn(num_filters, self.kernel_height, self.kernel_width, self.input_channels) * np.sqrt(2. / (self.kernel_height * self.kernel_width * self.input_channels))

        # Инициализация смещений для всех фильтров
        self.bias = np.zeros((num_filters,))
        
        self.im2col = Im2Col(kernel_size, stride, padding)

    def forward(self, input_images):
        """
        Прямой проход свертки.

        :param input_images: Входные изображения (N, H, W, C).
        :return: Результат свертки (N, out_h, out_w, K).
        """
        self.input_data = input_images  # Сохраняем входные данные для использования в обратном проходе
        batch_size, height, width, channels = input_images.shape
        out_h = (height + 2 * self.padding - self.kernel_height) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_width) // self.stride + 1

        im2col_matrix = self.im2col.transform(input_images)
        filters_col = self.weights.reshape(self.num_filters, self.kernel_height * self.kernel_width * self.input_channels)
        out_col = np.matmul(im2col_matrix, filters_col.T)
        out = out_col.reshape(batch_size, out_h, out_w, self.num_filters)

        if self.bias is not None:
            out += self.bias.reshape(1, 1, 1, self.num_filters)

        return out

    def backward(self, dout):
        """
        Обратное распространение для слоя свертки.
        
        :param dout: Градиенты с следующего слоя, форма (batch_size, out_h, out_w, num_filters)
        :return: Градиенты по входным данным
        """
        batch_size, height, width, channels = self.input_data.shape
        
        # Вычисляем размеры выходного тензора
        out_h = (height + 2 * self.padding - self.kernel_height) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_width) // self.stride + 1
        
        # Преобразуем градиенты в удобную форму
        dout_reshaped = dout.reshape(-1, self.num_filters)  # (batch_size * out_h * out_w, num_filters)
        
        # Получаем im2col представление входных данных
        im2col_matrix = self.im2col.transform(self.input_data)  # (batch_size * out_h * out_w, kernel_h * kernel_w * channels)
        
        # Градиенты по весам
        self.grad_weights = np.dot(im2col_matrix.T, dout_reshaped)  # (kernel_h * kernel_w * channels, num_filters)
        self.grad_weights = self.grad_weights.reshape(self.kernel_height, self.kernel_width, channels, self.num_filters)
        self.grad_weights = self.grad_weights.transpose(3, 0, 1, 2)  # (num_filters, kernel_h, kernel_w, channels)
        
        # Градиенты по смещению
        self.grad_bias = np.sum(dout_reshaped, axis=0)  # (num_filters,)
        
        # Градиенты по входным данным
        weights_reshaped = self.weights.transpose(1, 2, 3, 0).reshape(-1, self.num_filters)  # (kernel_h * kernel_w * channels, num_filters)
        grad_col = np.dot(dout_reshaped, weights_reshaped.T)  # (batch_size * out_h * out_w, kernel_h * kernel_w * channels)
        
        # Восстанавливаем форму градиентов через im2col
        grad_input = self.im2col.backward(grad_col, (batch_size, height, width, channels))
        return grad_input

    def update_params(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

if __name__ == "__main__":
    # Тестовые данные
    input_data = np.random.randn(1, 28, 28, 1)
    conv = Conv2D(num_filters=2, kernel_size=5, stride=1, padding=0)

    # Прямой проход
    out = conv.forward(input_data)
    print("Output shape:", out.shape)

    # Обратный проход
    dout = np.random.randn(*out.shape)
    grad_input = conv.backward(dout)
    print("Gradient input shape:", grad_input.shape)
    print("Weight gradients shape:", conv.grad_weights.shape)
    print("Bias gradients shape:", conv.grad_bias.shape)