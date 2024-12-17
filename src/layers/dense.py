import numpy as np
from src.layers.layer import Layer  # Предполагаем, что Layer это базовый класс для всех слоев.


class Dense(Layer):
    def __init__(self, input_size, output_size, init_method='he'):
        """
        Инициализирует слой с полностью связанными нейронами (Dense Layer) с случайными весами и смещениями.
        
        :param input_size: Количество входных признаков (например, количество нейронов предыдущего слоя).
        :param output_size: Количество нейронов в текущем слое (размер выходного вектора).
        :param init_method: Метод инициализации весов ('he', 'xavier', 'random').
        """
        self.input_size = input_size
        self.output_size = output_size

        # Инициализация весов в зависимости от метода
        if init_method == 'he':
            # Инициализация весов по методу He (для ReLU)
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        elif init_method == 'xavier':
            # Инициализация весов по методу Xavier
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1. / input_size)
        else:
            # Стандартная случайная инициализация весов
            self.weights = np.random.randn(input_size, output_size) * 0.01

        # Инициализация смещений (базовое смещение для каждого нейрона равно 0)
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data):
        """
        Прямой проход для слоя с полностью связанными нейронами (Dense Layer).
        Выполняет операцию: выход = вход * веса + смещение.

        :param input_data: Входные данные (например, выход предыдущего слоя).
        :return: Выходные данные, полученные после применения весов и смещения.
        """
        self.input_data = input_data  # Сохраняем входные данные для использования в обратном проходе
        output = np.dot(input_data, self.weights) + self.bias  # Линейная операция: y = xW + b
        return output

    def backward(self, dout):
        """
        Обратный проход для слоя с полностью связанными нейронами.
        Вычисляет градиенты функции потерь по отношению к весам и входным данным.

        :param dout: Градиенты функции потерь по отношению к выходу этого слоя (переданы из следующего слоя).
        :return: Градиенты по отношению к входным данным, весам и смещениям для использования в обновлении параметров.
        """

        # Вычисляем градиенты для весов и смещений
        grad_weights = np.dot(self.input_data.T, dout)  # Градиент функции потерь по весам
        grad_bias = np.sum(dout, axis=0, keepdims=True)  # Градиент функции потерь по смещениям

        # Градиент по отношению к входным данным (передается на предыдущий слой)
        grad_input = np.dot(dout, self.weights.T)

        # Сохраняем градиенты для последующего обновления весов
        self.grad_weights = grad_weights
        self.grad_bias = grad_bias
    
       # Возвращаем только градиент по отношению к входным данным
        return grad_input

    def update_params(self, learning_rate):
        """
        Обновляет параметры (веса и смещения) слоя с использованием градиентов и скорости обучения.
        
        :param learning_rate: Скорость обучения, используемая для обновления весов и смещений.
        """
        # Обновляем веса с использованием градиента и скорости обучения
        self.weights -= learning_rate * self.grad_weights
        
        # Обновляем смещения с использованием градиента и скорости обучения
        self.bias -= learning_rate * self.grad_bias
