import numpy as np


class Layer:
    """Основной класс слоев."""

    def forward(self, signal):
        raise NotImplementedError

    def backward(self, error_signal, is_need_out):
        raise NotImplementedError

    def update_params(self, learning_rate):
        raise NotImplementedError