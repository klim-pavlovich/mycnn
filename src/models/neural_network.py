from abc import ABC, abstractmethod

class NeuralNetwork(ABC):
    def __init__(self):
        self.layers = []  # Список для хранения слоев

    def add_layer(self, layer):
        """Добавляет слой в нейронную сеть."""
        self.layers.append(layer)

    def forward(self, X):
        """Выполняет прямой проход через все слои."""
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        """Выполняет обратный проход через все слои."""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        """Вычисляет функцию потерь."""
        pass

    @abstractmethod
    def compute_loss_gradient(self, output, y):
        """Вычисляет градиент функции потерь."""
        pass

    def update_params(self, learning_rate):
        """Обновляет параметры всех слоев."""
        for layer in self.layers:
            if hasattr(layer, 'update_params'):
                layer.update_params(learning_rate)

    def train(self, x, y, epochs, learning_rate):
        """Обучает нейронную сеть."""
        for epoch in range(epochs):
            # Прямой проход
            output = self.forward(x)
            # Вычисление потерь и обратное распространение
            loss = self.compute_loss(output, y)
            dout = self.compute_loss_gradient(output, y)
            self.backward(dout)
            # Обновление параметров
            self.update_params(learning_rate)

