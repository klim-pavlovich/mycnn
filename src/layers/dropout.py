import numpy as np

class Dropout:
    """
    Слой Dropout для регуляризации нейронной сети
    
    Args:
        p (float): вероятность отключения нейрона (dropout rate)
    """
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        
    def forward(self, x, training=True):
        """
        Прямой проход
        
        Args:
            x: входные данные
            training: режим обучения
        """
        if training:
            # Создаем маску случайных значений
            self.mask = np.random.binomial(1, 1-self.p, size=x.shape) / (1-self.p)
            # Применяем маску
            return x * self.mask
        else:
            return x
            
    def backward(self, dout):
        """
        Обратный проход
        
        Args:
            dout: градиент от следующего слоя
        """
        return dout * self.mask
