import numpy as np
from src.layers import Conv2D, Dense

class L2Regularization:
    def __init__(self, lambda_reg=0.0001):
        """
        Инициализация L2 регуляризации
        
        Parameters:
        -----------
        lambda_reg : float
            Коэффициент регуляризации
        """
        self.lambda_reg = lambda_reg
    
    def loss(self, model):
        """
        Вычисляет значение L2 регуляризации для всех весов модели
        
        Parameters:
        -----------
        model : объект модели
            Модель, содержащая слои с весами
        
        Returns:
        --------
        float : значение L2 регуляризации
        """
        l2_loss = 0
        for layer in model.layers:
            if isinstance(layer, Conv2D):
                l2_loss += np.sum(np.square(layer.weights))
            elif isinstance(layer, Dense):
                l2_loss += np.sum(np.square(layer.weights))
        return self.lambda_reg * l2_loss
    
    def gradients(self, model):
        """
        Вычисляет градиенты L2 регуляризации для всех весов модели
        
        Parameters:
        -----------
        model : объект модели
            Модель, содержащая слои с весами
        
        Returns:
        --------
        dict : словарь градиентов для каждого слоя с весами
        """
        l2_grads = {}
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Conv2D):
                l2_grads[i] = 2 * self.lambda_reg * layer.weights
            elif isinstance(layer, Dense):
                l2_grads[i] = 2 * self.lambda_reg * layer.weights
        return l2_grads