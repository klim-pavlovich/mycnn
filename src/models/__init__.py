from .my_cnn import MyCNN
from .le_net_5 import LeNet5
from .neural_network import NeuralNetwork
from .my_cnn_2 import MyCNN2
from .my_cnn_3 import MyCNN3

# Определяем, какие имена будут доступны при импорте через *
__all__ = [
    'MyCNN',
    'LeNet5',
    'NeuralNetwork',
    'MyCNN2',
    'MyCNN3'
]