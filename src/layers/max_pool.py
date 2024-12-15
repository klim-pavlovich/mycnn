import numpy as np
from src.layers.layer import Layer
from src.layers.im2col import Im2Col


class MaxPool(Layer):
    """
        Класс для операции максимального объединения (MaxPooling).

        :param pool_size: Размер окна.
        :param stride: Шаг окна.
        :param padding: Паддинг (если есть).
        :return: Срез (max pooling)
        """
    def __init__(self, pool_size=2, stride=2, padding=0):
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        
        if isinstance(pool_size, int):
            self.pool_height = self.pool_width = pool_size
        else:
            self.pool_height, self.pool_width = pool_size
            
        self.im2col = Im2Col(kernel_size=pool_size, stride=stride, padding=padding)
        
    def forward(self, input_images):
        """
        Прямой проход max pooling с использованием im2col
        """
        self.input_images = input_images
        batch_size, height, width, channels = input_images.shape
        
        # Получаем матрицу im2col
        self.im2col_matrix = self.im2col.transform(input_images)
        
        # Преобразуем в удобный формат для поиска максимумов
        windows = self.im2col_matrix.reshape(-1, self.pool_height * self.pool_width, channels)
        
        # Находим максимумы и их индексы
        self.max_indices = np.argmax(windows, axis=1)
        output = np.max(windows, axis=1)
        
        # Вычисляем размеры выходного тензора
        out_height = (height + 2 * self.padding - self.pool_height) // self.stride + 1
        out_width = (width + 2 * self.padding - self.pool_width) // self.stride + 1
        
        # Преобразуем в нужную форму
        output = output.reshape(batch_size, out_height, out_width, channels)
        
        return output
        
    def backward(self, dout, cache=None):
        """
        Обратное распространение для max pooling
        """
        batch_size, height, width, channels = self.input_images.shape
        
        # Преобразуем градиен��ы в плоский формат
        dout_flat = dout.reshape(-1, channels)
        
        # Создаем массив градиентов для окон
        dx_windows = np.zeros((dout_flat.shape[0], 
                             self.pool_height * self.pool_width,
                             channels))
        
        # Для каждого канала распределяем градиенты
        for i in range(dout_flat.shape[0]):
            for c in range(channels):
                dx_windows[i, self.max_indices[i, c], c] = dout_flat[i, c]
                
        # Преобразуем градиенты обратно через im2col
        dx_col = dx_windows.reshape(self.im2col_matrix.shape)
        
        # Используем backward im2col для восстановления формы градиентов
        dx = self.im2col.backward(dx_col, self.input_images.shape)
        
        return dx
    
    def update_params(self, learning_rate):
        pass

# Тестирование MaxPool
if __name__ == "__main__":
    # Тестовые данные
    input_data = np.random.randn(2, 28, 28, 6)
    
    # Создаем слой
    maxpool = MaxPool(pool_size=2, stride=2)
    
    # Прямой проход
    output = maxpool.forward(input_data)
    print("Forward output shape:", output.shape)
    
    # Обратный проход
    dout = np.random.randn(*output.shape)
    dx = maxpool.backward(dout)
    print("Backward output shape:", dx.shape)