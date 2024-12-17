import numpy as np

class Im2Col:
    """
    Класс для преобразования изображений в формат столбцов для ускоренной свертки.
    """

    def __init__(self, kernel_size, stride=1, padding=0):
        """
        :param kernel_size: Размер фильтра (F_h, F_w) | int.
        :param stride: Шаг свёртки.
        :param padding: Паддинг.
        """
        self.stride = stride
        self.padding = padding
        # Размеры фильтра
        if isinstance(kernel_size, int):
            self.kernel_height = self.kernel_width = kernel_size
        else:
            self.kernel_height, self.kernel_width = kernel_size

    def apply_padding(self, input_data):
        """Применяет паддинг к входным данным."""
        return np.pad(input_data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)

    def transform(self, input_images):
        """
        Преобразует батч изображений в формат столбцов.

        :param input_images: Входные изображения (batch_size, height, width, channels).
        :return: Развёрнутая матрица.
        """
        # Сохраняем размеры входных данных
        batch_size, height, width, channels = input_images.shape

        # Применяем паддинг
        if self.padding > 0:
            input_images = self.apply_padding(input_images)
            height_padded = height + 2 * self.padding
            width_padded = width + 2 * self.padding
        else:
            height_padded = height
            width_padded = width


        # Размеры выходной матрицы после свертки с паддингом
        out_height = (height_padded - self.kernel_height) // self.stride + 1
        out_width = (width_padded - self.kernel_width) // self.stride + 1

        # Добавляем проверку на корректность размеров
        if out_height <= 0 or out_width <= 0:
            raise ValueError(f"Invalid output size: out_height={out_height}, out_width={out_width}")

        shape = (batch_size, out_height, out_width, channels, self.kernel_height, self.kernel_width)
        strides = (
            input_images.strides[0],
            self.stride * input_images.strides[1],
            self.stride * input_images.strides[2],
            input_images.strides[3],
            input_images.strides[1],
            input_images.strides[2]
        )
        # Создаем подматрицы с использованием strides
        patches = np.lib.stride_tricks.as_strided(input_images, shape=shape, strides=strides)
        # Преобразуем в 2D форму: каждый столбец - это фрагмент изображения
        im2col_matrix = patches.reshape(batch_size * out_height * out_width,-1)
        return im2col_matrix
    
    def backward(self, grad_col, input_shape):
        """
        Оптимизированное обратное преобразование градиентов
        """
        batch_size, height, width, channels = input_shape
        
        # Размеры выходного тензора
        out_height = (height + 2 * self.padding - self.kernel_height) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_width) // self.stride + 1
        
        # Создаем расширенный массив для градиентов
        padded_height = height + 2 * self.padding
        padded_width = width + 2 * self.padding
        grad_padded = np.zeros((batch_size, padded_height, padded_width, channels))
        
        # Преобразуем градиенты в нужную форму
        grad_reshaped = grad_col.reshape(batch_size, out_height, out_width, -1)
        
        # Создаем индексы для всех окон свертки
        i0 = np.repeat(np.arange(out_height), out_width)
        i1 = np.tile(np.arange(out_width), out_height)
        
        # Создаем сетку индексов для окна свертки
        h_idx = i0 * self.stride
        w_idx = i1 * self.stride
        
        # Для каждой позиции в окне свертки
        for h in range(self.kernel_height):
            for w in range(self.kernel_width):
                grad_padded[:, h_idx + h, w_idx + w, :] += \
                    grad_reshaped[:, :, :, (h * self.kernel_width + w) * channels:(h * self.kernel_width + w + 1) * channels].reshape(batch_size, -1, channels)
        
        # Убираем паддинг если нужно
        if self.padding > 0:
            return grad_padded[:, self.padding:self.padding + height,
                             self.padding:self.padding + width, :]
        return grad_padded


if __name__ == "__main__":
    # Тест прямого и обратного преобразования
    input_data = np.random.randn(1, 28, 28, 1)
    im2col = Im2Col(kernel_size=5, stride=1, padding=0)
    
    # Прямое преобразование
    col = im2col.transform(input_data)
    print("Col shape:", col.shape)
    
    # Обратное преобразование
    grad_col = np.random.randn(*col.shape)
    grad_input = im2col.backward(grad_col, input_data.shape)
    print("Grad input shape:", grad_input.shape)