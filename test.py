def conv2d_im2col(input_images, filters, stride=1, padding=0, bias=None):
    """
    Реализация свёртки для батча изображений с использованием im2col и матричного умножения.
    :param input_images: Входные изображения (N, H, W, C).
    :param filters: Фильтры (K, F_h, F_w, C), где K - количество фильтров.
    :param stride: Шаг свёртки.
    :param padding: Паддинг.
    :param bias: Смещение для каждого фильтра (K).
    :return: Результат свертки (N, out_h, out_w, K).
    """
    N, H, W, C = input_images.shape  # N - количество изображений, H - высота, W - ширина, C - количество каналов
    F_n, F_h, F_w, C_f = filters.shape  # F_n - количество фильтров, C_f - количество каналов в фильтре (должно быть равно C)

    # Размеры выходного изображения для каждого изображения
    out_h = (H + 2 * padding - F_h) // stride + 1
    out_w = (W + 2 * padding - F_w) // stride + 1

    # Разворачиваем входные данные с помощью im2col
    im2col_matrix = im2col(input_images, (F_h, F_w), stride, padding)

    # Разворачиваем фильтры в одномерные векторы
    filters_col = filters.reshape(F_n, -1)

    # Матричное умножение: фильтры * im2col_matrix
    # Транспонируем im2col_matrix на (N, out_h * out_w, F_h * F_w * C)
    im2col_matrix = im2col_matrix.transpose(0, 2, 1)  # Размерность (N, out_h * out_w, F_h * F_w * C)

    # Умножаем фильтры на im2col_matrix
    out_col = filters_col.dot(im2col_matrix.transpose(0, 2, 1))  # Размерность (K, out_h * out_w, N)

    # Транспонируем и восстанавливаем размерность результата
    out = out_col.transpose(2, 1, 0).reshape(N, out_h, out_w, F_n)

    if bias is not None:
        out += bias.reshape(1, 1, 1, F_n)

    return out
def im2col(input_images, kernel_size, stride=1, padding=0):
    """
    Преобразует батч изображений в формат столбцов для ускоренной свертки.
    :param input_images: Входные изображения.
    :param kernel_size: Размер фильтра (F_h, F_w).
    :param stride: Шаг свёртки.
    :param padding: Паддинг.
    :return: Развёрнутая матрица.
    """
    batch_size, height, width, channels = input_images.shape  # batch_size - количество изображений, height - высота, width - ширина, channels - каналы
    # Размеры фильтра
    if isinstance(kernel_size, int):
        kernel_height = kernel_width = kernel_size
    else:
        kernel_height, kernel_width = kernel_size

    # Применяем паддинг, если он есть
    if padding > 0:
        input_images = np.pad(input_images, [(0, 0), (padding, padding), (padding, padding), (0, 0)], mode='constant', constant_values=0)

    # Размеры выходного изображения для каждого изображения
    out_h = (height + 2 * padding - kernel_height) // stride + 1
    out_w = (width + 2 * padding - kernel_width) // stride + 1

    # Массив для хранения развёрнутых данных
    im2col_matrix = np.zeros((batch_size, kernel_height * kernel_width * channels, out_h * out_w))

    for n in range(batch_size):  # Перебираем все изображения в батче
        col_idx = 0
        for i in range(0, out_h * stride, stride):
            for j in range(0, out_w * stride, stride):
                # Извлекаем "окно" (patch)
                patch = input_images[n, i:i+kernel_height, j:j+kernel_width, :]

                # Разворачиваем окно в одномерный вектор и записываем его в колонку
                im2col_matrix[n, :, col_idx] = patch.reshape(-1)

                # Увеличиваем индекс колонки
                col_idx += 1

    return im2col_matrix


def max_pooling(input, pool_size=2, stride=2, padding=0):
    """
    Реализация Max Pooling.

    :param input: Входной тензор
    :param pool_size: Размер окна
    :param stride: Шаг окна
    :return: Срез (max pooling)
    """
    batch_size, in_height, in_width, in_channels = input.shape

    # Применяем паддинг, если он есть
    if padding > 0:
        input = np.pad(input, [(0, 0), (padding, padding), (padding, padding), (0, 0)], mode='constant', constant_values=0)

    out_height = (in_height - pool_size) // stride + 1
    out_width = (in_width - pool_size) // stride + 1

    output = np.zeros((batch_size, out_height, out_width, in_channels))

    for i in range(0, in_height - pool_size + 1, stride):
        for j in range(0, in_width - pool_size + 1, stride):
            region = input[:, i:i+pool_size, j:j+pool_size, :]
            output[:, i//stride, j//stride, :] = np.max(region, axis=(1, 2))

    return output

class CNN:
    def __init__(self, kernel_size, num_filters, num_classes, input_channels=1, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.gamma = np.ones(self.num_filters)
        self.beta = np.zeros(self.num_filters)
        self.m_gamma = np.zeros_like(self.gamma)
        self.v_gamma = np.zeros_like(self.gamma)
        self.m_beta = np.zeros_like(self.beta)
        self.v_beta = np.zeros_like(self.beta)

        F_h, F_w = kernel_size
        self.filters = np.random.randn(num_filters, F_h, F_w, input_channels) * 0.01
        self.bias = np.zeros((num_filters,))

        self.fc_input_size = (28 - F_h + 1) * (28 - F_w + 1) * num_filters
        self.W_fc = np.random.randn(self.fc_input_size, num_classes) * 0.01
        self.b_fc = np.zeros((num_classes,))

        self.m_filters = np.zeros_like(self.filters)
        self.v_filters = np.zeros_like(self.filters)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)

        self.m_W_fc = np.zeros_like(self.W_fc)
        self.v_W_fc = np.zeros_like(self.W_fc)
        self.m_b_fc = np.zeros_like(self.b_fc)
        self.v_b_fc = np.zeros_like(self.b_fc)

        self.t = 0

    def forward(self, X):
        conv_out = conv2d_im2col(X, filters=self.filters, stride=1, padding=0, bias=self.bias)
        conv_out_norm, cache = self.batch_norm_forward(conv_out)

        N, H, W, C = X.shape
        F_h, F_w = self.filters.shape[1], self.filters.shape[2]

        out_h = (H - F_h + 1)
        out_w = (W - F_w + 1)

        self.fc_input = conv_out_norm.reshape(N, out_h * out_w * self.num_filters)

        fc_out = self.fully_connected(self.fc_input, self.W_fc, self.b_fc)
        fc_out_relu = self.relu(fc_out)
        y_pred = self.softmax(fc_out_relu)

        return y_pred, cache

    def batch_norm_forward(self, x, epsilon=1e-5):
        mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
        var = np.var(x, axis=(0, 1, 2), keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + epsilon)
        out = self.gamma.reshape(1, 1, 1, self.num_filters) * x_norm + self.beta.reshape(1, 1, 1, self.num_filters)

        cache = (x, mean, var, x_norm, epsilon)
        return out, cache
    
    def backpropagate(self, X, y_true, dout, cache):
        grad_filters = np.zeros_like(self.filters)  # Gradients for the filters
        grad_bias = np.zeros_like(self.bias)  # Gradients for the biases

        # Get the batch size and shape of X (assuming X is the convolutional input)
        N, H, W, C = X.shape  

        # Case 1: dout is 2D (Fully connected layer)
        if dout.ndim == 2:
            # In this case, dout is of shape (N, C), where C is the number of classes
            # Backprop through fully connected layer
            grad_W_fc = np.dot(self.fc_input.T, dout)  # Gradients w.r.t fully connected layer weights
            grad_b_fc = np.sum(dout, axis=0)  # Gradients w.r.t biases in fully connected layer
            
            # Return gradients related to fully connected layers only
            return grad_filters, grad_W_fc, grad_b_fc, None, None

        # Case 2: dout is 4D (Convolutional layer with batch normalization)
        elif dout.ndim == 4:
            # Reshape dout to match the 4D shape of the convolutional input
            dout_reshaped = dout.reshape(N, H, W, C)
            
            # Backprop through BatchNorm
            dx, grad_gamma, grad_beta = self.batch_norm_backward(dout_reshaped, cache)

            # Backprop through convolutional layer (gradient w.r.t filters)
            grad_filters = np.sum(X * dx, axis=(0, 1), keepdims=True)  # Gradient w.r.t filters

            # Return gradients related to convolutional layers and batch normalization
            return grad_filters, None, None, grad_gamma, grad_beta

        else:
            # If dout has unexpected dimensions, raise an error
            raise ValueError(f"Unexpected number of dimensions for dout: {dout.ndim}")

    def batch_norm_backward(self, dout, cache):
        x, mean, var, x_norm, epsilon = cache

        # Ensure the shape of dout is correct for backpropagation
        N, H, W, C = dout.shape
        grad_gamma = np.sum(dout * x_norm, axis=(0, 1, 2), keepdims=True)
        grad_beta = np.sum(dout, axis=(0, 1, 2), keepdims=True)

        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * (-0.5) * (var + epsilon)**(-1.5), axis=(0, 1, 2), keepdims=True)
        dmean = np.sum(dx_norm * (-1) / np.sqrt(var + epsilon), axis=(0, 1, 2), keepdims=True) + dvar * np.sum(-2 * (x - mean), axis=(0, 1, 2), keepdims=True) / N
        dx = dx_norm / np.sqrt(var + epsilon) + 2 * dvar * (x - mean) / N + dmean / N

        return dx, grad_gamma, grad_beta


    def fully_connected(self, input_data, weights, bias):
        return np.dot(input_data, weights) + bias

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def softmax_loss(self, y_true, y_pred):
        N = y_true.shape[0]
        log_p = np.log(y_pred + 1e-8)
        loss = -np.sum(y_true * log_p) / N
        return loss

    def compute_accuracy(self, y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


    def adam_update(self, grad_W_fc, grad_b_fc, grad_filters, grad_gamma, grad_beta):
        # Update filters
        self.m_filters = self.beta1 * self.m_filters + (1 - self.beta1) * grad_filters
        self.v_filters = self.beta2 * self.v_filters + (1 - self.beta2) * (grad_filters ** 2)
        
        # Update gamma and beta if they are not None
        if grad_gamma is not None:
            self.m_gamma = self.beta1 * self.m_gamma + (1 - self.beta1) * grad_gamma
            self.v_gamma = self.beta2 * self.v_gamma + (1 - self.beta2) * (grad_gamma ** 2)
        
        if grad_beta is not None:
            self.m_beta = self.beta1 * self.m_beta + (1 - self.beta1) * grad_beta
            self.v_beta = self.beta2 * self.v_beta + (1 - self.beta2) * (grad_beta ** 2)
        
        # Update fully connected weights and biases
        self.m_W_fc = self.beta1 * self.m_W_fc + (1 - self.beta1) * grad_W_fc
        self.v_W_fc = self.beta2 * self.v_W_fc + (1 - self.beta2) * (grad_W_fc ** 2)
        
        self.m_b_fc = self.beta1 * self.m_b_fc + (1 - self.beta1) * grad_b_fc
        self.v_b_fc = self.beta2 * self.v_b_fc + (1 - self.beta2) * (grad_b_fc ** 2)
        
        # Perform parameter updates using the corrected m and v values
        self.filters -= self.learning_rate * self.m_filters / (np.sqrt(self.v_filters) + self.epsilon)
        
        if grad_gamma is not None:
            self.gamma -= self.learning_rate * self.m_gamma / (np.sqrt(self.v_gamma) + self.epsilon)
        
        if grad_beta is not None:
            self.beta -= self.learning_rate * self.m_beta / (np.sqrt(self.v_beta) + self.epsilon)
        
        # Update fully connected weights and biases
        self.W_fc -= self.learning_rate * self.m_W_fc / (np.sqrt(self.v_W_fc) + self.epsilon)
        self.b_fc -= self.learning_rate * self.m_b_fc / (np.sqrt(self.v_b_fc) + self.epsilon)


    def train(self, X_train, y_train, epochs=10, batch_size=64):
        """Обучение сети с использованием оптимизатора Adam."""
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # Forward pass
                y_pred, cache = self.forward(X_batch)

                # Compute loss
                loss = self.softmax_loss(y_batch, y_pred)
                epoch_loss += loss

                # Compute accuracy
                acc = self.compute_accuracy(y_batch, y_pred)
                epoch_acc += acc

                # Backpropagation
                dout = y_pred - y_batch  # Derivative of softmax loss
                grad_filters, grad_W_fc, grad_b_fc, grad_gamma, grad_beta = self.backpropagate(X_batch, y_batch, dout, cache)

                # Update parameters using Adam
                self.adam_update(grad_W_fc, grad_b_fc, grad_filters, grad_gamma, grad_beta)

            # Print epoch summary
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (num_samples // batch_size):.4f}, Accuracy: {epoch_acc / (num_samples // batch_size):.4f}')
