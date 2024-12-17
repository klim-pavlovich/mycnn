from src.layers.max_pool import MaxPool
from src.layers.conv_2d import Conv2D
from src.layers.dense import Dense
from src.layers.batch_normalization import BatchNormalization
from src.layers.dropout import Dropout
from src.models.neural_network import NeuralNetwork
from core.losses import cross_entropy_loss, cross_entropy_gradient
from core.activations import leaky_relu, softmax, leaky_relu_gradient
from core.regularization import L2Regularization
from core.metrics import compute_f1_score, compute_accuracy
from core.optimizers import Adam
import numpy as np


class MyCNN3(NeuralNetwork):
    """
    Класс для создания и обучения модели, подобной LeNet-5. Добавлена регуляризация L2.
    Вычисление потерь и градиентов с регуляризацией.
    Обновление параметров с Адамом.
    Добавлен Дропаут.

    Args:
        NeuralNetwork (class): Базовый класс для нейронных сетей.
    """
    def __init__(self, lambda_reg=0.0001, dropout_rate=0.5):
        self.training = True
        self.regularization = L2Regularization(lambda_reg)

        # Первый блок
        self.conv1 = Conv2D(num_filters=32, kernel_size=3, stride=1, padding=0, input_channels=1)
        self.bn1 = BatchNormalization(num_channels=32)
        self.pool1 = MaxPool(pool_size=2, stride=2)

        # Второй блок
        self.conv2 = Conv2D(num_filters=64, kernel_size=5, stride=1, padding=0, input_channels=32)
        self.bn2 = BatchNormalization(num_channels=64)
        self.pool2 = MaxPool(pool_size=2, stride=2)

        # Полносвязные слои
        self.fc1 = Dense(input_size=1024, output_size=512)
        self.dropout1 = Dropout(p=dropout_rate)
        self.fc2 = Dense(input_size=512, output_size=128)
        self.dropout2 = Dropout(p=dropout_rate)
        self.fc3 = Dense(input_size=128, output_size=10)

        # Batch Normalization перед softmax
        self.bn_final = BatchNormalization(num_channels=10)

        # Список всех слоев
        self.layers = [self.conv1, self.pool1, self.conv2, self.pool2, self.fc1, self.fc2, self.fc3, self.bn1, self.bn2, self.bn_final, self.dropout1, self.dropout2]

    def forward(self, x):
        """
        Прямой проход через сеть.
        """
        self.input_data = x

        # Первый блок
        x = self.conv1.forward(x)
        x = self.bn1.forward(x, self.training)
        self.bn1_output = x
        x = leaky_relu(x)
        self.pool1_input = x
        x = self.pool1.forward(x)

        # Второй блок
        x = self.conv2.forward(x)
        x = self.bn2.forward(x, self.training)
        self.bn2_output = x
        self.pool2_input = x
        x = self.pool2.forward(x)

        # Полносвязные слои
        x = x.reshape(x.shape[0], -1)
        x = self.fc1.forward(x)
        x = self.dropout1.forward(x, self.training)
        self.fc1_output = x
        x = leaky_relu(x)
        x = self.fc2.forward(x)
        x = self.dropout2.forward(x, self.training)
        self.fc2_output = x
        x = leaky_relu(x)

        x = self.fc3.forward(x)

        # Преобразуем в 4D формат для batch normalization
        x = x.reshape(x.shape[0], 1, 1, x.shape[1])
        x = self.bn_final.forward(x, self.training)
        # Возвращаем к 2D формату
        x = x.reshape(x.shape[0], -1)
        x = softmax(x)
        return x

    def backward(self, dout):
        """
        Обратное распространение ошибки через сеть.
        """

        # Преобразуем в 4D для batch normalization
        dout = dout.reshape(dout.shape[0], 1, 1, dout.shape[1])
        dx, self.bn_final.dgamma, self.bn_final.dbeta = self.bn_final.backward(dout)
        # Возвращаем к 2D
        dout = dx.reshape(dx.shape[0], -1)

        # Полносвязные слои
        dout = self.fc3.backward(dout)
        dout = dout * leaky_relu_gradient(self.fc2_output)
        dout = self.dropout2.backward(dout)
        dout = self.fc2.backward(dout)
        dout = dout * leaky_relu_gradient(self.fc1_output)
        dout = self.dropout1.backward(dout)
        dout = self.fc1.backward(dout)

        # Восстанавливаем форму для сверточных слоев (batch_size, height, width, channels)
        dout = dout.reshape(dout.shape[0], 4, 4, 64)

        # Второй блок
        dout = self.pool2.backward(dout, self.pool2_input)
        dout = dout * leaky_relu_gradient(self.bn2_output)  # Градиент через Leaky ReLU
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)

        # Первый блок
        dout = self.pool1.backward(dout, self.pool1_input)
        dout = dout * leaky_relu_gradient(self.bn1_output)  # Градиент через Leaky ReLU
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)

        return dout

    def compute_loss(self, y_pred, y_true):
        # Основная функция потерь
        ce_loss = cross_entropy_loss(y_pred, y_true)
        # Добавляем L2 регуляризацию
        l2_loss = self.regularization.loss(self)
        return ce_loss + l2_loss

    def compute_loss_gradient(self, y_pred, y_true):
        return cross_entropy_gradient(y_pred, y_true)

    def update_params(self):
        l2_grads = self.regularization.gradients(self)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                if i in l2_grads:
                    grad_with_reg = layer.grad_weights + l2_grads[i]
                    params = {'weights': layer.weights, 'bias': layer.bias}
                    grads = {'weights': grad_with_reg, 'bias': layer.grad_bias}
                    self.optimizer.update(params, grads, f'conv_{i}')
                    layer.weights = params['weights']
                    layer.bias = params['bias']
                    
            elif isinstance(layer, Dense):
                if i in l2_grads:
                    grad_with_reg = layer.grad_weights + l2_grads[i]
                    params = {'weights': layer.weights, 'bias': layer.bias}
                    grads = {'weights': grad_with_reg, 'bias': layer.grad_bias}
                    self.optimizer.update(params, grads, f'dense_{i}')
                    layer.weights = params['weights']
                    layer.bias = params['bias']
                    
            elif isinstance(layer, BatchNormalization):
                params = {'gamma': layer.gamma, 'beta': layer.beta}
                grads = {'gamma': layer.dgamma, 'beta': layer.dbeta}
                self.optimizer.update(params, grads, f'bn_{i}')
                layer.gamma = params['gamma']
                layer.beta = params['beta']

    def evaluate(self, X, Y):
        """
        Оценка модели на данных.
        """
        self.training = False  # выключаем режим обучения
        predictions = self.forward(X)
        loss = self.compute_loss(predictions, Y)
        accuracy = compute_accuracy(Y, predictions)
        f1 = compute_f1_score(Y, predictions)
        return loss, accuracy, f1

    def train(self, X_train, Y_train, epochs, learning_rate, batch_size=32, X_val=None, Y_val=None):
        self.training = True
        num_samples = X_train.shape[0]
        
        # Инициализируем Adam с заданным learning rate
        self.optimizer = Adam(learning_rate=learning_rate)
        
        for epoch in range(epochs):
            # Массивы для накопления предсказаний и меток за эпоху
            epoch_predictions = []
            epoch_true_labels = []
            epoch_losses = []

            for i in range(0, num_samples, batch_size):
                # Получаем текущий батч
                X_batch = X_train[i:i+batch_size]
                Y_batch = Y_train[i:i+batch_size]

                # Прямой проход
                output = self.forward(X_batch)
                
                # Накапливаем предсказания и метки
                epoch_predictions.append(output)
                epoch_true_labels.append(Y_batch)

                # Вычисление потерь и обратное распространение
                loss = self.compute_loss(output, Y_batch)
                epoch_losses.append(loss)

                dout = self.compute_loss_gradient(output, Y_batch)
                self.backward(dout)

                # Обновление параметров без передачи learning_rate
                self.update_params()

            # Объединяем все предсказания и метки за эпоху
            epoch_predictions = np.vstack(epoch_predictions)
            epoch_true_labels = np.vstack(epoch_true_labels)

            # Вычисляем метрики за всю эпоху
            epoch_loss = np.mean(epoch_losses)
            epoch_accuracy = compute_accuracy(epoch_true_labels, epoch_predictions)
            epoch_f1 = compute_f1_score(epoch_true_labels, epoch_predictions)

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy*100:.2f}%, F1: {epoch_f1:.4f}")

            # Оценка на валидационной выборке используя метод evaluate
            if X_val is not None and Y_val is not None:
                val_loss, val_accuracy, val_f1 = self.evaluate(X_val, Y_val)
                print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy*100:.2f}%, F1: {val_f1:.4f}")

            # Возвращаем режим обучения после валидации
            self.training = True

    def evaluate(self, X, Y):
        """
        Оценка модели на данных.
        """
        self.training = False  # выключаем режим обучения
        predictions = self.forward(X)
        loss = self.compute_loss(predictions, Y)
        accuracy = compute_accuracy(Y, predictions)
        f1 = compute_f1_score(Y, predictions)
        return loss, accuracy, f1

    def save_weights(self, filename):
        """
        Сохранение весов модели в файл
        
        Args:
            filename (str): путь к файлу для сохранения
        """
        weights = {
            # Сверточные слои
            'conv1_weights': self.conv1.weights,
            'conv1_bias': self.conv1.bias,
            'conv2_weights': self.conv2.weights,
            'conv2_bias': self.conv2.bias,
            
            # Полносвязные слои
            'fc1_weights': self.fc1.weights,
            'fc1_bias': self.fc1.bias,
            'fc2_weights': self.fc2.weights,
            'fc2_bias': self.fc2.bias,
            'fc3_weights': self.fc3.weights,
            'fc3_bias': self.fc3.bias,
            
            # BatchNorm слои
            'bn1_gamma': self.bn1.gamma,
            'bn1_beta': self.bn1.beta,
            'bn1_mean': self.bn1.running_mean,
            'bn1_var': self.bn1.running_var,
            
            'bn2_gamma': self.bn2.gamma,
            'bn2_beta': self.bn2.beta,
            'bn2_mean': self.bn2.running_mean,
            'bn2_var': self.bn2.running_var,
            
            'bn_final_gamma': self.bn_final.gamma,
            'bn_final_beta': self.bn_final.beta,
            'bn_final_mean': self.bn_final.running_mean,
            'bn_final_var': self.bn_final.running_var,
        }
        np.save(filename, weights)
        print(f"Веса сохранены в {filename}")

    def load_weights(self, filename):
        """
        Загрузка весов модели из файла
        
        Args:
            filename (str): путь к файлу с весами
        """
        weights = np.load(filename, allow_pickle=True).item()
        
        # Загружаем веса в соответствующие слои
        self.conv1.weights = weights['conv1_weights']
        self.conv1.bias = weights['conv1_bias']
        self.conv2.weights = weights['conv2_weights']
        self.conv2.bias = weights['conv2_bias']
        
        self.fc1.weights = weights['fc1_weights']
        self.fc1.bias = weights['fc1_bias']
        self.fc2.weights = weights['fc2_weights']
        self.fc2.bias = weights['fc2_bias']
        self.fc3.weights = weights['fc3_weights']
        self.fc3.bias = weights['fc3_bias']
        
        # BatchNorm слои
        self.bn1.gamma = weights['bn1_gamma']
        self.bn1.beta = weights['bn1_beta']
        self.bn1.running_mean = weights['bn1_mean']
        self.bn1.running_var = weights['bn1_var']
        
        self.bn2.gamma = weights['bn2_gamma']
        self.bn2.beta = weights['bn2_beta']
        self.bn2.running_mean = weights['bn2_mean']
        self.bn2.running_var = weights['bn2_var']
        
        self.bn_final.gamma = weights['bn_final_gamma']
        self.bn_final.beta = weights['bn_final_beta']
        self.bn_final.running_mean = weights['bn_final_mean']
        self.bn_final.running_var = weights['bn_final_var']
            
        print(f"Веса загружены из {filename}")
