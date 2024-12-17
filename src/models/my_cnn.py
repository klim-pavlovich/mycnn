from src.layers.max_pool import MaxPool
from src.layers.conv_2d import Conv2D
from src.layers.dense import Dense
from src.layers.batch_normalization import BatchNormalization
from src.models.neural_network import NeuralNetwork
from core.losses import cross_entropy_loss, cross_entropy_gradient
from core.activations import leaky_relu, softmax, leaky_relu_gradient
from core.regularization import L2Regularization
from core.metrics import compute_f1_score, compute_accuracy
from core.optimizers import Adam
import numpy as np


class MyCNN(NeuralNetwork):
    """
    Добавлен Adam оптимизатор.

    Args:
        NeuralNetwork (class): Базовый класс для нейронных сетей.
    """
    def __init__(self, lambda_reg=0.0001):
        self.training = True
        self.regularization = L2Regularization(lambda_reg)

        # Слой свертки C1: 6 фильтров 5x5, шаг 1, padding 0
        self.conv1 = Conv2D(num_filters=6, kernel_size=5, stride=1, padding=0, input_channels=1)

        # Слой MaxPooling S2: размер 2x2, шаг 2
        self.pool1 = MaxPool(pool_size=2, stride=2)

        # Слой свертки C3: 16 фильтров 5x5, шаг 1, padding 0
        self.conv2 = Conv2D(num_filters=16, kernel_size=5, stride=1, padding=0, input_channels=6)

        # Слой MaxPooling S4: размер 2x2, шаг 2
        self.pool2 = MaxPool(pool_size=2, stride=2)

        # Полносвязный слой C5: 120 нейронов
        # Для входного изображения 28x28 после всех сверток и пулингов получаем 4x4x16 = 256
        self.fc1 = Dense(input_size=256, output_size=120)

        # Полносвязный слой F6: 84 нейрона
        self.fc2 = Dense(input_size=120, output_size=84)

        # Выходной слой: 10 нейронов (для 10 классов)
        self.fc3 = Dense(input_size=84, output_size=10)

        # Batch Normalization перед softmax
        self.bn_final = BatchNormalization(num_channels=10)

        # Список всех слоев
        self.layers = [self.conv1, self.pool1, self.conv2, self.pool2, self.fc1, self.fc2, self.fc3, self.bn_final]

    def forward(self, x):
        """
        Прямой проход через сеть.
        """
        self.input_data = x

        # Первый блок
        x = self.conv1.forward(x)
        self.conv1_output = x
        x = leaky_relu(x)
        self.pool1_input = x
        x = self.pool1.forward(x)

        # Второй блок
        x = self.conv2.forward(x)
        self.conv2_output = x
        x = leaky_relu(x)
        self.pool2_input = x
        x = self.pool2.forward(x)

        # Преобразуем данные перед подачей в полносвязные слои
        x = x.reshape(x.shape[0], -1)

        x = self.fc1.forward(x)
        self.fc1_output = x
        x = leaky_relu(x)
        self.relu1_output = x
        x = self.fc2.forward(x)
        self.fc2_output = x
        x = leaky_relu(x)
        self.relu2_output = x
        x = self.fc3.forward(x)
        self.fc3_output = x

        # Применяем BatchNorm перед softmax
        x = x.reshape(x.shape[0], 1, 1, x.shape[1]) # для batch normalization
        x = self.bn_final.forward(x, self.training)
        x = x.reshape(x.shape[0], -1) # обратно в 2D
        self.softmax_input = x
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
        dout = dout * leaky_relu_gradient(self.relu2_output)
        dout = self.fc2.backward(dout)
        dout = dout * leaky_relu_gradient(self.relu1_output)
        dout = self.fc1.backward(dout)

        # Восстанавливаем форму для сверточных слоев
        dout = dout.reshape(dout.shape[0], 4, 4, 16)

        # Свертки и пулинги
        dout = self.pool2.backward(dout, self.pool2_input)
        dout = dout * leaky_relu_gradient(self.conv2_output)
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout, self.pool1_input)
        dout = dout * leaky_relu_gradient(self.conv1_output)
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
                    # Подготавливаем параметры и градиенты для Adam
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

    def save_weights(self, filename):
        """
        Сохранение весов модели в файл.

        Args:
            filename (str): Путь к файлу для сохранения весов.
        """
        weights = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                weights[f'conv{i}_weights'] = layer.weights
                weights[f'conv{i}_bias'] = layer.bias
            elif isinstance(layer, Dense):
                weights[f'fc{i}_weights'] = layer.weights
                weights[f'fc{i}_bias'] = layer.bias
            elif isinstance(layer, BatchNormalization):
                weights[f'bn{i}_gamma'] = layer.gamma
                weights[f'bn{i}_beta'] = layer.beta
                weights[f'bn{i}_running_mean'] = layer.running_mean
                weights[f'bn{i}_running_var'] = layer.running_var

        np.save(filename, weights)
        print(f"Веса сохранены в {filename}")


    def load_weights(self, filename):
        """
        Загрузка весов модели из файла.

        Args:
            filename (str): Путь к файлу с весами.
        """
        weights = np.load(filename, allow_pickle=True).item()

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                layer.weights = weights[f'conv{i}_weights']
                layer.bias = weights[f'conv{i}_bias']
            elif isinstance(layer, Dense):
                layer.weights = weights[f'fc{i}_weights']
                layer.bias = weights[f'fc{i}_bias']
            elif isinstance(layer, BatchNormalization):
                layer.gamma = weights[f'bn{i}_gamma']
                layer.beta = weights[f'bn{i}_beta']
                layer.running_mean = weights[f'bn{i}_running_mean']
                layer.running_var = weights[f'bn{i}_running_var']

        print(f"Веса загружены из {filename}")

# Тестирование LeNet-5
if __name__ == "__main__":
    # Инициализация сети
    model = MyCNN(lambda_reg=0.0001)

    # Создаем тренировочные данные
    train_size = 100
    X_train = np.random.rand(train_size, 28, 28, 1)  # Увеличим размер тренировочного набора
    Y_train = np.eye(10)[np.random.choice(10, train_size)]

    # Создаем валидационные данные
    val_size = 20
    X_val = np.random.rand(val_size, 28, 28, 1)
    Y_val = np.eye(10)[np.random.choice(10, val_size)]

    # Создаем тестовые данные
    test_size = 30
    X_test = np.random.rand(test_size, 28, 28, 1)
    Y_test = np.eye(10)[np.random.choice(10, test_size)]

    # Параметры обучения
    epochs = 10
    learning_rate = 0.001
    batch_size = 32

    # Обучение модели
    print("Starting training...")
    model.train(
        X_train=X_train,
        Y_train=Y_train,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        X_val=X_val,
        Y_val=Y_val
    )

    # Тестирование модели
    print("\nEvaluating on test set...")
    test_loss, test_accuracy, test_f1 = model.evaluate(X_test, Y_test)
    print(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy*100:.2f}%, F1: {test_f1:.4f}")

    # Проверка предсказаний на одном батче
    print("\nSample predictions:")
    sample_batch = X_test[:5]
    predictions = model.forward(sample_batch)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(Y_test[:5], axis=1)
    print("Predicted classes:", predicted_classes)
    print("True classes:     ", true_classes)