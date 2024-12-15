from src.layers.max_pool import MaxPool
from src.layers.conv_2d import Conv2D
from src.layers.dense import Dense
from src.models.neural_network import NeuralNetwork
from core.losses import cross_entropy_loss, cross_entropy_gradient
from core.activations import leaky_relu
import numpy as np

class LeNet5(NeuralNetwork):
    def __init__(self):
        # Инициализируем слои в соответствии с архитектурой LeNet-5

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
        
        # Список всех слоев
        self.layers = [self.conv1, self.pool1, self.conv2, self.pool2, self.fc1, self.fc2, self.fc3]


    def forward(self, x):
        print("Input shape:", x.shape)

        # Прямой проход через сеть
        x = self.conv1.forward(x)        # Проход через первый сверточный слой
        x = leaky_relu(x)
        print("After Conv1 shape:", x.shape)

        self.pool1_input = x
        x = self.pool1.forward(x)        # Проход через первый слой пулинга
        print("After Pool1 shape:", x.shape)

        x = self.conv2.forward(x)        # Проход через второй сверточный слой
        x = leaky_relu(x)
        print("After Conv2 shape:", x.shape)

        self.pool2_input = x
        x = self.pool2.forward(x)        # Проход через второй слой пулинга
        print("After Pool2 shape:", x.shape)

        # Преобразуем данные перед подачей в полносвязные слои
        x = x.reshape(x.shape[0], -1)    # Преобразуем (batch_size, height, width, channels) в (batch_size, features)
        print("After reshape shape:", x.shape)

        x = self.fc1.forward(x)           # Проход через первый полносвязный слой
        x = leaky_relu(x)
        x = self.fc2.forward(x)           # Проход через второй полносвязный слой
        x = leaky_relu(x)
        x = self.fc3.forward(x)           # Проход через выходной слой
        return x

    def backward(self, dout):
        # Начинаем обратное распространение ошибки через сеть
        dout = self.fc3.backward(dout)
        dout = self.fc2.backward(dout)
        dout = self.fc1.backward(dout)

        # Проверка на nan или inf в градиентах
        if np.any(np.isnan(dout)) or np.any(np.isinf(dout)):
            print("Warning: NaN or Inf detected in gradients!")

        # Восстанавливаем форму для второго пулинга (batch_size, height, width, channels)
        dout = dout.reshape(dout.shape[0], 4, 4, 16)

        dout = self.pool2.backward(dout, self.pool2_input)
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout, self.pool1_input)
        dout = self.conv1.backward(dout)

        return dout

    def compute_loss(self, y_pred, y_true):
        loss = cross_entropy_loss(y_pred, y_true)
        print(f"Loss: {loss}")
        print(loss)
        return loss

    def compute_loss_gradient(self, y_pred, y_true):
        return cross_entropy_gradient(y_pred, y_true)

    def train(self, X_train, Y_train, epochs, learning_rate, batch_size=32, X_val=None, Y_val=None):
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            for i in range(0, num_samples, batch_size):
                # Получаем текущий батч
                X_batch = X_train[i:i+batch_size]
                Y_batch = Y_train[i:i+batch_size]

                # Прямой проход
                output = self.forward(X_batch)
                
                # Вычисление потерь и обратное распространение
                loss = self.compute_loss(output, Y_batch)
                dout = self.compute_loss_gradient(output, Y_batch)
                self.backward(dout)
                
                # Обновление параметров
                self.update_params(learning_rate)
            
            # Оценка на валидационной выборке, если она предоставлена
            if X_val is not None and Y_val is not None:
                val_output = self.forward(X_val)
                val_loss = self.compute_loss(val_output, Y_val)
                val_accuracy = np.mean(np.argmax(val_output, axis=1) == np.argmax(Y_val, axis=1))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy * 100:.2f}%")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

# Тестирование LeNet-5
if __name__ == "__main__":
    # Инициализация сети
    model = LeNet5()

    # Создадим 20 случайных изображений 28x28 с 1 каналом для обучения и 5 для валидации
    X_train = np.random.rand(20, 28, 28, 1)
    Y_train = np.eye(10)[np.random.choice(10, 20)]  # Случайные one-hot метки для 10 классов

    X_val = np.random.rand(5, 28, 28, 1)
    Y_val = np.eye(10)[np.random.choice(10, 5)]  # Случайные one-hot метки для 10 классов

    # Обучение модели
    epochs = 5
    learning_rate = 0.001
    model.train(X_train, Y_train, epochs, learning_rate, X_val, Y_val)