import numpy as np

def one_hot_encode(y, num_classes=10):
    """ Преобразует метки классов в формат one-hot encoding.

    Аргументы:
        y (numpy.ndarray): Вектор меток классов (целые числа от 0 до num_classes-1).
        num_classes (int): Количество классов. По умолчанию 10.

    Возвращает:
        numpy.ndarray: Массив one-hot векторов размерности (len(y), num_classes), где для каждой метки будет установлен соответствующий элемент в единицу.
    """
    # Если y уже one-hot, не нужно повторно кодировать
    if len(y.shape) == 2 and y.shape[1] == num_classes:
        return y  # Уже в формате one-hot

    # Иначе, выполняем стандартное кодирование
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot
