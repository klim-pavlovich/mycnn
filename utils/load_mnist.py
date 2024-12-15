# Импортируем нужные библиотеки
import os
import gzip
import numpy as np

def read_idx_file(file_path):
    """ Читает файл в формате IDX (используемый для хранения изображений и меток MNIST).
    Если файла содержит изображения, то он будет интерпретирован как набор изображений, если метки - то как массив меток.

    Аргументы:
        file_path (str): Путь к файлу, который нужно прочитать.

    Исключения:
        ValueError: Если формат файла не соответствует ожидаемому.

    Возвращает:
        numpy.ndarray: Массив с изображениями или метками, в зависимости от типа файла.
    """

    # Проверяем, существует ли директория
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        with gzip.open(file_path, 'rb') as f:
            # Чтение магического числа
            magic = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()  # Используем byteswap для правильной обработки порядка байтов
            print(f"Read magic number: {magic[0]}")

            if magic == 2051:  # Это изображение
                # Чтение метаданных: количество изображений, строки и столбцы
                num_images = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0]
                rows = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0]
                cols = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0]
                print(f"Images: num_images={num_images}, rows={rows}, cols={cols}")

                # Загружаем все пиксели
                images = np.frombuffer(f.read(), dtype=np.uint8)
                print(f"Loaded {images.size} image pixels.")

                # Проверяем количество пикселей
                expected_pixels = num_images * rows * cols
                if images.size != expected_pixels:
                    raise ValueError(f"Mismatch in number of pixels. Expected {expected_pixels}, but got {images.size}.")

                # Преобразуем в массив изображений
                images = images.reshape(num_images, rows, cols)
                return images
            elif magic == 2049:  # Это метки
                num_labels = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0]
                labels = np.frombuffer(f.read(), dtype=np.uint8)
                return labels
            else:
                raise ValueError(f"Unknown IDX file format: {magic}")
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        raise
    except ValueError as e:
        print(f"Произошла ошибка при чтении файла: {e}")
        raise

def load_mnist_data():
    """ Загружает данные MNIST для обучения и тестирования,
    используя функцию 'read_idx_file' для чтения сжатый файлов данных.

    Возвращает:
        tuple:
            - numpy.ndarray: Массив изображений для обучающей выборки, размерность (num_images, rows, cols).
            - numpy.ndarray: Массив меток для обучающей выборки, размерность (num_labels,).
            - numpy.ndarray: Массив изображений для тестовой выборки, размерность (num_images, rows, cols).
            - numpy.ndarray: Массив меток для тестовой выборки, размерность (num_labels,).
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..\data')
    # Чтение изображений и меток
    train_images = read_idx_file(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = read_idx_file(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    test_images = read_idx_file(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = read_idx_file(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    return train_images, train_labels, test_images, test_labels