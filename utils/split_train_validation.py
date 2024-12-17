from sklearn.model_selection import train_test_split

def split_train_validation(X_train, Y_train, test_size=0.3, random_state=42):
    """
    Разделяет обучающую выборку на обучающую и валидационную в пропорции 70% / 30%.

    Аргументы:
        X_train (numpy.ndarray): Изображения для обучающей выборки.
        Y_train (numpy.ndarray): Метки для обучающей выборки.
        test_size (float): Пропорция для валидационной выборки (по умолчанию 0.3 — 30%).
        random_state (int): Число для генератора случайных чисел (для воспроизводимости).

    Возвращает:
        X_train_split, X_val_split, Y_train_split, Y_val_split
    """
    # Разделяем данные на обучающую и валидационную выборки
    X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(
        X_train, Y_train, test_size=test_size, random_state=random_state
    )

    return X_train_split, X_val_split, Y_train_split, Y_val_split
