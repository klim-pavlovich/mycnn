import numpy as np

# Нормализации-коректировки
def adjust_learning_rate(self, epoch, decay_rate=0.96, decay_epoch=10, warmup_epochs=5):
    """Корректировка learning rate с warm-up"""
    if epoch < warmup_epochs:
        # Постепенно увеличиваем learning rate
        self.learning_rate = self.learning_rate * (1 + 0.1 * epoch)  # Примерно увеличиваем в 1.5 раза

    elif epoch % decay_epoch == 0:
        # После warm-up уменьшаем learning rate
        self.learning_rate *= decay_rate


def adam_update(self, grad_W_fc, grad_b_fc, grad_filters, grad_gamma, grad_beta):
    """Адам-оптимизатор."""
    # Обновление фильтров
    self.m_filters = self.beta1 * self.m_filters + (1 - self.beta1) * grad_filters
    self.v_filters = self.beta2 * self.v_filters + (1 - self.beta2) * (grad_filters ** 2)

    # Обновление gamma и beta, если градиенты существуют
    if grad_gamma is not None:
        self.m_gamma = self.beta1 * self.m_gamma + (1 - self.beta1) * grad_gamma
        self.v_gamma = self.beta2 * self.v_gamma + (1 - self.beta2) * (grad_gamma ** 2)

    if grad_beta is not None:
        self.m_beta = self.beta1 * self.m_beta + (1 - self.beta1) * grad_beta
        self.v_beta = self.beta2 * self.v_beta + (1 - self.beta2) * (grad_beta ** 2)

    # Обновление весов и смещений в полносвязном слое
    self.m_W_fc = self.beta1 * self.m_W_fc + (1 - self.beta1) * grad_W_fc
    self.v_W_fc = self.beta2 * self.v_W_fc + (1 - self.beta2) * (grad_W_fc ** 2)

    self.m_b_fc = self.beta1 * self.m_b_fc + (1 - self.beta1) * grad_b_fc
    self.v_b_fc = self.beta2 * self.v_b_fc + (1 - self.beta2) * (grad_b_fc ** 2)

    # Обновление фильтров свертки
    self.filters -= self.learning_rate * self.m_filters / (np.sqrt(self.v_filters) + self.epsilon)

    if grad_gamma is not None:
        self.gamma -= self.learning_rate * self.m_gamma / (np.sqrt(self.v_gamma) + self.epsilon)

    if grad_beta is not None:
        self.beta -= self.learning_rate * self.m_beta / (np.sqrt(self.v_beta) + self.epsilon)

    # Обновление весов в полносвязном слое
    self.W_fc -= self.learning_rate * self.m_W_fc / (np.sqrt(self.v_W_fc) + self.epsilon)
    self.b_fc -= self.learning_rate * self.m_b_fc / (np.sqrt(self.v_b_fc) + self.epsilon)