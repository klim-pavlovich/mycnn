import numpy as np

class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, training=True):
        """
        Forward pass for Dropout. 
        During training, randomly drops units from the input.
        
        :param x: Input data of shape (batch_size, ...)
        :param training: Boolean flag indicating whether we are in training mode or inference mode.
        :return: Output after applying Dropout.
        """
        if not training:
            return x  # No dropout during inference (testing)
        
        # Create a mask with the same shape as x, setting a fraction of units to zero
        self.mask = (np.random.rand(*x.shape) < self.dropout_rate) / self.dropout_rate  # Scale the remaining activations
        return x * self.mask  # Apply the mask

    def backward(self, dout):
        """
        Backward pass for Dropout. Simply multiplies the upstream gradients by the dropout mask.
        
        :param dout: Upstream gradients (gradient of the loss with respect to the output)
        :return: Gradient of the loss with respect to the input.
        """
        return dout * self.mask  # Apply the same mask used in the forward pass
