import numpy as np

class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.x = None
        self.t = None

    def forward(self, x, t):
        self.x = x
        self.t = t
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
        softmax_output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        loss = -np.sum(t * np.log(softmax_output + 1e-7)) / x.shape[0]  # add small value to avoid log(0)
        return loss

    def backward(self):
        batch_size = self.x.shape[0]
        dx = (self.x - self.t) / batch_size
        return dx