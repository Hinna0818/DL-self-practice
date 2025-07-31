import sys
sys.path.append('..')
import numpy as np
from common.layers.Matmul import MatMul
from common.layers.SoftmaxWithLoss import SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        # Initialize weight matrices
        self.W_in = np.random.randn(vocab_size, hidden_size)
        self.W_out = np.random.randn(hidden_size, vocab_size)

        # Create layers
        self.in_layer0 = MatMul(self.W_in)
        self.in_layer1 = MatMul(self.W_in)
        self.out_layer = MatMul(self.W_out)
        self.loss_layer = SoftmaxWithLoss()

        # Store parameters and gradients
        layers = [self.in_layer0, self.in_layer1, self.out_layer, self.loss_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        # store the distribution of input words
        self.word_vecs = self.W_in

    # foward
    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = 0.5 * (h0 + h1)
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
        
    # backward
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None

