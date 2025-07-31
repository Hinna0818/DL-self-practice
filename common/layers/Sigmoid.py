import numpy as np

class Sigmoid():
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))  # Sigmoid activation function
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (self.out * (1 - self.out))
        return dx

if __name__ == "__main__":
    # Example usage
    sigmoid = Sigmoid()
    x = np.random.randn(4, 3)
    out = sigmoid.forward(x)
    print("Output:", out)

    dout = np.random.randn(4, 3)
    dx = sigmoid.backward(dout)
    print("Gradient:", dx)
    print("Input:", x)
    print("Output after backward pass:", sigmoid.out)