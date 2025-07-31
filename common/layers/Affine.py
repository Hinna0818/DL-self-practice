import numpy as np

class Affine:
    def __init__(self, w, b):
        self.params = [w, b] ## 初始化权重和偏差
        self.grads = [np.zeros_like(w), np.zeros_like(b)] ## 初始化梯度
        self.x = None ## 用于存储输入数据
    
    def forward(self, x):
        w, b = self.params
        out = np.dot(x, w) + b ## 计算前向传播输出
        self.x = x
        return out
    
    def backward(self, dout):
        w, b = self.params
        dx = np.dot(dout, w.T) ## 计算输入的梯度
        dw = np.dot(self.x.T, dout) ## 计算权重的梯度
        db = np.sum(dout, axis = 0) ## 计算偏差的梯度

        self.grads[0][...] = dw
        self.grads[1][...] = db
        return dx


## example
if __name__ == "__main__":
    # Example usage
    w = np.random.randn(2, 3)
    b = np.random.randn(3)
    affine = Affine(w, b)

    x = np.random.randn(4, 2)
    out = affine.forward(x)

    dout = np.random.randn(4, 3)
    dx = affine.backward(dout)
    print("Output:", out)