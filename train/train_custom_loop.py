import sys
sys.path.append("..")
import numpy as np
from optimizer.SGD import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from layers import two_layer_net

## 设定超参数
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

## 读入数据、模型和优化器
x, t = spiral.load_data()
model = two_layer_net.TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

## 学习用的变量
data_size = len(x)
max_iters = data_size // batch_size
total_loss, loss_count = 0, 0
loss_list = []

## 训练循环
for epoch in range(max_epoch):
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters * batch_size:(iters + 1) * batch_size]
        batch_t = t[iters * batch_size:(iters + 1) * batch_size]

        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iters + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            loss_list.append(avg_loss)
            print(f"Epoch {epoch + 1}, Iteration {iters + 1}, Loss: {avg_loss:.4f}")
            total_loss, loss_count = 0, 0