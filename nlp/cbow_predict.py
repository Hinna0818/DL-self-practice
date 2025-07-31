import sys
sys.path.append('..')
import numpy as np
from common.layers.Matmul import MatMul

## CBOW (Continuous Bag of Words) model for predicting a target word based on context words.
# demo data
c0 = np.array([1, 0, 0, 0, 0, 0, 0])
c1 = np.array([0, 0, 1, 0, 0, 0, 0])

# initialize weight matrix
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# formulate the MatMul layers
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# foward
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)  # average context vectors
s = out_layer.forward(h)

print("Output scores:", s)