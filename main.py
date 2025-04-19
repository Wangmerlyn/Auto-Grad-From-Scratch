import numpy as np

import GradTensor

# generate a random tensor with shape (2, 3)
tensor_x = GradTensor.GradTensor(np.random.rand(2, 3), requires_grad=False)
tensor_w = GradTensor.GradTensor(np.random.rand(3, 2), requires_grad=True)
tensor_b = GradTensor.GradTensor(np.random.rand(2, 2), requires_grad=True)
print(f"tensor_x: {tensor_x}")
print(f"tensor_w: {tensor_w}")
print(f"tensor_b: {tensor_b}")

matmul_result = tensor_x @ tensor_w
print(f"matmul_result: {matmul_result}")
matmul_with_bias = matmul_result + tensor_b
print(f"matmul_with_bias: {matmul_with_bias}")

L = matmul_with_bias.sum()
print(f"L: {L}")
print("Starting backward pass")
print("============================================")
L.backward()
grad_on_w_0_0 = tensor_w.grad.data[0, 0]


delta = 1e-6
tensor_x_ = GradTensor.GradTensor(tensor_x.data, requires_grad=False)
tensor_w_ = GradTensor.GradTensor(tensor_w.data, requires_grad=False)
tensor_w_.data[0, 0] += delta
tensor_b_ = GradTensor.GradTensor(tensor_b.data, requires_grad=False)

matmul_result_ = tensor_x_ @ tensor_w_
matmul_with_bias_ = matmul_result_ + tensor_b_
L_ = matmul_with_bias_.sum()

grad_legacy = (L_.data - L.data) / delta
print(f"grad_legacy: {grad_legacy}, grad_on_w_0_0: {grad_on_w_0_0}")
