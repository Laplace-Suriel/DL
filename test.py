import torch

# 创建一个tensor，并设置requires_grad=True以跟踪计算历史
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 对tensor进行操作
y = x + 2
print(y)

# y是操作的结果，所以它有grad_fn属性
print(y.grad_fn)

# 对y进行更多操作
z = y * y * 3
out = z.mean()

print(z, out)

# 因为out包含一个标量，out.backward()等价于out.backward(torch.tensor(1.))
out.backward()

# 打印梯度 d(out)/dx
print(x.grad)