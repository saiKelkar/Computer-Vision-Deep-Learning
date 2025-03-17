Generalized concept of matrices -- can have any number of dimensions

0D Tensor (Scalar) -- single number (e.g., 7)
1D Tensor (Vector) -- List of numbers (e.g., \[1, 2, 3])
2D Tensor (Matrix) -- A table of numbers (e.g., \[\[1, 2], \[3, 4]])
3D Tensor -- collection of matrices (e.g., RGB image with 3 color channels)
4D Tensor -- collection of 3D tensors (e.g., a batch of images)

```
# Create a Tensor with just one column
a = torch.ones(5)
print(a)
# Output: ([1., 1., 1., 1., 1.])

# Create tensors with custom values
b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(b)

# 3D tensor
g = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])

# Access a tensor element
print(g[1][0][0]) # Output: tensor(5.)

# Create a tensor for CPU
# This will occupy CPU RAM
tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device = 'cpu')

# Create a tensor for GPU
# This will occupy GPU RAM
tensor_gpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device = 'cuda')
```