Batching <-- grouping multiple data samples (e.g., images) together into a single tensor so that they can be processed simultaneously by the model

```
batch_tensor = torch.stack([img_tensor_0, img_tensor_1])
print("Batch Tensor shape:", batch_tensor.shape)
# Outputs: ([2, 28, 28, 3])
# 2 <-- batch size
# 28 x 28 <-- image dimensions
# 3 <-- color channels (BGR)
```

```
# Reordering dimensions
batch_input = batch_tensor.permute(0, 3, 1, 2)
print("Batch Tensor Shape: ", batch_input.shape)
# Output: ([2, 3, 28, 28])
# ([2, 28, 28, 3]) <-- NHWC (NumPy format)
# ([2, 3, 28, 28]) <-- NCHW (PyTorch format)

# Essential as nn.Conv2d() expects images in NCHW format
(batch_size, channels, height, width)
```