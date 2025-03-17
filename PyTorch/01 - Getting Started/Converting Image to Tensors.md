PyTorch models work with Tensors as:
- Tensors allow GPU acceleration, making computations much faster
- Tensors support automatic differentiation for training neural networks
- PyTorch layers (like `nn.Conv2d`, `nn.Linear`) expects inputs in tensor format

```
# Loading Images
digit_0_array_og = cv2.imread("mnist_0.jpg")
digit_1_array_og = cv2.imread("mnist_1.jpg")

digit_0_array_gray = cv2.imread("mnist_0.jpg", cv2.IMREAD_GRAYSCALE)
digit_1_array_gray = cv2.imread("mnist_1.jpg", cv2.IMREAD_GRAYSCALE)

# Visualize / Plot the images

# Understand the shape
print("Image array shape: ", digit_0_array_og.shape)
# if the image is 28 x 28 with 3 channels, we convert it to 28 x 28 <-- a grayscale image
```

```
# Normalizing and Converting to Tensors
img_tensor_0 = torch.tensor(digit_0_array_og, dtype = torch.float32) / 255.0

# Why divide by 255? 
# The normalized pixel values from 0 - 255 to 0 - 1 <-- makes model training more stable and efficient
# Normalization <-- doesn't change the visual appearance - it just rescales pixel intensity
```