[[Initializations and Downloads]]

```
# Display 18 x 18 pixel image
Image(filename = "checkerboard_18x18.png")

# Display 84 x 84 pixel image
Image(filename = "checkerboard_84x84.jpg")

# Reading images using openCV
cb_img = cv2.imread("checkerboard_18x18.png", 0)
print(cb_img) # <-- prints the image data -- pixel values -- element of a 2D numpy array
# Each pixel value is 8-bits [0-255]
```

0 as a second argument to cv2.imread() <-- loads the image in grayscale mode
Some common flags to use:

| Flag                 | Value | Description                                                    |
| -------------------- | ----- | -------------------------------------------------------------- |
| cv2.IMREAD_COLOR     | 1     | Loads a color image (default) -- Ignores transparency          |
| cv2.IMREAD_GRAYSCALE | 0     | Loads an image in grayscale (black and white)                  |
| cv2.IMREAD_UNCHANGED | -1    | Loads the image as is, including alpha channels (transparency) |

```
# Print the size of image
print("Image size (H, W) is:", cb_img.shape)
# Image size (H, W) is: (18, 18)

# Print data-type of image
print("Data type of image is:", ch_img.dtype)
# Data type of image is: uint8

# Display image
plt.imshow(cb_img)
plt.show()
```

Image Data type:

uint8
- unsigned (only positive numbers - no negative numbers)
- int <-- integers, whole numbers (no decimals)
- 8-bit <-- each value is stored in 8 bits (1 byte)

uint8 <-- preferred to int32 or float64 as:
- it uses less memory (8 bits per value) compared to 32-bit or 64-bit numbers
- most image formats (png, or jpg) use 8-bit color channels
- image processing libraries like OpenCV are optimized for uint8

```
# Set color map to gray scale for proper rendering
plt.imshow(cb_img, cmap = "gray")
plt.show()

# Read and display Coca-cola logo
Image("coca-cola-logo.png")

# Read in image
coke_img = cv2.imread("coca-cola-logo.png", 1)
plt.imshow(coke_img)
plt.show() # <-- shows reversed color channels
# background of logo which is red, is now displayed as blue <-- this is because openCV usually reverses the color channels while saving the image
# to display as the original image
# Reverse the color channels
coke_img_channels_reversed = coke_img[:, :, ::-1]
plt.imshow(coke_img_channels_reversed)
plt.show()
```

- OpenCV (cv2.imread()) <-- by default reads images in BGR format
- Matplotlib (plt.imshow()) <-- expects images in RGB format

To fix and display the correct image, we need to reverse the color channels from BGR -> RGB

```
coke_img_channels_reversed = coke_img[:, :, ::-1]
```

Understanding \[:, :, ::-1]:

- Image structure
  - coke_img is a 3D NumPy array representing the image, (height, width, channels)
    for example: a 100 x 100 color image will have, coke_img.shape = (100, 100, 3)
- ":" symbol used for slicing
  coke_img\[:, :, ::-1] <-- "::" take everything, and "-1" reverse the order

