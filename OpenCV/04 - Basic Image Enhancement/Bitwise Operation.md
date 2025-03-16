```
dst = cv2.bitwise_and(src1, src2[, dst[, mask]])

src1 <-- first input array or a scalar
src2 <-- second input array or a scalar

mask <-- optional operation, 8-bit channel array, that specifies elements of the output array to be changed
```

Bitwise "and":

| Image 1 Pixel | Image 2 Pixel | Result Pixel |
| ------------- | ------------- | ------------ |
| 255 (white)   | 255 (white)   | 255 (white)  |
| 255 (white)   | 0 (black)     | 0 (black)    |
| 0 (black)     | 255 (white)   | 0 (black)    |
| 0 (black)     | 0 (black)     | 0 (black)    |

Bitwise "or":

| Image 1 Pixel | Image 2 Pixel | Result Pixel |
| ------------- | ------------- | ------------ |
| 255 (white)   | 255 (white)   | 255 (white)  |
| 255 (white)   | 0 (black)     | 255 (white)  |
| 0 (black)     | 255 (white)   | 255 (white)  |
| 0 (black)     | 0 (black)     | 0 (black)    |

Bitwise "xor":

| Image 1 Pixel | Image 2 Pixel | Result Pixel |
| ------------- | ------------- | ------------ |
| 255 (white)   | 255 (white)   | 0 (black)    |
| 255 (white)   | 0 (black)     | 255 (white)  |
| 0 (black)     | 255 (white)   | 255 (white)  |
| 0 (black)     | 0 (black)     | 0 (black)    |

Application: Logo Manipulation

```
# Read the images
# Resize the images to make them same size

# Create mask for original image
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# Apply global thresholding to create binary mask of the logo
retval, img_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
# Plot the images

# Invert the mask
img_mask_inv = cv2.bitwise_not(img_mask)
# Plot the image

# Apply background on the mask
img_background = cv2.bitwise_and(img_background_rgb, img_background_rgb, mask = img_mask)

# Isolate forground from image
img_foreground = cv2.bitwise_and(img_rgb, img_rgb, mask = img_mask_inv)

# Result: Merge foreground and background
result = cv2.add(img_background, img_foreground)
# Plot the final image
```