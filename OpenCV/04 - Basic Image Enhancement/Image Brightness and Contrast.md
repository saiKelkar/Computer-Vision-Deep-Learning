[[Initializations and Downloads]]
[[Loading and Reading Images]]

```
# Addition or Brightness
# To increase or decrease the brightness of the image, we create a matrix of the same shape as our target image, and multiply it by some int (50, in this case)
# If we add that new matrix to the target image <-- image gets brighten, otherwise, darker
matrix = np.ones(img_rgb.shape, dtype = "uint8") * 50

img_rgb_brighter = cv2.add(img_rgb, matrix)
img_rgb_darker = cv2.subtract(img_rgb, matrix)

# Plot both the images
```

0 -- black
255 -- white

When we add or subtract <-- we shift all pixel values by the same amount -- this brightens or darkens the image uniformly but doesn't change the difference between light and dark areas
for example:
our original pixels are \[50, 100, 150, 200]
add 50, we get \[100, 150, 200, 250] -- brighter but the difference between pixels is still 50
subtract 50, we get \[0, 50, 100, 150] -- darker but same contrast


```
# Multiplication or Contrast
# To improve the contrast of the image <-- multiplication is used
# Contrast is how different the bright and dark areas of the image look -- it controls the difference between light and dark
# High contrast makes bright parts brighter and dark parts darker <-- image looks sharper and more vivid
# Low contrast makes everything more similar in brightness <-- colors look dull and the image looks faded or hazy
# Contrast scales pixel values by a factor

matrix_low_contrast = np.ones(img_rgb.shape) * 0.8
matrix_high_contrast = np.ones(img_rgb.shape) * 1.2

# Converting img_rgb to np.float64 <-- prevents rounding errors and data loss
# We convert it back to uint8 after processing to get valid pixel values
img_rgb_darker = np.uint8(cv2.multiply(np.float64(img_rgb), matrix_low_contrast))
img_rgb_brighter = np.uint8(cv2.multiply(np.float64(img_rgb), matrix_high_contrast))

# Plot both the images
# When we plot this, in the high contrast images <-- values which are already high, become greater than 255 (the overflow issue)
# To overcome the overflow issue
# np.clip() <-- clips the values in an array to stay within a specific range
img_rgb_higher = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), matrix_high_contrast), 0, 255))
```

When we multiply <-- we scale the difference between pixel values
- multiply by a value > 1 -- increases contrast (makes bright brighter and darks darker)
- multiply by a value < 1 -- decreases contrast (makes everything closer to gray)

for example:
original pixels \[50, 100, 150, 200]
multiply by 1.5, we get \[75, 150, 225, 255] -- increased contrast (values spread further apart)
multiply by 0.5, we get \[25, 50, 75, 100] -- reduced contrast (values closer together)
