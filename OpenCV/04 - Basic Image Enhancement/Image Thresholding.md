[[Loading and Reading Images]]

Binary Images <-- monochromatic images (black and white images, values ranging from 0 to 1)
These images are commonly used to create Image Masks.
Image Masks <-- allow us to process on specific parts of an image keeping the other parts intact

Image Thresholding <-- used to create binary images from grayscale images
We can use different thresholds to create different binary images from the same original image

```
retval, dst = cv2.threshold(src, thresh, maxval, type[, dst])

# thresh <-- threshold value
# maxval <-- maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types
# type <-- thresholding type

dst = cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])

# adaptiveMethod <-- adaptive thresholding algorithm to use
# thresholdType <-- must be either THRESH_BINARY or THRESH_BINARY_INV
# blockSize <-- size of pixel neighbourhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on
# C <-- constant subtracted from the mean or weighted mean (normally it is positive but maybe zero or negative as well)
```


```
# Perform global thresholding
retval, img_thresh_gbl_1 = cv2.threshold(img_read, 50, 255, cv2.THRESH_BINARY)

# Perform global thresholding
retval, img_thresh_gbl_2 = cv2.threshold(img_read, 130, 255, cv2.THRESH_BINARY)

# Perform adaptive thresholding
img_thresh_adp = cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

# Plot the images
```