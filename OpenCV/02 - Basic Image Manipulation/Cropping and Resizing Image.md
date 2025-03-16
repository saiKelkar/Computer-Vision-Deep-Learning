[[Initializations and Downloads]]
[[Loading and Reading Images]]

```
# Cropping image
cropped_region = img_NZ_rgb[200:400, 300:600]
# 200:400 <-- rows (Y-axis, vertical direction) 
-- starts from 200 and ends at 400, not including 400
-- defines the height of the cropped region

# 300:600 <-- columns (X-axis, horizontal direction)
-- starts from column 300 and end at column 600 (not including 600)
-- defines the width of the cropped region

plt.imshow(cropped_region)
plt.show()
```

```
# Resizing image
dst = resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
dst <-- output image
dsize <-- output image size
fx <-- scale factor along the horizontal axis
fy <-- scale factor along the vertical axis
interpolation <-- stating the type of interpolation

resized_cropped_region_2x = cv2.resize(cropped_region, None, fx = 2, fy = 2)
plt.imshow(resized_cropped_region_2x)
plt.show()

# If we specify the exact size of the output image to resize
desired_width = 100
desired_height = 200
dim = (desired_width, desired_height)

resized_cropped_region = cv2.resize(cropped_region, dsize = dim, interpolation = cv2.INTER_AREA)
plt.imshow(resized_cropped_region)
plt.show()
```

Interpolation:
- Works by using known data to estimate values at the unknown points
- for example: if we wanted to understand the pixel intensity of a picture at a selected location within the grid (say coordinates (x, y), but only (x - 1, y - 1) and  (x + 1, y + 1) are known, we'll estimate the value at (x, y) using linear interpolation)
- the greater the quantity of already known values, the higher would be the accuracy of the estimated pixel values

Different interpolation algorithms include nearest neighbor, bilinear, bicubic, etc. 
In OpenCV:

| name                | description                                                                                                                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| INTER_NEAREST       | nearest neighbor interpolation                                                                                                                                                              |
| INTER_LINEAR        | bilinear interpolation                                                                                                                                                                      |
| INTER_CUBIC         | bicubic interpolation                                                                                                                                                                       |
| INTER_AREA          | resampling using pixel area relation -- may be a preferred method for image decimation as it gives better results (but when the image is zoomed, it is similar to the INTER_NEAREST method) |
| INTER_LANCZOS4      | Lanczos interpolation over 8 x 8 neighbourhood                                                                                                                                              |
| INTER_LINEAR_EXACT  | bit exact bilinear interpolation                                                                                                                                                            |
| INTER_NEAREST_EXACT | bit exact nearest neighbor interpolation -- this produces same result as the nearest neighbor method in PIL, scikit-image or Matlab                                                         |
| INTER_MAX           | mask for interpolation codes                                                                                                                                                                |
| WRAP_FILL_OUTLIERS  | flag, fills all the destination image pixels -- if some of them correspond to outliers in the source image, they are set to zero                                                            |
| WRAP_INVERSE_MAP    | flag, inverse transformation                                                                                                                                                                |
```
# Resize while maintaining aspect ratio
desired_width = 100

aspect_ratio = desired_width / cropped_region.shape[1]
desired_height = int(cropped_region.shape[0] * aspect_ratio)

dim = (desired_width, desired_height)

# Resize image
resized_cropped_region = cv2.resize(cropped_region, dsize = dim, interpolation = cv2.INTER_AREA)
plt.imshow(resized_cropped_region)
plt.show()

# To actually show the cropped resized image
# Swap channel order [:, :, ::-1]
# Save resized image to disk (cv2.imwrite())
# Display the image using IPython (Image(filename = "..."))
```
