[[Initializations and Downloads]]
[[Loading and Reading Images]]

```
img_NZ_bgr = cv2.imread("New_Zealand_Lake.jpg", cv2.IMREAD_COLOR)

# Dividing the multi-channel array into several single-channel array
b, g, r = cv2.split(img_NZ_bgr)

# Show the channels

# plt.figure() <-- creates a new figure (a blannk canvas where we can draw plots)
plt.figure(figsize = [20, 5])

# cmap <-- stands for color map, which defines how pixel values are mapped to colors
# when we extract the red channel from an image, it becomes a grayscale intensity map
# .subplot(141) <-- this is where we arrange our plots
# 1 -- 1 row
# 4 -- 4 columns
# 1 -- 1st position (i.e., the first image goes into the left section)
plt.subplot(141); plt.imshow(r, cmap = "gray"); plt.title("Red Channel"); plt.show()

# Convert color channels
img_NZ_rgb = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_NZ_rgb)
plt.show()

# Change to HSV color channel
img_NZ_hsv = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2HSV)
plt.imshow(img_NZ_hsv)
plt.show()

# We can split this into components
h, s, v = cv2.split(img_NZ_hsv)

# and plot them individually
# To modify an individual channel
h_new = h + 10
img_NZ_merged = cv2.merge((h_new, s, v))
img_NZ_rgb = cv2.cvtColor(img_NZ_merged, cv2.COLOR_HSV2RGB)

# Saving the images
cv2.imwrite("New_Zealand_Lake_SAVED.png", img_NZ_bgr)
Image(filename = "New_Zealand_Lake_SAVED.png")
```
