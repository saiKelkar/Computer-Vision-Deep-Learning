[[Initializations and Downloads]]
[[Loading and Reading Images]]

```
# dst = cv2.flip(src, flipCode)
dst <-- output 
src <-- source image
flipCode <-- flag to specify how to flip the array
0 -- flipping around x-axis
1 -- flipping around y-axis
-1 -- flipping around both axis

img_NZ_rgb_flipped_horz = cv2.flip(img_NZ_rgb, 1)
img_NZ_rgb_flipped_vert = cv2.flip(img_NZ_rgb, 0)
img_NZ_rgb_flipped_both = cv2.flip(img_NZ_rgb, -1)

# Show the images
```