[[Initializations and Downloads]]
[[Loading and Reading Images]]

```
# Drawing a line
img = cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])

imageLine = image.copy()
cv2.line(imageLine, (200, 100), (400, 100), (0, 255, 255), thickness = 5, lineType = cv2.LINE_AA)

plt.imshow(imageLine[:, :, ::-1])
plt.show()
```

```
# Drawing a circle
img = cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])

imageCircle = image.copy()
cv2.circle(imageCircle, (900, 500), 100, (0, 0, 255), thickness = 5, lineType = cv2.LINE_AA)

plt.imshow(imageCircle[:, :, ::-1])
plt.show()
```

```
# Drawing a rectangle
img = cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])

imageRectangle = image.copy()
cv2.rectangle(imageRectangle, (500, 100), (700, 600), (255, 0, 255), thickness = 5, lineType = cv2.LINE_8)

plt.imshow(imageRectangle[:, :, ::-1])
plt.show()
```

```
# Adding text
img = cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])

imageText = image.copy()
text = "Apollo 11 Saturn V Launch, July 16, 1969"
fontScale = 2.3
fontFace = cv2.FONT_HERSHEY_PLAIN
fontColor = (0, 255, 0)
fontThickness = 2

cv2.putText(imageText, text, (200, 700), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA)

plt.imshow(imageText[:, :, ::-1])
plt.show()
```
