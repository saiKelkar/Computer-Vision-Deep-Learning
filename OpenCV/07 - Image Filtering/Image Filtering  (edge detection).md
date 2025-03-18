[[Initializations and Downloads]]

```
import cv2
import sys
import numpy

PREVIEW = 0 # Preview mode
BLUR = 1 # Blurring filter
FEATURES = 2 # Corner feature detection
CANNY = 3 # Canny edge detection

feature_params = dict(maxCorners = 500, qualityLevel = 0.2, minDistance = 15, blockSize = 9)

s = 0
if len(sys.argv) > 1:
	s = sys.argv[1]

image_filter = PREVIEW
alive = True

win_name = "Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None
source = cv2.VideoCapture(s)

while alive:
	has_frame, frame = source.read()
	if not has_frame:
		break

	# Flip the frame horizontally <-- by default the camera shows a mirror image -- by flipping frame (left to right) to make the view natural, like a selfie 
	frame = cv2.flip(frame, 1)
	if image_filter == PREVIEW:
		result = frame
	elif image_filter == CANNY:
		result = cv2.Canny(frame, 80, 150)
	elif image_filter == BLUR:
		result = cv2.blur(frame, (13, 13))
	elif image_filter == FEATURES:
		result = frame
		# convert to grayscale <-- most CV algorithms work better on grayscale images
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect corners -- finds the most important points in the image (like corners of a shape)
		corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
		# if corners are found <-- draw green circle at each point
		if corners is not None:
			for x, y in numpy.float32(corners).reshape(-1, 2):
				cv2.circle(result, (x, y), 10, (0, 255, 0), 1)

	cv2.imshow(win_name, result)

# ord() <-- converts character into its corresponding ASCII or Unicode value
	key = cv2.waitKey(1)
	if key == ord("Q") or key == ord("q") or key == 27:
		alive == False
	elif key == ord("C") or key == ord("c"):
        image_filter = CANNY
    elif key == ord("B") or key == ord("b"):
        image_filter = BLUR
    elif key == ord("F") or key == ord("f"):
        image_filter = FEATURES
    elif key == ord("P") or key == ord("p"):
        image_filter = PREVIEW

source.release()
cv2.destroyWindow(win_name)
```