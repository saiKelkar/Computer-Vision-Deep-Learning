[[Initializations and Downloads]]

```
import cv2
# sys allows to access system-specific parameters (like command-line arguments)
import sys

# Sets the default video source to 0 <-- usually refers to primary webcam
s = 0
# sys.srgv <-- list of command-line arguments passed to the script
# sys.argv[0] <-- the script's name
# sys.argv[1] <-- the first argument (if provided)
# if we run the video file path like python3 script.py video.mp4 <-- it will capture from that file (otherwise, default is 0 -- our webcam)
if len(sys.argv) > 1:
	s = sys.argv[1]

# cv2.VideoCapture <-- open the video stream
source = cv2.VideoCapture(s)

# win_name is window name
win_name = 'Camera Preview'

# cv2.WINDOW_NORMAL <-- allows the window to be resized
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# cv2.waitKey <-- waits 1 millisecond for a key press
# 27 <-- ASCII value for Escape key -- used to exit the loop
while cv2.waitKey(1) != 27: 
	# has_frame <-- bool indicating if the frame was successfully read
	# frame <-- actual image (frame) captured from the video
	has_frame, frame = source.read()
	
	# if a frame cannot be read (e.g., video ends), the loop stops
	if not has_frame:
		break

	# cv2.imshow() <-- displays the current frame in the window
	cv2.imshow(win_name, frame)

# source.release() <-- frees the video capture object (releases the webcam or closes the video file)
source.release()

# cv2.destroyWindow() <-- closes the preview window
cv2.destroyWindow(win_name)
```