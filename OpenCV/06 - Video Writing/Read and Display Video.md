[[Initializations and Downloads]]

```
# Read video from source
source = 'race_car.mp4' # source = 0 for webcam
cap = cv2.VideoCapture(source)

if not cap.isOpened():
	print("Error opening video stream or file")
```

```
# Read and display one frame
ret, frame = cap.read()
plt.imshow(frame[:, :, ::-1])
plt.show()

# To display the video file
video = YouTubeVideo("Video_Name", width = 700, height = 438)
display(video)
```

