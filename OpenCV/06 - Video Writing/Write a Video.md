```
# VideoWriter object = cv2.VideoWriter(filename, fourcc, fps, frameSize)

filename <-- name of output video file
fourcc <-- 4-character code of codec used to compress the frames
for example:
VideoWriter::fourcc('P', 'I', 'M', '1') is a MPEG-1 codec
fps <-- frame rate of the created video stream
frameSize <-- size of video frames

# Default resolutions of the frames are obtained
# Convert the resolution from float to integer

# cap.get(3) <-- retrieves the frame width in pixels -- id 3
# cap.get(4) <-- retrieves the frame height in pixels -- id 4
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
out_avi = cv2.VideoWriter("race_car_out.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (frame_width, frame_height))

out_mp4 = cv2.VideoWriter("race_car_out.mp4", cv2.VideoWriter_fourcc(*"XVID"), 10, (frame_width, frame_height))

# Read frames and write to file
while cap.isOpened():
	ret, frame = cap.read()

	if ret:
		out_avi.write(frame)
		out.mp4.write(frame)

	else:
		break

# When everything is done, release the VideoCapture and VideoWriter objects
cap.release()
out_avi.release()
out_mp4.release()
```

```
# To display on Google Colab or Jupyter Notebook <-- we install and use the ffmpeg package
Using ffmpeg, we will change the encoding of the .mp4 file from XVID to H264

# Installing ffmpeg
!pip install imageio[ffmmpeg]

import imageio_ffmpeg as ffmpeg
# built-in Python module that allows to run system-level commands like ffmpeg
import subprocess

# Change video encoding of mp4 file from XVID to H264
input_file = "race_car_out.mp4"
output_file = "race_car_out_x264.mp4"

# Get the correct path to the ffmpeg executable
# ffmpeg.get_ffmpeg_exe() <-- finds the path of the ffmpeg executable installed by imageio[ffmpeg]
# ffmpeg may not be globally accessible on the system, hence this ensures we use the correct executable path
ffmpeg_path = ffmpeg.get_ffmpeg_exe()

# Run ffmpeg
-y <-- automatically overwrite the output file (if it exists)
-i input_file <-- input video file (XVID - encoded video)
-c:v libx264 <-- convert the video codec to H264 -- widely supported and efficient format
-hide_banner <-- hides ffmpeg's startup information
-loglevel error <-- shows only error, suppressing unnecessary output
subprocess.run([ffmpeg_path, "-y", "-i", input_file, "-c:v", "libx264", output_file, "-hide_banner", "-loglevel", "error"])

print("Video conversion to H.264 completed successfully")
```

```
# Render the converted mp4 video
# rb <-- read binary (we use binary mode as video files are stored as binary data, not text)
with open(output_file, "rb") as  mp4_file:
	mp4_data = mp4_file.read()

# b64encode(mp4_data) <-- encodes the binary video data into Base64 format
# Base64 encoding converts binary data (like videos, images) into a text representation
# Why Base64 <-- it allows us to embed and display the video directly in a Jupyter Notebook without saving it to a public URL
data_url = "data:video/mp4;base64," + b64encode(mp4_data).decode()

# We embed the video directly into the notebook interface <-- we can watch it without needing an external video player
HTML(f"""<video width = 700 controls><source src = "{data_url}" type = "video/mp4"></video>""")
```