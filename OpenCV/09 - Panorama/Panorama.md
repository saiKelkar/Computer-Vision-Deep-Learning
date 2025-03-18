[[Initializations and Downloads]]
[[Loading and Reading Images]]

```
# Read Images

# glob <-- used to find file paths matching a specific pattern
# glob.glob() <-- returns a list of file paths matching the given pattern
imagefiles = glob.glob(f"boat{os.sep}*")
imagefiles.sort()

images = []
for filename in imagefiles:
	img = cv2.imread(filename)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	images.append(img)

num_images = len(images)


# Display Images

plt.figure(figsize = [30, 10])
num_cols = 3
num_rows = math.ceil(num_images / num_cols)
for i in range(0, num_images):
	plt.subplot(num_rows, num_cols, i + 1)
	plt.axis("off")
	plt.imshow(images[i])
```

```
# Stitch images
stitcher = cv2.Stitcher_create()
status, result = stitcher.stitch(images)

if status == 0:
	plt.figure(figsize = [30, 10])
	plt.imshow(result)
	plt.show()
```