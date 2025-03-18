[[Initializations and Downloads]]

```
# Capture multiple exposures of same scene
def readImagesAndTimes():
	# List of file names
	filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]

	# List of exposure times
	times = np.array([1 / 30.0, 0.25, 2.5, 15.0], dtype = np.float32)

	# Read images
	images = []
	for filename in filenames:
		im = cv2.imread(filename)
		images.append(im)
	
	return images, times
```

```
# Align images
# Read images and exposure times
images, times = readImagesAndTimes()

# Align Images
# cv2.createAlignMTB() <-- used to create Median Threshold Bitmap object, which helps align images that are taken under different exposure levels
alignMTB = cv2.createAlignMTB()
alignMTB.process(images, images)

# Find Camera Response Function (CRF)
calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)

# Plot CRF
x = np.arange(256, dtype = np.uint8)
y = np.squuze(responseDebevec)

ax = plt.figure(figsize = (30, 10))
plt.title("Debevec Inverse Camera Response Function", fontsize = 24)
plt.xlabel("Measured Pixel Value", fontsize = 22)
plt.ylabel("Calibrated Intensity", fontsize = 22)
plt.xlim([0, 260])
plt.grid()
plt.plot(x, y[:, 0], "b", x, y[:, 1], "g", x, [:, 2], "r")

# Merge images into an HDR linear image
mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
```

```
# Tonemapping
# Tonemap using Drago's method to obtain 24-bit color image
tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
ldrDrago = tonemapDrago.process(hdrDebevec)
ldrDrago = 3 * ldrDrago

# Saving images
cv2.imwrite("ldr-Drago.jpg", 255 * ldrDrago)

# Plotting image


# Tonemap using Reinhard's method
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
ldrReinhard = tonemapReinhard.process(hdrDebevec)

# Saving images
cv2.imwrite("ldr-Reinhard.jpg", 255 * ldrReinhard)


# Tonemap using Mantiuk's method
tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
ldrMantiuk = 3 * ldrMantiuk

# Save image
# Plot image
```