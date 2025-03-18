[[Initializations and Downloads]]
[[Loading and Reading Images]]

Find key points in both images: (filled form photo and a digital form)

```
# Convert images to grayscale
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors
MAX_NUM_FEATURES = 500
orb = cv2.ORB_create(MAX_NUM_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

# Display
im1_display = cv2.drawKeypoints(im1, keypoints1, outImage = np.array([]), color = (255, 0, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

im2_display = cv2.drawKeypoints(im2, keypoints2, outImage = np.array([]), color = (255, 0, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Plot images
```

Match keypoints in two images

```
# Match features
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Converting to list for sorting as tuples are immutable objects
matches = list(matcher.match(descriptors1, descriptors2, None))

# Sort matches by score
matches.sort(key = lambda x: x.distance, reverse = False)

# Remove not so good matches
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]

# Draw top matches
im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

# Plot the images
```

Find Homography

```
# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype = np.float32)
points2 = np.zeros((len(matches), 2), dtype = np.float32)

# match.queryIdx <-- index of the keypoint in the first image
# match.trainIdx <-- index of matching keypoint in the second image
for i, match in enumerate(matches):
	points1[i, :] = keypoints1[match.queryIdx].pt
	points2[i, :] = keypoints2[match.trainIdx].pt

# Homography <-- transformation matrix that maps points from one image plane to another
# RANSAC <-- random sample consensus -- helps to filter out incorrect matches by finding the best-fit transformation
# Find homography
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
```

Wrap images

```
# Use homography to wrap image
height, width, channels = im1.shape
im2_reg = cv2.wrapPerspective(im2, h, (width, height))

# Plot results
```