Motivation:
Task of assigning an input image one label from a fixed set of categories. 

Example:
![[Pasted image 20250629171942.png]]

Challenges: 
1. Viewpoint variation -- a single instance of an object can be oriented in many ways with respect to the camera.
2. Scale variation -- Visual classes often exhibit variation in their size (size in the real world, not only in terms of their extent in the image)
3. Deformation -- Many objects of interest are not rigid bodies and can be deformed in extreme ways.
4. Occlusion -- The objects of interest can be occluded. Sometimes only a small portion of an object (as little as few pixels) could be visible.
5. Illumination conditions -- The effects of illumination are drastic on the pixel level.
6. Background clutter -- The objects of interest may blend into their environment, making them hard to identify. 
7. Intra-class variation -- The classes of interest can often be relatively broad, such as a chair. There are many different types of these objects, each with their own appearance. 

![[Pasted image 20250629172545.png]]

Data-driven approach:
Relies on accumulating a training dataset of labelled images.

Image classification pipeline:
- Input: consists of a set of N images, each labelled with one of the K different classes (referred as training set)
- Learning: use training set to learn what every one of the classes looks like (referred as training a classifier, or learning a model)
- Evaluation: evaluate the quality of the classifier by asking it to predict labels for a new set of images that it has never seen before. Then compare the true labels (also called ground truth) of these images to the ones predicted by the classifier. 

Nearest Neighbour Classifier:
(Rarely used in practice, but allows us to get a general idea about the basic approach to an image classification problem)

Classifier -- takes a test image -- compares it to every single one of the training images -- predicts the label of the closest training image

How to compare two images?
(L1 distance)
e.g., two blocks of 32 x 32 x 3 -- compare them pixel by pixel -- and add up all the differences

$d_\text{1}$ ($I_\text{1}$,$I_\text{2}$) = $\sum_{p}$$\left| I_\text{1}^p - I_\text{2}^p \right|$ 

![[Pasted image 20250629174454.png]]

