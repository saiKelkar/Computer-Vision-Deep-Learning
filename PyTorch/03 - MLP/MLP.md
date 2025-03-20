```
# Gives a detailed summary of the model's architecture -- kind of like model.summary() in Keras
# It helps us inspect and debug our neural networks by giving us insights into the layer structure, output shapes, and number of parameters
!pip install -q torchinfo
```

```
import torch
from torchvision import datasets, transforms

# Load the Fashion MNIST dataset (without normalization)
# dataset object behaves like a Python list of (image, label) pair
dataset = datasets.FashionMNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())

# Stack all images into a single tensor
# Flatenning the images into a 1D vector
# torch.cat <-- concatenates given sequence into a list of tensors
# img.view(-1) for img, _ in dataset <-- similar to writing for img, _ in dataset -- a loop
# img.view(-1) <-- .view(row, column) -- reshapes the vector into a format we want
# .view(-1) <-- this asks to arrange all the pixels in a straight line -- in a single row -- but since we don't know how long the row would be, we ask .view() to infer based on the input images how much pixel-long the row would be -- hence -1
all_pixels = torch.cat([img.view(-1) for img, _ in dataset])

# Compute mean and std (Mean and Standard Deviation)
# This is used to then normalize the images in the dataset
# .item() <-- converts the result to a standard Python number
mean = all_pixels.mean().item()
std = all_pixels.std().item()

# Print 4 digits after the decimal in both mean and std
print(f"Mean: {mean:.4f}, Std: {std:.4f}")
```

```
# Set a seed for reproducibility
# Seed <-- a starting point for a random number generator -- when we give a random number generator the same seed, it will always produce the same sequence of random numbers
# By setting a seed -- we make the random processes predictable -- this is called reproducibility
# Why do we need Random Numbers in ML?
# Initializing model weights <-- neural network start with random weights
# shuffling data or data augmentation 

# We iinitialize 4 random values to target different part of the system where randomness can occur
# We use multiple libraries and different devices -- each one manages its own random number generator -- seeding only one won't control randomness everywhere

# Set a seed for reproducibility
def set_seeds():
    # Set random seed values -- any integer value can be used
    SEED_VALUE = 42

	# Python's built-in randomness
    random.seed(SEED_VALUE)
    # NumPy random numbers
    np.random.seed(SEED_VALUE)
    # PyTorch randomness (CPU)
    torch.manual_seed(SEED_VALUE)

    # Fix seed to make training deterministic
    if torch.cuda.is_available():
	    # Fixes randomness for single GPU
        torch.cuda.manual_seed(SEED_VALUE)
        # Ensures reproducibility on multi-GPU setups
        torch.cuda.manual_seed_all(SEED_VALUE)
        # deterministic behaviour <-- forces cuDNN to pick the same computation path every single time (slightly slower but fully reproducible)
        torch.backends.cudnn.deterministic = True
        # Allows PyTorch to auto-tune the best kernel for GPU, which speeds up training but can introduce variability
        torch.backends.cudnn.benchmark = True

set_seeds()
```

```
# Download the training set without normalization
# transforms.Compose() <-- chains together a sequence of transformations
raw_transform = transforms.Compose([transforms.ToTensor()])
train_set_raw = datasets.FashionMNIST(root = './F_MNIST_data', download = True, train = True, transform = raw_transform)

# Compute mean and std from training set
all_pixels = torch.cat([img.view(-1) for img, _ in train_set_raw])
mean = all_pixels.mean().item()
std = all_pixels.std().item()

print(f"Computed Mean: {mean:.4f}, Computed Std: {std:.4f}")
```

```
# Define the new transform using computed mean and std
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# Reload datasets with proper normalization
# Here, as we write train = True and train = False <-- this helps PyTorch divide the data into train_set and val_set (or test_set)
train_set = datasets.FashionMNIST(root = "F_MNIST_data", download = True, train = True, transform = transform)
val_set = datasets.FashionMNIST(root = "F_MNIST_data", download = True, train = False, transform = transform)

print("Total Train Images:", len(train_set))
print("Total Val Images:", len(val_set))
```

```
# We shuffle the train dataset to avoid the MLP network from learning the sequence pattern in the dataset
train_loader = torch.utils.data.DataLoader(train_set, shuffle = True, batch_size = 64)
val_loader = torch.utils.data.DataLoader(val_set, shuffle = False, batch_size = 64)

# batch_size <-- number of samples passed to the model in one forward and backward pass during training
```

```
# class to idx mapping
class_mapping = {
    0: "Tshirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# class_mapping <-- dictionary used to map numerical labels (which the model outputs) to human-readable class names
```

```
# Dataset visualization
def visualize_images(trainloader, num_images = 20):
    fig = plt.figure(figsize = (10, 10))

    # Iterate over the first batch
    # iter(trainloader) <-- converts the trainloader -- which holds batches of images -- into an iterator
    images, labels = next(iter(trainloader))

    # To calculate the number of rows and columns for subplots
    num_rows = 4
    num_cols = int(np.ceil(num_images / num_rows))

    for idx in range(min(num_images, len(images))):
        image, label = images[idx], labels[idx]

        ax = fig.add_subplot(num_rows, num_cols, idx + 1, xticks = [], yticks = [])
        ax.imshow(np.squeeze(image), cmap = "gray")
        ax.set_title(f"{label.item()}: {class_mapping[label.item()]}")

    fig.tight_layout()
    plt.show()

visualize_images(train_loader, num_images = 16)
```

```
# Multi Layer Perceptron Model Implementation
# MLP is a custom neural network class that inherits from torch.nn.Module
class MLP(nn.module):
    def __init__(self, num_classes):
	    # super().__init() <-- calls the parent class -- nn.Module -- initializer, which is required to register all layers and parameters
	    # we are adding onto the exisitng class (nn.Module) here
        super().__init__()
        # nn.Linear(in_features, out_features) <-- in_features -- number of input features, out_features -- number of output features
        # this performs linear transformation -- output = input x W + b
        self.fc0 = nn.Linear(784, 512)
        # nn.BatchNorm1d(num_features) <-- applies batch normalization to stabilize and speed up the training
        self.bn0 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)

		# Dropout randomly turns off neurons during training with a probability p = 0.3
		# This is done to prevent overfitting -- ensures that the model does not rely on specific neurons
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.shape[0], -1)
        # First fully connected layer with ReLU, batch norm, and dropout
        # self.fc0(x) <-- performs matrix multiplication
        # self.bn0 <-- normalizes output
        # F.relu() <-- applies ReLU function
        # Dropout <-- randomly drops 30% of neuron
        x = F.relu(self.bn0(self.fc0(x)))
        x = self.dropout(x)

        x = F.relu(self.bn1(self.fc1(x)))

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        # Output layer with softmax activation
        x = F.relu(self.bn3(self.fc3(x)))

		# self.fc4(x) <-- projects the 64 features to the number of classes (10)
		# F.log_softmax() <-- converts raw output to log-probabilities
		# softmax function -- converts raw model output into probabilities
		# Log-Softmax -- takes the logarithm of those probabilities
        x = F.log_softmax(self.fc4(x), dim = 1)

        return x

# Instantiate the model
mlp_model = MLP(num_classes = 10)
```

```
# A dummy input size of (B, C, H, W) = (1, 1, 28, 28) is passed
print(summary(mlp_model, input_size = (1, 1, 28, 28), row_settings = ["var_names"]))
# Display the model summary
```

``` Output:
==========================================================================================
Layer (type (var_name))                  Output Shape              Param #
==========================================================================================
MLP (MLP)                                [1, 10]                   --
├─Linear (fc0)                           [1, 512]                  401,920
├─BatchNorm1d (bn0)                      [1, 512]                  1,024
├─Dropout (dropout)                      [1, 512]                  --
├─Linear (fc1)                           [1, 256]                  131,328
├─BatchNorm1d (bn1)                      [1, 256]                  512
├─Linear (fc2)                           [1, 128]                  32,896
├─BatchNorm1d (bn2)                      [1, 128]                  256
├─Dropout (dropout)                      [1, 128]                  --
├─Linear (fc3)                           [1, 64]                   8,256
├─BatchNorm1d (bn3)                      [1, 64]                   128
├─Linear (fc4)                           [1, 10]                   650
==========================================================================================
Total params: 576,970
Trainable params: 576,970
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 0.58
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.02
Params size (MB): 2.31
Estimated Total Size (MB): 2.33
==========================================================================================
```