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