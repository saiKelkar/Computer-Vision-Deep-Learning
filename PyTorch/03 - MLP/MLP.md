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

```
# Training configuration
# Negative Log Likelihood Loss
criterion = F.nll_loss

optimizer = optim.Adam(mlp_model.parameters(), lr = 1e-2)
num_epochs = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```
def train(model, trainloader, criterion, optimizer, DEVICE):
    model.train()
    model.to(DEVICE)
    running_loss = 0
    correct_predictions = 0
    total_samples = 0

    for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # clear the model's memory of old mistakes (or it will keep piling up errors)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Calculates how much the model needs to improve
        loss.backward()
        # Updates the model to make better guesses next time
        optimizer.step()

		# Adds up all the mistakes to check how much we improve
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim = 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    avg_loss = running_loss / len(trainloader)
    accuracy = 100 * correct_predictions / total_samples
    return avg_loss, accuracy
```

```
def validation(model, val_loader, criterion, DEVICE):
    model.eval()
    model.to(DEVICE)

    running_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct_predictions / total_samples
    return avg_loss, accuracy
```

```
def main(model, trainloader, val_loader, epochs = 5, DEVICE = "cuda"):

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, DEVICE)
        val_loss, val_accuracy = validation(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1:0>2}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Plotting loss and accuray
    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label = "Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label = "Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label = 'Train Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label = 'Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
```

```
main(mlp_model, train_loader, val_loader, epochs = num_epochs, DEVICE = DEVICE)
```

```
images, gt_labels = next(iter(val_loader))

rand_idx = random.choice(range(len(images)))

plt.imshow(images[rand_idx].squeeze())
plt.title("Ground Truth Label:" + str(int(gt_labels[rand_idx])), fontsize = 12)
plt.axis("off")
plt.show()
```

```
bold = f"\033[1m"
reset = f"\033[0m"
```

```
mlp_model.eval()

with torch.no_grad():
    batch_outputs = mlp_model(images.to(DEVICE))

prob_score_batch = batch_outputs.softmax(dim = 1).cpu()

prob_score_test_image = prob_score_batch[rand_idx]
pred_cls_id = prob_score_test_image.argmax()

print("Predictions for each class on the test image:\n")

for idx, cls_prob in enumerate(prob_score_test_image):
    if idx == pred_cls_id:
        print(f"{bold}Class: {idx} - {class_mapping[idx]}, Probability: {cls_prob:.3f}{reset}")
    else:
        print(f"Class: {idx} - {class_mapping[idx]}, Probability: {cls_prob:.3f}")
```

```
!pip install scikit-learn
!pip install seaborn
from sklearn.metrics import confusion_matrix
import seaborn as nn
```

```
def prediction_batch(model, batch_inputs):
    model.eval()

    batch_outputs = model(batch_inputs)

    with torch.no_grad():
        batch_probs = batch_outputs.softmax(dim = 1)

    batch_cls_ids = batch_probs.argmax(dim = 1)

    return batch_cls_ids.cpu()
```

```
val_target_labels = []
val_predicted_labels = []

for image_batch, target_batch in val_loader:
    image_batch = image_batch.to(DEVICE)

    batch_pred_cls_id = prediction_batch(mlp_model, image_batch)

    val_predicted_labels.append(batch_pred_cls_id)
    val_target_labels.append(target_batch)

val_target_labels = torch.cat(val_target_labels).numpy()
val_predicted_labels = torch.cat(val_predicted_labels).numpy()
```

```
cm = confusion_matrix(y_true=val_target_labels, y_pred = val_predicted_labels)

plt.figure(figsize= [15,8])

# Plot the confusion matrix as a heatmap.
nn.heatmap(cm, annot=True, fmt='d', annot_kws={"size":14})
plt.xlabel("Predicted")
plt.ylabel("Targets")
plt.title(f"Confusion Matrix", color="gray")
plt.show()
```