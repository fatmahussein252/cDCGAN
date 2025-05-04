# ---- Dependencies
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

# Set random seed for reproducibility across runs
torch.manual_seed(42)

# Apply transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the data as normalization will guarantee N(0,1) distribution
])

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1: Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv1_out_dim = (self.conv1.out_channels,
                              (28 - self.conv1.kernel_size[0] + 2 * self.conv1.padding[0]) // self.conv1.stride[0] + 1,
                              (28 - self.conv1.kernel_size[0] + 2 * self.conv1.padding[0]) // self.conv1.stride[0] + 1)

        # Pooling Layer
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool1_out_dim = (self.conv1.out_channels,
                              (self.conv1_out_dim[1] - self.pool1.kernel_size + 2 * self.pool1.padding) // self.pool1.stride + 1,
                              (self.conv1_out_dim[1] - self.pool1.kernel_size + 2 * self.pool1.padding) // self.pool1.stride + 1)

        # Layer 2: Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=self.pool1_out_dim[0], out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv2_out_dim = (self.conv2.out_channels,
                              (self.pool1_out_dim[1] - self.conv2.kernel_size[0] + 2 * self.conv2.padding[0]) // self.conv2.stride[0] + 1,
                              (self.pool1_out_dim[1] - self.conv2.kernel_size[0] + 2 * self.conv2.padding[0]) // self.conv2.stride[0] + 1)

        # Pooling Layer
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool2_out_dim = (self.conv2.out_channels,
                              (self.conv2_out_dim[1] - self.pool2.kernel_size + 2 * self.pool2.padding) // self.pool2.stride + 1,
                              (self.conv2_out_dim[1] - self.pool2.kernel_size + 2 * self.pool2.padding) // self.pool2.stride + 1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=int(np.prod(self.pool2_out_dim)), out_features=120)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=84)
        self.dropout2 = nn.Dropout(0.5)
        
        self.out_layer = nn.Linear(in_features=self.fc2.out_features, out_features=10)

    def forward(self, input_image):
        # Layer 1: Conv + ReLU + Pool
        layer1_out = self.pool1(F.relu(self.conv1(input_image)))

        # Layer 2: Conv + ReLU + Pool
        layer2_out = self.pool2(F.relu(self.conv2(layer1_out)))

        # Flatten the output for fully connected layers
        layer2_out = layer2_out.view(-1, int(np.prod(self.pool2_out_dim)))

        # Fully Connected Layers
        fc1_out = self.dropout1(F.relu(self.fc1(layer2_out)))
        fc2_out = self.dropout2(F.relu(self.fc2(fc1_out)))
        output = self.out_layer(fc2_out)  # Output: 10 classes

        return output


def check_accuracy(model, dataloader, data_length):
    # Function to test and calculate accuracy of a pre-trained model
    model.eval()
    correct = 0
    for i, (inputs, labels) in enumerate(dataloader):
        outputs = F.softmax(model(inputs), dim=1)
        _, predictions = torch.max(outputs, 1)
        correct += (predictions == labels).sum()
    return correct / data_length * 100



# Training loop
def train_loop(model, train_dataloader, data_length, num_epochs):
    criterion = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            # Zero the parameter gradients to avoid grads accumulation
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Accuracy: {check_accuracy(model, train_dataloader, data_length):.1f}%")


