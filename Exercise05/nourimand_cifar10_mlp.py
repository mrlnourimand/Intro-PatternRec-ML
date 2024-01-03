"""
DATA.ML.100: Introduction to Pattern Recognition and Machine Learning
Ex 05, title: Neural network learning (CIFAR-10).

(This program works with CIFAR-10 dataset. It uses training dataset to train the
 Neural Network(NN), then uses test data to see how the data is rained and
 prints its accuracy.)
 I used PyTorch in this program. I started with 5 neurons which was too small.
 Then I increased the neurons to 100 in the first fully connected layer, add
 another layer with 100-input and 25-output and finally the layer with 25-input
 and 10-output(CIFAR-10 classes)
 I changed the parameters, changes learning rate, played with the activation
 methods(ReLu, Sigmoid, Dropout) and also with optimization algorithms(SGD and
 Adam). The screenshot is the best result among them with 46.51% accuracy in 200
 epochs (it was 47.84% accuracy in 100 epochs).

Creator: Maral Nourimand
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from random import random
from tqdm import tqdm


# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3072, 100)  # Input size: 32x32x3 = 3072
        self.fc12 = nn.Linear(100, 25)  # adding another layer
        self.fc2 = nn.Linear(25, 10)  # Output size: 10 (CIFAR-10 classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input, reshaping it into a 1D tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc12(x))
        # x = self.drop(x)
        x = self.sigmoid(self.fc2(x))
        return x


def unpickle(file):
    """
    Function to load CIFAR-10 data

    :param file: directory of the loaded file.
    :return: the CIFAR-10 dataset.
    """
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


def display_random_images(X, Y, label_names):
    """
    Function to display random CIFAR-10 images

    :param X: dictionary of data
    :param Y: dictionary of labels of data
    :param label_names: name of the label classes
    """
    for i in range(X.shape[0]):
        # Show some images randomly
        if random() > 0.999:
            plt.figure(1)
            plt.clf()
            plt.imshow(X[i])
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
            plt.pause(1)


def main():

    ############################################################################
    #                            PART 1                                        #
    ############################################################################

    # Load CIFAR-10 data
    datadict = unpickle(r'cifar-10-batches-py\data_batch_1')
    X = datadict["data"]
    Y = datadict["labels"]

    labeldict = unpickle(r'cifar-10-batches-py\batches.meta')
    label_names = labeldict["label_names"]

    # Reshape and convert data
    # This line of code reshapes the image data into a 4D tensor and rearranges
    # the dimensions to match the expected format for image data in deep
    # learning. It also ensures that the data is represented as unsigned 8-bit
    # integers, which is suitable for image pixel values.
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y = np.array(Y)

    # Display random images
    display_random_images(X, Y, label_names)

    ############################################################################
    #                               PART 2                                     #
    ############################################################################

    # Create an instance of the SimpleNN model
    model = SimpleNN()

    # Define CIFAR-10 dataset transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    # Load CIFAR-10 training data and create a data loader
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 200
    losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")
        losses.append(running_loss / len(trainloader))

    # Plot the training loss curve
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')

    # Testing the model on CIFAR-10 test data
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print classification accuracy for test data
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    plt.show()


if __name__ == "__main__":
    main()


