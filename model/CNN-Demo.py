# -*- coding:utf-8 -*-
"""

    refer: https://raw.githubusercontent.com/yunjey/pytorch-tutorial/master/tutorials/02-intermediate/convolutional_neural_network/main.py
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

# Device configuration
from preprocess.TrafficDataset import TrafficDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 500
num_classes = 4
batch_size = 100
learning_rate = 0.001
#
# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=False,
#                                           transform=transforms.ToTensor())
input_file = '../data/data_split_train_v2_711/train_1pkt_images_merged.csv'
train_data = TrafficDataset(input_file, transform=None)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=30, shuffle=True, num_workers=4)
test_loader = train_loader


#
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x1 square convolution
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 1), stride=1),
            # nn.BatchNorm2d(16),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=(2,1), stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 1), stride=1, padding=0),
            # nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=(2,1), stride=2)
        )
        self.fc = nn.Linear(32 * 52 * 1, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        images = images.view([images.shape[0], 1, -1, 1])
        images = Variable(images).float()
        labels = Variable(labels).type(torch.LongTensor)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = images.view([images.shape[0], 1, -1, 1])  # (nSamples, nChannels, x_Height, x_Width)
        images = Variable(images).float()
        labels = Variable(labels).type(torch.LongTensor)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
