"""Pytorch custom loss function example

This file demonstrates how to train a model using a
binary crossentropy loss function

"""
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
from typing import Any
from layer import Featureset, Train, Dataset
from PIL import Image
import io
import base64


# Pytorch custom loss function    

class Custom_BCE(nn.Module):
  def __init__(self):
    super(Custom_BCE, self).__init__()

  def forward(self, y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    loss_value=y_true*torch.log2(y_pred) + (1-y_pred)*torch.log2(1-y_pred)
    return loss_value
  
criterion = Custom_BCE()


def train_model(train: Train, ds:Dataset("catsdogs"), pf: Featureset("cat_and_dog_features")) -> Any:
   
    df = ds.to_pandas().merge(pf.to_pandas(), on='id')
    training_set = df[(df['path'] == 'training_set/dogs') | (df['path'] == 'training_set/cats')]
    testing_set = df[(df['path'] == 'test_set/dogs') | (df['path'] == 'test_set/cats')]
    X_train = torch.stack(training_set['content'].map(load_process_train_images))
    X_test = torch.stack(testing_set['content'].map(load_process_test_images))
    train.register_input(X_train)
    train.register_output(df['category'])

    training_dataset = TensorDataset(X_train, torch.Tensor(training_set['category'].values))
    testing_dataset = TensorDataset(X_test, torch.Tensor(testing_set['category'].values))

    training_data = DataLoader(training_dataset, 16)
    testing_data = DataLoader(testing_dataset, 16)

    loss = []

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 1)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.sigmoid(self.fc3(x))
            return x
        
    net = Net()


    criterion = Custom_BCE()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(training_data, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                loss.append(running_loss)
                running_loss = 0.0

    print('Finished Training')

    train.log_metric("Training Loss", loss)
    return net


def train_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomCrop(204),
        transforms.Normalize((0, 0, 0),(1, 1, 1))
    ])

def test_transforms(): 
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0),(1, 1, 1))
    ])


def load_process_train_images(content):
    image_decoded = base64.b64decode(content)
    image = Image.open(io.BytesIO(image_decoded)).resize([224, 224])
    image = train_transforms(image)
    return image

def load_process_test_images(content):
    image_decoded = base64.b64decode(content)
    image = Image.open(io.BytesIO(image_decoded)).resize([224, 224])
    image = test_transforms(image)
    return image
