import torch
import torchvision
import pandas as pd
import torch.nn as nn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid

import torchvision.transforms as transforms
import numpy as np

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

#import modules

classes = ['mask_weared_incorrect', 'with_mask', 'without_mask']

#here I train the model directly on moodle so you can consider downloading the dataset and change the DATA_DIR or also training your model on kaggle
data_dir = '../input/face-mask-detection/Dataset' # path to the database

transform_face = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize([64,64]),
        transforms.ToTensor(),
])

#preparing dataset
dataset = ImageFolder(data_dir, transform = transform_face)

#split the dataset into 2 train_ds and valid_ds for avoiding overfitting
train_len = int(0.8 * len(dataset))
valid_len = len(dataset) - train_len

train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_len, valid_len])

#preparing the device and other parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
batch_size = 50
learning_rate = 0.01


#preparing the dataloader
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

#to check if our batch of data is properly shuffled
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.imshow(make_grid(images, nrow=10).permute(1, 2, 0))
        break

show_batch(train_dl)

#declare the model
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5)
        
        self.pool  = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        
        self.fc3 = nn.Linear(84, len(classes))
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 16 * 13 * 13)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return  x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#calculating the accuracy of dataset
def acc(data):
    n_correct = 0
    n_samples = 0
    
    for (images, labels) in data:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        
        n_correct += (predictions == labels).sum().item()
        n_samples += len(images)
    
    return  100 * n_correct / n_samples

#training loop
valid_acc = []
train_acc = []
train_loss = []

for epoch in range(num_epochs):
    n_correct = 0
    n_samples = len(train_ds)
    
    losses = []
    
    for (images, labels) in train_dl:
        images = images.to(device)
        labels = labels.to(device)

        #forward step
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        losses.append(loss.item())
        
        #backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        train_loss.append(np.mean(losses))
        
        train_acc.append(acc(train_dl))
        valid_acc.append(acc(valid_dl))
        
        if (len(valid_acc) > 1 and valid_acc[-1] < valid_acc[-2]):
            #this is a well known method for avoiding overfitting
            break
        
        print(f'Epoch {epoch + 1}/{num_epochs}: train_loss = {train_loss[-1]:.4f}, train_acc = {train_acc[-1]}%, valid_acc = {valid_acc[-1]}%')
    
#saving model
torch.save(model.state_dict(), 'mask-classifier.model')