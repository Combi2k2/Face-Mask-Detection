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

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5)
        
        self.pool  = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        
        self.fc3 = nn.Linear(84, 3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 16 * 13 * 13)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return  x

model = ConvNet()
model.load_state_dict(torch.load('mask-classifier.model', map_location=torch.device('cpu')))

print(model)

import cv2

cv2.namedWindow("preview")
cam = cv2.VideoCapture(0)
itr = 0

while (1):
    ret, frame = cam.read()
    key = cv2.waitKey(20)

    if (ret == False):  break
    if (key == 27):     break

    #cv2.imshow("preview", frame)

    frame = cv2.resize(frame, dsize = (64, 64), interpolation=cv2.INTER_CUBIC)
    frame = torch.from_numpy(frame)

    frame = frame.permute(2, 0, 1)

    frame = model.pool(F.relu(model.conv1(frame)))
    frame = model.pool(F.relu(model.conv2(frame)))

    print(frame)
    break

    frame = frame.permute(1, 2, 0).numpy()
    frame = cv2.resize(frame, dsize = (600, 400), interpolation=cv2.INTER_CUBIC)

    cv2.imshow(f'preview', frame)

    #if (itr == 100):
    #    cv2.imwrite('test1.jpg',frame)
    #    break
    itr += 1

cam.release()
cv2.destroyWindow("preview")