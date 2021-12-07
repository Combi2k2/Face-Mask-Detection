import torch
#from torch._C import T
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

from matplotlib import cm

import PIL

transform_face = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize([64,64]),
        transforms.ToTensor(),
])

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

for itr in range(1000000000):
    ret, frame = cam.read()
    key = cv2.waitKey(20)

    if (ret == False):  break
    if (key == 27):     break

    cv2.imshow(f'Original', frame)#np.vstack((row1, row2)))

    frame = PIL.Image.fromarray(frame)
    frame = transform_face(frame)

    frame = frame.permute(2, 0, 1)
    frame = frame.reshape(1, 3, 64, 64)

    with torch.no_grad():
        frame = model.pool(F.relu(model.conv1(frame)))
        frame = model.pool(F.relu(model.conv2(frame)))

    SAVE_FIG = False

    if cv2.waitKey(32) == ord('a'):
        SAVE_FIG = True
    
    rows = [None] * 4
    
    for i, channel in enumerate(frame[0]):
        channel = channel.numpy()
        channel = cv2.resize(channel, dsize = (500, 500), interpolation=cv2.INTER_CUBIC)

        if (i % 4 == 0):    rows[i // 4] = channel
        else:               rows[i // 4] = np.hstack((rows[i // 4], channel))
    
    PIC = np.vstack((rows[0], rows[1], rows[2], rows[3]))
    cv2.imshow(f'Transformed', PIC)#np.vstack((row1, row2)))

cam.release()
cv2.destroyWindow("preview")