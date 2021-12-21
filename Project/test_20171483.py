# -*- coding: utf-8 -*-


import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

import random
import os
import glob


class MyDataset(Dataset):
    def __init__(self, npy_dir):
        self.dir_path = npy_dir
        self.to_tensor = transforms.ToTensor()

        # all npy path
        self.npy_path = glob.glob(os.path.join(npy_dir, '*','*.npy'))

    def __getitem__(self, index):
        # load data
        single_data_path = self.npy_path[index]
        data = np.load(single_data_path, allow_pickle=True)
        
        image = data[0]
        image = self.to_tensor(image)
        label = data[1]
       
        return (image, label)

    def __len__(self):
        return len(self.npy_path)

from google.colab import drive
drive.mount('/content/drive')

# !unzip /content/drive/MyDrive/Font_npy_90_val.zip



# load dataset
# google drive 내 디렉토리 사용
valid_data = MyDataset("/content/Font_npy_90_val")


# define dataloader
batch_size = 64
num_classes = 52
num_epochs = 7

valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                           batch_size=batch_size,
                                           shuffle=False)

class ConvNet(nn.Module):
  def __init__(self, num_classes = 52):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Sequential( # 7 x 7 kernel
        nn.Conv2d(1, 16, 7, stride=1, padding=3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.conv2 = nn.Sequential( # 1st bottleneck layer
        nn.Conv2d(16, 16, 1, stride=1, padding=0),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 1, stride=1, padding=0),
        nn.BatchNorm2d(32),

        nn.Conv2d(32, 16, 1, stride=1, padding=0),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 1, stride=1, padding=0),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.conv3 = nn.Sequential( # 2nd bottleneck layer
        nn.Conv2d(32, 32, 1, stride=1, padding=0),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 128, 1, stride=1, padding=0),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(128, 32, 1, stride=1, padding=0),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 128, 1, stride=1, padding=0),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.conv4 = nn.Sequential( # 3rd bottleneck layer
        nn.Conv2d(128, 128, 1, stride=1, padding=0),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 256, 1, stride=1, padding=0),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.Conv2d(256, 128, 1, stride=1, padding=0),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 256, 1, stride=1, padding=0),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.conv5 = nn.Sequential( # 4th bottleneck layer
        nn.Conv2d(256, 256, 1, stride=1, padding=0),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 512, 1, stride=1, padding=0),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.Conv2d(512, 256, 1, stride=1, padding=0),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 512, 1, stride=1, padding=0),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.AvgPool2d(kernel_size=2, stride=2) # Average pooling
    )
        
    self.conv_dropout = nn.Dropout(0.5) # dropout after conv layer
    self.lin_dropout = nn.Dropout(0.3) # dropout after first fc layer

    self.fc1 = nn.Sequential(
        nn.Linear(2048 , 256),
        nn.BatchNorm1d(256)
    )

    self.fc2 = nn.Linear(256, 52)
            
  def forward(self, x):

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)

    x = x.reshape(x.size(0), -1)
    x = self.conv_dropout(x)

    x = F.relu(self.fc1(x))
    x = self.lin_dropout(x)
    x = F.softmax(self.fc2(x), dim = 1)
    return x
  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)

# Model test
model.load_state_dict(torch.load('/content/20171483.pth'))

criterion = nn.CrossEntropyLoss()
loss = 0
model.eval()

with torch.no_grad():
    correct = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss += criterion(outputs, labels).item()

        _,predicted = torch.max(outputs.data,1)
        correct += (predicted==labels).sum()

    print('Accuracy of the 20171483.pth network on the {} test images: {} %'.format(len(valid_data), 100 * correct / len(valid_data)))
    print('Loss of the 20171483.pth network on the {} test images: {:.4f}'.format(len(valid_data), loss / len(valid_loader)))
