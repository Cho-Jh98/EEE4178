# -*- coding: utf-8 -*-
"""
인공지능개론_project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ES-0QW6nFjvGiKglK_sEa6babioXILKv
"""

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms
import torchsummary

import matplotlib.pyplot as plt
import random
import glob
import time
import os


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

 !unzip /content/drive/MyDrive/Font_npy_90_train.zip
 !unzip /content/drive/MyDrive/Font_npy_90_val.zip



# load dataset
# google drive 내 디렉토리 사용
train_data = MyDataset("/content/Font_npy_90_train")
valid_data = MyDataset("/content/Font_npy_90_val") 


# define dataloader
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                           batch_size=batch_size,
                                           shuffle=False)

# visualize data
label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'P': 24, 'Q': 25, 
'R': 26, 'S': 27, 'T': 28, 'U': 29, 'V': 30, 'W': 31, 'X': 32, 'Y': 33, 
'Z': 34, 'a': 35, 'b': 36, 'd': 37, 'e': 38, 'f': 39, 'g': 40, 'h': 41, 
'i': 42, 'j': 43, 'm': 44, 'n': 45, 'o': 46, 'q': 47, 'r': 48, 't': 49, 'u': 50, 'y': 51}

# image_show function : num 수 만큼 dataset 내의 data를 보여주는 함수
def image_show(dataset, num):
  fig = plt.figure(figsize=(10,10))

  for i in range(num):
    plt.subplot(1, num, i+1)
    plt.imshow(dataset[i+1200][0].squeeze(), cmap = "gray")
    plt.title(dataset[i+1200][1])

image_show(train_data, 8)

batch_size = 64
num_classes = 52

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

# weight initializer (He_initializer)
def weight_init_He_Uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(submodule.weight, nonlinearity='relu')
        submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

model.apply(weight_init_He_Uniform)

batch_size = 64
max_pool_kernel = 2
num_epochs = 7
num_classes = 52

# uncomment code below to see model structure
# torchsummary.summary(model, (1, 90, 90), batch_size = batch_size, device = 'cuda')

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# learning rate scheduler : decrease to 1/2 after 3 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

total_step = len(train_loader)
train_loss_list = [] 
train_loss_value = 0
val_loss_list = []
val_loss_value = 0

train_acc_list = []
train_acc_value = 0
val_acc_list = []
val_acc_value = 0
save_best = 0

# Train
start = time.time()
for epoch in range(num_epochs):
  total = 0
  correct = 0
  model.train()
  print("============training epoch {}============".format(epoch+1))
  for i, (images, labels) in enumerate(train_loader):

    # Assign Tensors to Configured Device
    images = images.to(device)
    labels = labels.to(device)

    # Forward Propagation
    outputs = model(images)

    # Get Loss, Compute Gradient, Update Parameters
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # evaluate loss and accuracy of train data set
    _, predicted = torch.max(outputs, 1)
    correct += (predicted == labels).sum().item()
    train_loss_value = loss.item()      
    if (i+1) %100 == 0 :
      train_loss_list.append(train_loss_value)
      print("Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))


  train_acc_value = correct*100 / (len(train_loader)*batch_size)
  train_acc_list.append(train_acc_value)

  # Print Loss and Accuracy of Training
  print('\nCompleted training Epoch', epoch + 1, 
        '\n Training Accuracy: %.2f%%' %(train_acc_value),
        '\n Training Loss: %.4f' %train_loss_value)
  
  # calculate and print Loss and Accuracy of Training
  model.eval()
  with torch.no_grad():
      correct, val_acc, val_loss = 0, 0, 0
    
      for images, labels in valid_loader:
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          val_loss += criterion(outputs, labels)
          _,predicted = torch.max(outputs.data,1)
          correct += (predicted==labels).sum()

      val_loss = val_loss/len(valid_loader)
      val_acc = 100*correct/len(valid_data)
      val_acc_list.append(val_acc)
      print('\nAccuracy and Loss for {} valid images\n valindation accuracy: {:.2f}%\n validation loss: {:.4f}'.format(len(valid_data), 100 * correct / len(valid_data), val_loss))

  if val_acc > save_best:
    print("Your model is improved than the last model and saved as \'model_{:02d}.pth\'\n\n".format(epoch+1))
    torch.save(model.state_dict(), 'model_epoch_{:02d}.pth'.format(epoch+1))
    save_best = val_acc
  else:
    print("Your model haven't improved from last one.\n\n")


end = time.time()
# print the time it took for training
print("Train takes {:.2f}minutes".format((end-start)/60))

# print loss curve
loss_plot = plt.figure(figsize=(9,6))
loss_plot = plt.plot(train_loss_list[::5], label = 'train')
loss_plot = plt.title('loss curve')
loss_plot = plt.xlabel('Epoch')
loss_plot = plt.ylabel('Loss')
loss_plot = plt.legend(loc='best')
plt.show(loss_plot)

# print acc curve
acc_plot = plt.figure(figsize=(9,6))
acc_plot = plt.plot(train_acc_list, label = 'train')
acc_plot = plt.plot(val_acc_list, label = 'val')
acc_plot = plt.title('acc curve')
acc_plot = plt.xlabel('Epoch')
acc_plot = plt.ylabel('Accuracy')
acc_plot = plt.legend(loc='best')
plt.show(acc_plot)


# Model test
model_test = ConvNet().to(device)
model_test.load_state_dict(torch.load('20171483_model.pth'))
model_test.eval()

with torch.no_grad():
    correct = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_test(images)
        _,predicted = torch.max(outputs.data,1)
        correct += (predicted==labels).sum()

    print('Accuracy of the last_model network on the {} test images: {} %'.format(len(valid_data), 100 * correct / len(valid_data)))
