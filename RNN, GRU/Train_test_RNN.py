import torch
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import random_split

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.FashionMNIST(root='./datasets',
                                                  train=True,
                                                  transform=transform,
                                                  download=True)

test_data = torchvision.datasets.FashionMNIST(root='./datasets',
                                              train=False,
                                              transform=transform,
                                              download=True)

num_classes = len(train_dataset.classes)
in_channel = 1 # gray_scale

batch_size = 64
learning_rate = 0.0005
num_epochs = 20
max_pool_kernel = 2

input_size = 28
sequence_length = 28

out_node = num_classes
num_layers = 3
hidden_size = 256

# train_valdiation split with 0.1 validation
train_dataset, val_dataset = random_split(train_dataset, [6000,54000])

train_loader = DataLoader(dataset=val_dataset, 
                          batch_size = batch_size,
                          shuffle = True)

val_loader = DataLoader(dataset=val_dataset, 
                      batch_size = batch_size,
                      shuffle = False)

test_loader = DataLoader(dataset=test_data, 
                         batch_size=batch_size,
                         shuffle=False)

# print number of training/test dataset and shape of image
print('number of training dataset:', len(train_loader) * batch_size)
print('number of test dataset:', len(test_data.data))
print('type and shape of dataset:', train_loader.dataset.dataset.data[0].shape)

# plot MNIST_fashion image
row, col = 5,5
count = 1
plt.figure(figsize=(10,10))
plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=0.4)

for a in range(row):
  for b in range(col):
    plt.subplot(row, col, count)
    count += 1
    plt.imshow(test_data.data[count].reshape(28,28), cmap='gray')
    plt.title(test_data.classes[test_data[count][1]])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model
class RNN(nn.Module):
  def __init__(self, intput_size, hidden_size, num_layers, num_classes):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4)
    self.fc = nn.Linear(hidden_size, num_classes)

    torch.nn.init.xavier_uniform_(self.fc.weight) # weight initializer


  def forward(self, x):
    # set initial hidden states
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # torch.size([10, 50, 128]), don't need for RNN training

    #Forward propagate RNN
    out, _  = self.rnn(x, (h0)) # output: tensor [batch_size, seq_length, hidden_size]

    #Decode the hidden state of the last time step
    out = self.fc(out[:,-1,:])

    return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# count number of parameters in each layer
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Parameters in RNN model")
for p in model.parameters():
  print(p.size())
print("\n")

# count total number of parameters of model
model_hp = count_parameters(model)
print('model\'s hyper parameters', model_hp)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

####### Train #######
train_loss_list = [] 
train_loss_value = 0
val_loss_list = []
val_loss_value = 0

train_acc_list = []
train_acc_value = 0
val_acc_list = []
val_acc_value = 0

best_model_loss, best_epoch = 10, 0

total_step = len(train_loader)
for epoch in range(num_epochs):
  correct = 0
  model.train()
  for i, (image, label) in enumerate(train_loader):
    model.train()
    image = image.reshape(-1, sequence_length, input_size).to(device)
    label = label.to(device)

    # Forward
    output = model(image)
    loss = criterion(output, label)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # evaluate accuracy and loss
    _, predicted = torch.max(output, 1)
    correct += (predicted == label).sum().item()
    train_loss_value += loss.item()


  train_acc_value = correct/(len(train_loader) * batch_size)
  train_loss_value = train_loss_value/len(train_loader)
  train_acc_list.append(train_acc_value)
  train_loss_list.append(train_loss_value)

  print("===Epoch {}===\ntrain loss: {:.4f}, train accuracy:{:.4f}".format(epoch, loss.item(), train_acc_value))
  
  # calculate accuracy and loss of validation set
  with torch.no_grad():
    val_loss_value = 0
    val_acc_value = 0
    val_correct = 0
    model.eval()
    for k, (image, label) in enumerate(val_loader):
      image = image.reshape(-1, sequence_length, input_size).to(device)
      label = label.to(device)

      out = model(image)
      val_loss = criterion(out, label).item()
      val_loss_value += val_loss
      _, val_predicted = torch.max(out, 1)
      val_correct += (val_predicted == label).sum().item()
        
    val_acc_value = val_correct/(len(val_loader) * batch_size)
    val_loss_value = val_loss_value/len(val_loader)

    val_acc_list.append(val_acc_value)
    val_loss_list.append(val_loss_value)
    print("validaiton loss: {:.4f}, validation accuracy: {:.4f}\n".format(val_loss_value, val_acc_value))
  
  # save model
  if best_model_loss > val_loss_value :
    best_model_loss = val_loss_value
    best_epoch = epoch
    print("your model is saved as RNN_epoch_{}.pth\n\n".format(best_epoch))
    torch.save(model.state_dict(), "RNN_epoch_{}.pth".format(best_epoch))

test_model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
test_model.load_state_dict(torch.load('./rnn_20171483.pth'))
test_model.eval()
with torch.no_grad():
  correct = 0

  for image, label in test_loader:
    image = image.reshape(-1, sequence_length, input_size).to(device)
    label = label.to(device)

    output = test_model(image)

    _, pred = torch.max(output.data, 1)
    correct += (pred == label).sum().item()

  print('Test Accuracy of RNN model on the {} test images: {}%'.format(len(test_data), 100 * correct / len(test_data)))

## plot loss curve

loss_plot = plt.figure(figsize=(15,9))
loss_plot = plt.plot(train_loss_list, label = 'train')
loss_plot = plt.plot(val_loss_list, label = 'val')

loss_plot = plt.title('loss curve')
loss_plot = plt.xlabel('loss')
loss_plot = plt.ylabel('Epoch')

loss_plot = plt.legend(loc='best')
plt.show(loss_plot)

## plot accuracy curve

acc_plot = plt.figure(figsize=(15,9))
acc_plot = plt.plot(train_acc_list, label = 'train')
acc_plot = plt.plot(val_acc_list, label = 'val')

acc_plot = plt.title('accuracy curve')
acc_plot = plt.xlabel('accuracy (%)')
acc_plot = plt.ylabel('Epoch')

acc_plot = plt.legend(loc='best')

plt.show(acc_plot)
