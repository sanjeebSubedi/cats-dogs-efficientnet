import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import PIL
import multiprocessing
import torch.optim as optim
import numpy as np
import math
from efficientnet_pytorch import EfficientNet
from customDataset import CatsDogsDataset
import runtimeVariables as vars

def set_device():
  global device
  device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
  print(f'Device = {device}')

def load_data(main_dir, transformations, train_split, batch_size=32,  is_test_set=False):
    dataset = CatsDogsDataset(main_dir, transform = transforms.Compose(transformations), is_test_set=is_test_set)
    train_size = math.ceil(train_split*len(dataset))
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, valid_loader    

def get_model(model_name, num_of_classes, input_channels, image_size):
  return EfficientNet.from_pretrained(model_name, num_classes=num_of_classes,
                                      in_channels=input_channels,
                                      image_size=image_size)
  
def adam(model, lr):
  return optim.Adam(model.parameters(), lr=lr)

def get_num_correct(preds, labels):
  return preds.argmax(dim=1).eq(labels).sum().item()

def train_model(train_loader, model, optimizer, num_epochs):
  for epoch in range(num_epochs):
    loss_after_epoch = accuracy_after_epoch = 0
    for batch in train_loader:
      images = batch[0].to(device)
      labels = batch[1].to(device)
      # print(labels)
      # images = images.cuda()
      optimizer.zero_grad()
  
      preds = model(images)
      # print(preds.argmax(dim=1))
      loss = F.cross_entropy(preds, labels)
  
      loss.backward()
      optimizer.step()
  
      loss_after_epoch += loss
      accuracy_after_epoch += get_num_correct(preds, labels)
    print(f'Epoch: {epoch} \t Accuracy: {accuracy_after_epoch} \t Loss: {loss_after_epoch/100}')
    
def validate_model(model, loader):
  mode = model.to(device)
  model.eval()
  num_correct = 0
  with torch.no_grad():
    for images,labels in loader:
      images = images.to(device)
      labels = labels.to(device)
      # print(labels)
      preds = model(images)
      # print(f'preds: {preds.argmin(dim=1)}')
      num_correct += get_num_correct(preds, labels)
      # print(num_correct)
    
    print(f'Predictions accuracy: {num_correct}')


#Hyperparameters and paths
main_dir = 'drive/MyDrive/kaggle/dogs_vs_cats/train'
image_size = 224
rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]
train_split = 0.8
num_workers = multiprocessing.cpu_count()
model_name = 'efficientnet-b0'
training_epochs = 10
learning_rate = 0.0001
model_save_path = 'drive/MyDrive/dogs_vs_cats/models/'
batch_size = 100


transform_list = [transforms.ToPILImage(), transforms.Resize(image_size),
                  transforms.ToTensor(),transforms.Normalize(rgb_mean, rgb_std)]

print('Setting device...')
set_device()
print('Creating training and validation dataloaders...')
train_loader, valid_loader = load_data(main_dir, transformations=transform_list,
                                       batch_size=batch_size, train_split=train_split)
print('Defining model')
model = get_model(model_name, num_of_classes=2, input_channels=3, image_size=image_size)
print('Initializing the optimizer...')
optimizer = adam(model, learning_rate)
print('Training the model...')
trained_model = train_model(train_loader, model, optimizer,
                           num_epochs=training_epochs)
print('Saving the model to disk!')
trained_model.save(model_save_path+model_name+'cats_dogs')
print('validating the model...')
validate_model(trained_model, valid_loader)



# optimizer = adam(model=model, lr=0.0001)

# model = get_model(model_name, num_of_classes=2, input_channels=3, image_size=224)


# def create_dataset(train_dir, transformations, is_test_set=False):
#   return CatsDogsDataset(train_dir, transform = transforms.Compose(transformations), is_test_set=False)
# cats_dogs_dataset = create_dataset(train_data_dir, transform_list)

# def create_train_valid(dataset, train_set_total, valid_set_total):
#   train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_total, valid_set_total])
#   return (train_set, valid_set)

# train_set, valid_set = create_train_valid(cats_dogs_dataset, 20000, 5000)

# def create_dataloader(dataset, batch_size, num_workers, shuffle=True):
#   loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
#   return loader

# train_loader = create_dataloader(train_set, TRAIN_BATCH_SIZE, NUM_WORKERS)
