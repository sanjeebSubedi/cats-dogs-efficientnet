import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL
import torch.optim as optim
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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=vars.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=True, num_workers=vars.num_workers)
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
      optimizer.zero_grad()
  
      preds = model(images)
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
      preds = model(images)
      num_correct += get_num_correct(preds, labels)
    
    print(f'Predictions accuracy: {num_correct}')

transform_list = [transforms.ToPILImage(), transforms.Resize(vars.image_size),
                  transforms.ToTensor(),transforms.Normalize(vars.rgb_mean, vars.rgb_std)]

print('Setting device...')
set_device()
print('Creating training and validation dataloaders...')
train_loader, valid_loader = load_data(vars.main_dir, transformations=transform_list,
                                       batch_size=vars.batch_size, train_split=vars.train_split)
print('Defining model')
model = get_model(vars.model_name, num_of_classes=2, input_channels=3, image_size=vars.image_size)
print('Initializing the optimizer...')
optimizer = adam(model, vars.learning_rate)
print('Training the model...')
trained_model = train_model(train_loader, model, optimizer,
                           num_epochs=vars.training_epochs)
print('Saving the model...')
torch.save(trained_model, vars.model_save_path)
print('validating the model...')
validate_model(trained_model, valid_loader)