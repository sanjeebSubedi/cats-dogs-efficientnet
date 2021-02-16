import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# import PIL
import torch.optim as optim
import os
import numpy as np
import time
import math
from customDataset import CatsDogsDataset
import runtimeVariables as vars
from efficientNetCustom import EfficientNetCustom

def set_device():
  global device
  device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
  print(f'Device set to {device}')

def load_data(main_dir, transformations, train_split, batch_size=32,  is_test_set=False, num_workers=0):
    dataset = CatsDogsDataset(main_dir, transform = transformations, is_test_set=is_test_set)
    global num_classes, train_size, valid_size, dataset_size
    num_classes = dataset.get_num_classes()
    dataset_size = len(dataset)
    train_size = math.ceil(train_split*len(dataset))
    valid_size = len(dataset)-train_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, valid_loader    

# def get_model(model_name, num_classes, input_channels, image_size):
#   if load_pretrained_weights:
#     return EfficientNet.from_pretrained(model_name, input_channels, 
#                                         num_classes=num_classes,
#                                         image_size=image_size)
#   else:
#     return EfficientNet.from_name(model_name, input_channels,
#                                   num_classes=num_classes,
#                                   image_size=image_size)
  
def adam(model, lr):
  return optim.Adam(model.parameters(), lr=lr)

def get_num_correct(preds, labels):
  return preds.argmax(dim=1).eq(labels).sum().item()

def train_model(train_loader, model, optimizer, num_epochs, valid_loader=None):
  model = model.to(device)
  model.train()
  for epoch in range(1, num_epochs+1):
    start_time = time.time()
    loss_after_epoch, acc_after_epoch = 0, 0
    # images, labels = next(iter(train_loader))
    for images, labels in train_loader:
      torch.cuda.empty_cache()
      images = images.to(device)
      labels = labels.to(device)
      preds = model(images)
      loss = F.cross_entropy(preds, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      loss_after_epoch += loss
      acc_after_epoch += get_num_correct(preds, labels)

    print(f'Epoch:{epoch}/{num_epochs}  Acc:{(acc_after_epoch/train_size):.5f}', end = '  ')
    print(f'Loss:{(loss_after_epoch/100):.5f}  Duration:{(time.time()-start_time):.2f}s', end='\n' if valid_loader is None else '  ')
    if valid_loader is not None:
      validate_model(model, valid_loader)
    if epoch % vars.checkpoint_save_frequency == 0:
      checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  'epoch': epoch, 'loss': loss.item()}
      save_checkpoint(checkpoint, vars.checkpoint_dir, vars.checkpoint_filename)

    
def validate_model(model, loader):
  model = model.to(device)
  model.eval()
  num_correct, val_loss = 0, 0
  with torch.no_grad():
    for images,labels in loader:
      images, labels = images.to(device), labels.to(device)
      preds = model(images)
      val_loss += F.cross_entropy(preds, labels)
      num_correct += get_num_correct(preds, labels)
    
    print(f'Val_acc:{(num_correct/valid_size):.5f}  Val_loss:{(val_loss/100):.5f}')

def save_checkpoint(state, save_path, filename):
  print('Saving checkpoint...')
  save_path = os.path.join(save_path, filename)
  torch.save(state, save_path)

def get_checkpoint(checkpoint_path):
  print('Loading checkpoint...')
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  # epoch = checkpoint['epoch']
  # loss = checkpoint['loss']
  return model, optimizer#, epoch, loss


transform_list = [transforms.ToPILImage(), transforms.Resize(vars.image_size),
                  transforms.ToTensor(), transforms.Normalize(vars.rgb_mean, vars.rgb_std)]

print('Setting device...')
set_device()
print('Creating training and validation dataloaders...')
train_loader, valid_loader = load_data(vars.main_dir, transformations=transforms.Compose(transform_list),
                                       batch_size=vars.batch_size, train_split=vars.train_split, num_workers=vars.num_workers)
print(f'Total of {num_classes} classes were found in the dataset!')
print('Defining model')
print('Initializing the optimizer...')
print('Training the model...')
if vars.load_checkpoint:
  model, optimizer = get_checkpoint(vars.checkpoint_path)
else:
  model = EfficientNetCustom(vars.model_name, in_channels=3, num_classes=num_classes,
                             train_only_last_layer=vars.train_only_last_layer,
                             load_pretrained_weights=vars.load_pretrained_weights)
  # model = get_model(model_name, num_classes, input_channels=3, image_size=image_size)
  optimizer = adam(model, vars.learning_rate)

trained_model = train_model(train_loader, model, optimizer,
                            num_epochs=vars.training_epochs, valid_loader=valid_loader)