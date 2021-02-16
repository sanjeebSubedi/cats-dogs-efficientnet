import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetCustom(nn.Module):
  def __init__(self, model_name, in_channels, num_classes,
               load_pretrained_weights=True, train_only_last_layer=False):
    super(EfficientNetCustom, self).__init__()
    self.model_name = model_name
    self.in_channels = in_channels
    self.num_classes = num_classes
    # self.image_size = EfficientNet.get_image_size(self.model_name)
    self.load_pretrained_weights = load_pretrained_weights
    self.train_only_last_layer = train_only_last_layer
    
    if self.load_pretrained_weights:
      self.features = EfficientNet.from_pretrained(self.model_name, in_channels=self.in_channels)
    else:
      self.features = EfficientNet.from_name(self.model_name, in_channels=self.in_channels)

    if self.train_only_last_layer:
      print('Training only last layer...')
      for param in self.features.parameters():
        param.requires_grad = False
    
    in_ftrs = self.features._fc.in_features
    self.features._fc = nn.Linear(in_ftrs, self.num_classes)
    # self.features._fc.requires_grad = True

  def forward(self, inputs):
    x = self.features(inputs)
    return x