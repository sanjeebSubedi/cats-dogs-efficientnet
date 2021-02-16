import multiprocessing
from efficientnet_pytorch import EfficientNet
import os

num_workers = multiprocessing.cpu_count()
main_dir = '/content/drive/MyDrive/local_leaves/train'
model_name = 'efficientnet-b0'
image_size = EfficientNet.get_image_size(model_name)
rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]
train_split = 0.8     # percentage of images to be used for training
training_epochs = 10  # number of epochs to train
learning_rate = 0.0001  # learning rate for the optimizer
checkpoint_dir = '/content/drive/MyDrive/local_leaves/models' # directory to save the checkpoint to
batch_size = 32
checkpoint_save_frequency = 2   # number of epochs to be trained brfore saving checkpoint
checkpoint_filename = model_name + '.pth.tar'  # filename for checkpoint
load_pretrained_weights = True  # HP to load pretrained imagenet weights
train_only_last_layer = True    # HP to specify whether to train whole network or last layer only
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename) # full path of checkpoint
load_checkpoint = False   # option to load saved checkpoint