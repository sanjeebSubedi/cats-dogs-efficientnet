import multiprocessing
num_workers = multiprocessing.cpu_count()
main_dir = 'drive/MyDrive/kaggle/dogs_vs_cats/train'
image_size = 224
rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]
train_split = 0.8
model_name = 'efficientnet-b0'
training_epochs = 10
learning_rate = 0.0001
model_save_path = 'drive/MyDrive/dogs_vs_cats/models/'
batch_size = 100