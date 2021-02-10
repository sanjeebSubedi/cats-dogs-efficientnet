import os
from skimage import io 
from torch.utils.data import Dataset

class CatsDogsDataset(Dataset):
  def __init__(self, root_dir, transform=None, is_test_set=False):
    self.root_dir = root_dir
    self.transform = transform
    self.is_test_set = is_test_set
    labels_dir = os.listdir(self.root_dir)
    self.filenames = []
    for dir in labels_dir:
      image_names = os.listdir(os.path.join(self.root_dir, dir))
      for image_name in image_names:
        self.filenames.append(os.path.join(dir, image_name))
 
  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.filenames[index])
    image = io.imread(img_path)
    # image = PIL.Image.open(img_path)
    label = 0 if 'dog' in self.filenames[index] else 1
    
    if self.transform is not None:
      image = self.transform(image)
 
    if self.is_test_set:
      return image
    else:
      return(image, label)
 
  def __len__(self):
    return len(self.filenames)