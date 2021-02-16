import os
from skimage import io
from torch.utils.data import Dataset

class CatsDogsDataset(Dataset):
  def __init__(self, root_dir, transform=None, is_test_set=False):
    self.root_dir = root_dir
    self.transform = transform
    self.is_test_set = is_test_set
    self.data = []
    self.classes = sorted(list(os.listdir(self.root_dir)))
    self.classes = [class_.lower() for class_ in self.classes]
    # for dir in labels_dir:
    #   image_names = os.listdir(os.path.join(self.root_dir, dir))
    #   for image_name in image_names:
    #     self.filenames.append(os.path.join(dir, image_name))
    for root, dirs, files in os.walk(self.root_dir):
      for filename in files:
        self.data.append(os.path.join(root,filename))
 
  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.data[index])
    image = io.imread(img_path)
    # image = PIL.Image.open(img_path)
    class_name = os.path.split(os.path.dirname(img_path))[1]
    label = self.classes.index(class_name.lower())
    
    if self.transform is not None:
      image = self.transform(image)
 
    if self.is_test_set:
      return image
    else:
      return(image, label)
 
  def __len__(self):
    return len(self.data)

  def get_num_classes(self):
    return len(self.classes)