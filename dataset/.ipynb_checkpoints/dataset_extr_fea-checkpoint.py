import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class extract_train_dataset(Dataset):
    def __init__(self, index_file, transform, image_root):
        self.ann = {}
        for f in index_file:
            if 'train' in f:
                self.ann['train'] = json.load(open(f, 'r'))
            elif 'val' in f:
                self.ann['val'] = json.load(open(f, 'r'))
            elif 'test' in f:
                self.ann['test'] = json.load(open(f, 'r'))
        self.image_list = []
        for ann in self.ann['train']:
            image = ann['image']
            self.image_list.append(image)   
        train_len = len(self.image_list)
        for ann in self.ann['val']:
            images = []
            images.append(ann['image'])
            captions = ann['caption']
            images = images * len(captions)
            self.image_list.extend(images)
            
        for ann in self.ann['test']:
            images = []
            images.append(ann['image'])
            captions = ann['caption']
            images = images * len(captions)
            self.image_list.extend(images)
            
        self.transform = transform
        self.image_root = image_root

    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image_list[index])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)

        return index, image
    
    
    
