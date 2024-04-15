import json
import os
import random

import pandas as pd
import ast
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
        image_caption = dict()
        row = 0
        for ann in self.ann['train']:
            image = ann['image']
            if image not in image_caption.keys():
                image_caption[image] = [row]
                self.image_list.append(image)
            else:
                image_caption[image].append(row)
            row += 1  
        
        for ann in self.ann['val']:
            self.image_list.append(ann['image'])
            
        for ann in self.ann['test']:
            self.image_list.append(ann['image'])
            
        self.transform = transform
        self.image_root = image_root

    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image_list[index])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)

        return index, image
    
    
class extract_flickr_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root):
        data = pd.read_csv(ann_file)   
        self.image_list = [image for image in data['filename']]
        
        self.transform = transform
        self.image_root = image_root

    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image_list[index])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)

        return index, image
    
