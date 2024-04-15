import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, path_to_ids, eos='[SEP]', max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        with open(path_to_ids, 'r') as docids:
            self.docids = json.load(docids)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {} 
        self.eos = eos
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])
     
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words) 
        

        img_id = self.docids[index][0]
        # img_id += self.eos
        
        caption_id = self.docids[index][1]
        # caption_id += self.eos

        return image, caption, img_id, caption_id
    
    
class index_dataset(Dataset):
    def __init__(self, index_file, transform, image_root, path_to_ids, eos='[SEP]', max_words=30):
        self.ann = {}
        self.image_list = []
        self.caption_list = []
        train_len = 0
        for f in index_file:
            if 'train' in f:
                self.ann['train'] = json.load(open(f, 'r'))
                for ann in self.ann['train']:
                    image = ann['image']
                    caption = ann['caption']
                    self.image_list.append(image)
                    self.caption_list.append(caption)
                train_len = len(self.image_list)
            elif 'val' in f:
                self.ann['val'] = json.load(open(f, 'r'))
                for ann in self.ann['val']:
                    images = []
                    images.append(ann['image'])
                    captions = ann['caption']
                    images = images * len(captions)
                    self.image_list.extend(images)
                    self.caption_list.extend(captions)
            elif 'test' in f:
                self.ann['test'] = json.load(open(f, 'r'))
                for ann in self.ann['test']:
                    images = []
                    images.append(ann['image'])
                    captions = ann['caption']
                    images = images * len(captions)
                    self.image_list.extend(images)
                    self.caption_list.extend(captions)

        with open(path_to_ids, 'r') as docids:
            # tmp = json.load(docids)
            # self.docids = tmp[:train_len]
            self.docids = json.load(docids)
        for i, docid in enumerate(self.docids[train_len :], start=train_len):
            self.docids[i][0] = 'index ' + docid[0]
            self.docids[i][1] = 'index ' + docid[1]
        # for i, docid in enumerate(self.docids):
        #     self.docids[i][0] = 'retrieval ' + docid[0]
        #     self.docids[i][1] = 'retrieval ' + docid[1]
            
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        self.eos = eos
    
    def __len__(self):
        return len(self.docids)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image_list[index])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)

        caption = pre_caption(self.caption_list[index], self.max_words)

        img_id = self.docids[index][0]
        img_id += self.eos

        caption_id = self.docids[index][1]
        caption_id += self.eos

        return image, caption, img_id, caption_id
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, path_to_ids, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        with open(path_to_ids, 'r') as docids:
            self.docids = json.load(docids)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
                             
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        img_id = []
        img_id.append(self.docids[index][0])
        caption_ids = self.docids[index][1]
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  
        
        captions = []
        for caption in self.ann[index]['caption']:
            captions.append(pre_caption(caption, self.max_words))
        img_id = img_id * len(captions)
        
        return image, captions, img_id, caption_ids
      
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, caption
            
