import json
import os
import random

from torch.utils.data import Dataset

from dataset.utils import pre_caption

class genid_dataset(Dataset):
    def __init__(self, content, summary, max_words=30):
        self.all_content = json.load(open(content, 'r'))
        self.all_summary= json.load(open(summary, 'r'))
        self.max_words = max_words
                               
    def __len__(self):
        return len(self.all_content)
    
    def __getitem__(self, index):
        content = pre_caption(self.all_content[index], self.max_words)
        summary = pre_caption(self.all_summary[index], self.max_words)

        return content, summary
    
