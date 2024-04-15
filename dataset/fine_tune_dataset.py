import json
import os
import random

from torch.utils.data import Dataset

from dataset.utils import pre_caption

class fine_tune_dataset(Dataset):
    def __init__(self, content, summary, docid, eos=' [SEP]', max_words=30):
        self.all_content = json.load(open(content, 'r'))
        self.all_summary= json.load(open(summary, 'r'))
        self.docid = json.load(open(docid, 'r'))
        self.max_words = max_words
        self.eos = eos
                               
    def __len__(self):
        return len(self.all_content)
    
    def __getitem__(self, index):
        content = pre_caption(self.all_content[index], self.max_words)
        summary = pre_caption(self.all_summary[index], self.max_words)
        docid = self.docid[index]
        index_docid = 'index ' + docid + self.eos
        retrieval_docid = 'retrieval ' + docid + self.eos
        return content, summary, index_docid, retrieval_docid
    
    
    
