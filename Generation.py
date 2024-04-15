import argparse
import os
import re
from ruamel import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import statistics
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.ALBEF2 import ALBEF2
from models.image_encoder import interpolate_pos_embed
from models.tokenizer import BertTokenizer
from models.dec_tokenizer import Dec_Tokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, retrieval_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer

def find_pair(list1, list2):
    # Convert lists to sets for finding matches or similarities
    set1 = set(list1)
    set2 = set(list2)

    # Find matching pairs
    matching_pairs = set1.intersection(set2)

    # If there are matching pairs, return the first match
    if matching_pairs:
        return list(matching_pairs)[0], "matching"

    # If there are no matching pairs, find the most similar pair
    max_overlap = -1
    most_similar_pair = (None, None)

    for item1 in list1:
        for item2 in list2:
            # Calculate overlap as the number of common characters
            overlap = len(set(item1).intersection(item2))
            if overlap > max_overlap:
                max_overlap = overlap
                most_similar_pair = (item1, item2)

    return most_similar_pair, "most similar"


@torch.no_grad()
def gen_id(model, tokenizer, ids_tokenizer, data_loader, device, config, k=10):
    model.eval()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate: '
    print_freq = 5
    index_list = []
    contents_list = []
    
    bos_token_id = ids_tokenizer.convert_tokens_to_ids(config['bos'])
    eos_token_id = ids_tokenizer.convert_tokens_to_ids(config['eos'])
    pad_token_id = ids_tokenizer.convert_tokens_to_ids(config['pad'])
    retrieval_token_id = ids_tokenizer.convert_tokens_to_ids('retrieval')
    special_token_id = {}
    special_token_id['BOS'] = bos_token_id
    special_token_id['EOS'] = eos_token_id
    special_token_id['PAD'] = pad_token_id
    
    for i, (contents, summaries) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        for idx, content in enumerate(contents):
            contents[idx] = ''.join(re.findall('[A-Za-z .!?]+', content))
        content_input = tokenizer(contents, padding=True, truncation=True, max_length=30, return_tensors="pt").to(device)
        summary_input = tokenizer(summaries, padding=True, truncation=True, max_length=30, return_tensors="pt").to(device)
        
        content_ids = torch.full((content_input.input_ids.size(0), 1), retrieval_token_id, dtype=torch.long).to(device)
        summary_ids = torch.full((summary_input.input_ids.size(0), 1), retrieval_token_id, dtype=torch.long).to(device)
        
        summary_outputs = model(summary=summary_input, retrieval_id=summary_ids, train=False, k=k, special_token=special_token_id)
        content_outputs = model(summary=content_input, retrieval_id=content_ids, train=False, k=k, special_token=special_token_id)
        
        for ii in range(len(contents)):
            st = ii * k
            ed = st + k
            summary_output = summary_outputs[st: ed, :]
            content_output = content_outputs[st: ed, :]
            
            content_preds = []
            summary_preds = []
            for content, summary in zip(content_output, summary_output):
                content_id = ids_tokenizer.decode(content, skip_special_tokens=True)
                content_id = content_id.split()
                content_id = " ".join(content_id[1:-1]) 
                content_preds.append(content_id)
                
                summary_id = ids_tokenizer.decode(summary, skip_special_tokens=True)
                summary_id = summary_id.split()
                summary_id = " ".join(summary_id[1:-1])
                summary_preds.append(summary_id)
            
            pair, pair_type = find_pair(summary_preds, content_preds)
            if pair_type == "most similar":
                _, content_docid = pair
            else:
                content_docid = pair
            contents_list.append(content_docid)
            
    with open('contents_docid.json', 'w') as f:
        json.dump(contents_list, f)
        

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    train_dataset = create_dataset('generate', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank)
    else:
        samplers = None

    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=1536,
            num_workers=4,
            pin_memory=True,
            # sampler=samplers[0],
            shuffle=False,
            drop_last=False,
        )         
    
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    ids_tokenizer = Dec_Tokenizer.from_pretrained(args.ids_tokenizer)
    
    #### Model #### 
    print("Creating model")
    model = ALBEF2(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, ids_tokenizer=ids_tokenizer, tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
              
        msg = model.load_state_dict(state_dict,strict=False)  
        
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  
        
    
    model = model.to(device)   
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    print("Start Generating DocID ")
    
    start_time = time.time()    
    gen_id(model, tokenizer, ids_tokenizer, train_loader, device, config)  
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Generating DocID time {}'.format(total_time_str)) 

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Retrieval_coco.yaml')
    parser.add_argument('--output_dir', default='./output/Retrieval_coco')        
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--ids_tokenizer', default='./data/id_vocab.txt')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
