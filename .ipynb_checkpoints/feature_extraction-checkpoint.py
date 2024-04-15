import argparse
import os
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

from models.extract_feature import ALBEF
from models.image_encoder import interpolate_pos_embed
from models.tokenizer import BertTokenizer

import utils

from dataset import create_dataset, create_sampler, create_loader, retrieval_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer

@torch.no_grad()
def extract_feature(model, data_loader, device, config):
    model.eval()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Extract: '
    print_freq = 50
    index_list = []
    image_feats_list = []
    k = 0
    for i, (index, image) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)    
        image_feats = model(image) 
        # image_feats  = image_feats.cpu() 
        image_feats_list.append(image_feats)
        index_list.extend(index)
        
        if i % 32 == 0 and i != 0:
            img_feats = torch.cat(image_feats_list, dim=0)
            img_feats = img_feats.to(device)
            index_tensor = torch.tensor(index_list, dtype=torch.long, device=device)
            
            if dist.get_rank() == 0:
                gathered_feats = [torch.zeros_like(img_feats, device=device) for _ in range(dist.get_world_size())]
                gathered_indices = [torch.zeros_like(index_tensor, device=device) for _ in range(dist.get_world_size())]
            else:
                gathered_feats = gathered_indices = None
                
            dist.gather(img_feats, gather_list=gathered_feats, dst=0)
            dist.gather(index_tensor, gather_list=gathered_indices, dst=0)

            if dist.get_rank() == 0:
                final_feats = torch.cat(gathered_feats, dim=0)
                torch.save(final_feats, f'./image_embeds/img_embeds_{k}.pt')
                final_indices = torch.cat(gathered_indices, dim=0)
                torch.save(final_indices, f'./image_embeds/index_{k}.pt')
                k += 1
                del final_feats
                del final_indices
            
            del img_feats
            del index_tensor
            del gathered_feats
            del gathered_indices
            
            image_feats_list = []
            index_list = []
            
    img_feats = torch.cat(image_feats_list, dim=0)
    img_feats = img_feats.to(device)
    index_tensor = torch.tensor(index_list, dtype=torch.long, device=device)

    if dist.get_rank() == 0:
        gathered_feats = [torch.zeros_like(img_feats, device=device) for _ in range(dist.get_world_size())]
        gathered_indices = [torch.zeros_like(index_tensor, device=device) for _ in range(dist.get_world_size())]
    else:
        gathered_feats = gathered_indices = None

    dist.gather(img_feats, gather_list=gathered_feats, dst=0)
    dist.gather(index_tensor, gather_list=gathered_indices, dst=0)

    if dist.get_rank() == 0:
        final_feats = torch.cat(gathered_feats, dim=0)
        torch.save(final_feats, f'./image_embeds/img_embeds_{k}.pt')
        final_indices = torch.cat(gathered_indices, dim=0)
        torch.save(final_indices, f'./image_embeds/index_{k}.pt')


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
    train_dataset = create_dataset('extract', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [False], num_tasks, global_rank)
    else:
        samplers = None

    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=256,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            sampler=samplers[0],
            drop_last=False,
        )         

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        # for key in list(state_dict.keys()):
        #     if 'bert' in key:
        #         encoder_key = key.replace('bert.','')         
        #         state_dict[encoder_key] = state_dict[key] 
        #         del state_dict[key]                
        msg = model.load_state_dict(state_dict,strict=False)  
        
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  
        
    
    model = model.to(device)   

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        
    print("Start Extracting Feature")
    start_time = time.time()    

    extract_feature(model, train_loader, device, config)  

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Extracting Feature time {}'.format(total_time_str)) 

 
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
