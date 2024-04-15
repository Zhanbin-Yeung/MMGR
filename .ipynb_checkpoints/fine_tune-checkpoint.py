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
       
def train(model, data_loader, optimizer, tokenizer, ids_tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('index_loss_lm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('retrieval_loss_lm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    for i,(contents, summaries, index_docids, retrieval_docids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
            
#         clean_titles = [title.replace('.mp4', '').replace('.pdf', '') for title in titles]
        
#         title_contents = [s1 + s2 for s1, s2 in zip(clean_titles, contents)]
        
        contents_input = tokenizer(contents, padding='longest', max_length=30, truncation=True, return_tensors="pt").to(device)
        summaries_input = tokenizer(summaries, padding='longest', max_length=30, truncation=True, return_tensors="pt").to(device)
        
        index_ids_input = ids_tokenizer(index_docids, padding='longest', max_length=10, return_tensors="pt").to(device)
        retrieval_ids_input = ids_tokenizer(retrieval_docids, padding='longest', max_length=10, return_tensors="pt").to(device)
        

        index_loss_lm, retrieval_loss_lm = model( content=contents_input,summary=summaries_input,
                                                  index_id=index_ids_input,
                                                  retrieval_id=retrieval_ids_input,
                                                  alpha=alpha) 

        loss = index_loss_lm + retrieval_loss_lm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        # metric_logger.update(loss_itm=loss_itm.item())
        # metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(index_loss_lm=index_loss_lm.item())
        metric_logger.update(retrieval_loss_lm=retrieval_loss_lm.item())
        metric_logger.update(loss_lm=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, ids_tokenizer, device, config, k=10, max_batches=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation: '
    print_freq = 50

    bos_token_id = ids_tokenizer.convert_tokens_to_ids(config['bos'])
    eos_token_id = ids_tokenizer.convert_tokens_to_ids(config['eos'])
    pad_token_id = ids_tokenizer.convert_tokens_to_ids(config['pad'])
    retrieval_token_id = ids_tokenizer.convert_tokens_to_ids('retrieval')
    special_token_id = {}
    special_token_id['BOS'] = bos_token_id
    special_token_id['EOS'] = eos_token_id
    special_token_id['PAD'] = pad_token_id

    txt2img_recall_at_k = []
    if max_batches is None:  
        max_batches = len(data_loader)  
    for idx, (_, summaries, _, retrieval_docids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if idx >= max_batches:  
            break
        
        # clean_titles = [title.replace('.mp4', '').replace('.pdf', '') for title in titles]

        summaries_input = tokenizer(summaries, padding='longest', max_length=30, return_tensors="pt").to(device)
        
        t2i_ids = torch.full((summaries_input.input_ids.size(0), 1), retrieval_token_id, dtype=torch.long).to(device)

        t2i_outputs = model(summary=summaries_input, retrieval_id=t2i_ids, train=False, k=k, special_token=special_token_id)

#         titles_input = tokenizer(clean_titles, padding='longest', max_length=30, return_tensors="pt").to(device)
        
#         t2i_ids = torch.full((titles_input.input_ids.size(0), 1), retrieval_token_id, dtype=torch.long).to(device)

#         t2i_outputs = model(summary=titles_input, retrieval_id=t2i_ids, train=False, k=k, special_token=special_token_id)
        
        
        for i, retrieval_docid in enumerate(retrieval_docids):
            st = i * k
            ed = st + k
            t2i_output = t2i_outputs[st: ed, :]
            pred_ids = []
            
            for output in t2i_output:
                pred_id = ids_tokenizer.decode(output, skip_special_tokens=True)
                pred_id = pred_id.split()
                pred_id = " ".join(pred_id[1:]) 
                pred_ids.append(pred_id)
                
            retrieval_docid = retrieval_docid.split()
            retrieval_docid = " ".join(retrieval_docid[1:-1])
            if retrieval_docid in pred_ids:
                txt2img_recall_at_k.append(1)
            else:
                txt2img_recall_at_k.append(0)

            # print('Current idx:', i , 'doc_ids:', retrieval_docid)
            # print('Current idx:', i , 'pred_ids:', pred_ids)
            
    txt2img_recall_at_k = statistics.mean(txt2img_recall_at_k)

    return txt2img_recall_at_k


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
    train_dataset = create_dataset('fine-tune', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank)
    else:
        samplers = None

    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            sampler=samplers[0],
            # shuffle=True,
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
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

        
    print("Start Fine-tuning ")
    for epoch in range(0, max_epoch):
        start_time = time.time()    
        
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, ids_tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  

            if epoch % 3 == 0 :
                txt2img_recall_at_k = evaluation(model, train_loader, tokenizer, ids_tokenizer, device, config, k=10, max_batches=100)
                print('Eval Epoch: ', epoch, 't2i:', txt2img_recall_at_k)
            if epoch == max_epoch - 1:
                txt2img_recall_at_k = evaluation(model, train_loader, tokenizer, ids_tokenizer, device, config, k=10)
                print('Eval Epoch: ', epoch, 't2i:', txt2img_recall_at_k)

        if args.evaluate: 
            break

        if utils.is_main_process():               
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                         
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'fine_tune_checkpoint_title_%02d.pth'%epoch))  
            # torch.save(save_obj, os.path.join(args.output_dir, 'fine_tune_checkpoint_%02d.pth'%epoch))  

           
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()     
        torch.cuda.empty_cache()
        
    # start_time = time.time()    
    # gen_id(model, tokenizer, ids_tokenizer, train_loader, device, config)  
    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Generating DocID time {}'.format(total_time_str)) 

 
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
