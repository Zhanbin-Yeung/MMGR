import argparse
import os
from ruamel import yaml
import numpy as np
import random
import math
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

from models.ALBEF import ALBEF
from models.image_encoder import interpolate_pos_embed
from models.tokenizer import BertTokenizer
from models.dec_tokenizer import Dec_Tokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, retrieval_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer

def sample(images, texts, img_ids, caption_ids, ratio=1):
    re_idx = []
    for i, img_id in enumerate(img_ids):
        if not img_id.startswith('index'):
            re_idx.append(i)
            
    index_img_ids = [img_id if img_id.startswith('index') else 'index ' + img_id for img_id in img_ids]
    index_caption_ids = [caption_id if caption_id.startswith('index') else 'index ' + caption_id for caption_id in caption_ids]
     
    index = {
        "imgs": images,
        "txts": texts,
        "img_ids": index_img_ids,
        "cap_ids": index_caption_ids
    }
    if len(re_idx) == 0:
        retrieval = {"imgs": None, "txts": None, "img_ids": None, "cap_ids": None } 
    else:
        num_re = min(len(re_idx), math.ceil(len(texts) * ratio))
        re_idx = random.sample(re_idx, num_re)
        re_texts = [texts[i] for i in re_idx]
        re_img_ids = ['retrieval ' + img_ids[i] for i in re_idx]
        re_caption_ids = ['retrieval ' + caption_ids[i] for i in re_idx]
        retrieval = {
            "imgs": images[re_idx],
            "txts": re_texts,
            "img_ids": re_img_ids,
            "cap_ids": re_caption_ids
        }

    return index, retrieval

def train(model, data_loader, optimizer, tokenizer, ids_tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('index_loss_lm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('retrieval_loss_lm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    for i,(image, text, img_id, caption_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
            
        index, retrieval = sample(image, text, img_id, caption_id)
        index_images = index['imgs'].to(device,non_blocking=True) 
        index_text_input = tokenizer(index['txts'], padding='longest', max_length=30, return_tensors="pt").to(device)
        index_img_id_input = ids_tokenizer(index['img_ids'], padding='longest', max_length=10, return_tensors="pt").to(device)
        index_caption_id_input = ids_tokenizer(index['cap_ids'], padding='longest', max_length=10, return_tensors="pt").to(device)
        
        if retrieval['imgs'] is not None:
            retrieval_images = retrieval['imgs'].to(device,non_blocking=True) 
            retrieval_text_input = tokenizer(retrieval['txts'], padding='longest', max_length=30, return_tensors="pt").to(device)
            retrieval_img_id_input = ids_tokenizer(retrieval['img_ids'], padding='longest', max_length=10, return_tensors="pt").to(device)
            retrieval_caption_id_input = ids_tokenizer(retrieval['cap_ids'], padding='longest', max_length=10, return_tensors="pt").to(device)

            index_loss_lm, retrieval_loss_lm = model( retrieval_images, 
                                                      retrieval_text_input, 
                                                      index_images, 
                                                      index_text_input, 
                                                      index_image_id=index_img_id_input, 
                                                      index_text_id=index_caption_id_input,
                                                      retrieval_image_id=retrieval_img_id_input,
                                                      retrieval_text_id=retrieval_caption_id_input,
                                                      alpha=alpha) 
        else : 
            index_loss_lm, retrieval_loss_lm = model( index_image=index_images, 
                                                      index_text=index_text_input, 
                                                      index_image_id=index_img_id_input, 
                                                      index_text_id=index_caption_id_input,
                                                      alpha=alpha) 
#         images = image.to(device,non_blocking=True) 
#         text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)
#         img_id_input = ids_tokenizer(img_id, padding='longest', max_length=10, return_tensors="pt").to(device)
#         caption_id_input = ids_tokenizer(caption_id, padding='longest', max_length=10, return_tensors="pt").to(device)
        
#         index_loss_lm, retrieval_loss_lm = model( images, 
#                                                   text_input, 
#                                                   retrieval_image_id=img_id_input,
#                                                   retrieval_text_id=caption_id_input,
#                                                   alpha=alpha) 
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

    img2txt_recall_at_k = []
    txt2img_recall_at_k = []
    if max_batches is None:  
        max_batches = len(data_loader)  
    for idx, (images, captions, image_ids, caption_ids_list) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if idx >= max_batches:  
            break
            
        images = images.to(device, non_blocking=True) 
        captions_input = tokenizer(captions, padding='longest', max_length=30, return_tensors="pt").to(device)
        
        i2t_ids = torch.full((images.size(0), 1), retrieval_token_id, dtype=torch.long).to(device)
        t2i_ids = torch.full((captions_input.input_ids.size(0), 1), retrieval_token_id, dtype=torch.long).to(device)

        i2t_outputs, t2i_outputs = model(images, captions_input, retrieval_image_id=i2t_ids, retrieval_text_id=t2i_ids, train=False, k=k, special_token=special_token_id)

        for i, caption_ids in enumerate(caption_ids_list):
            st = i * k
            ed = st + k
            i2t_output = i2t_outputs[st: ed, :]
            pred_ids = []
            
            for output in i2t_output:
                pred_id = ids_tokenizer.decode(output, skip_special_tokens=True)
                pred_id = pred_id.split()
                pred_id = " ".join(pred_id[1:]) 
                pred_ids.append(pred_id)
            
            if type(caption_ids) != list:
                caption_ids = [caption_ids]
            for txt_id in caption_ids:
                if txt_id in pred_ids:
                    img2txt_recall_at_k.append(1)
                else:
                    img2txt_recall_at_k.append(0)
                # print('Current idx:', i , 'caption_ids:', tmp)
                # print('Current idx:', i , 'caption_pred_ids:', pred_ids)

        for i, image_id in enumerate(image_ids):
            st = i * k
            ed = st + k
            t2i_output = t2i_outputs[st: ed, :]
            pred_ids = []
            
            for output in t2i_output:
                pred_id = ids_tokenizer.decode(output, skip_special_tokens=True)
                pred_id = pred_id.split()
                pred_id = " ".join(pred_id[1:]) 
                pred_ids.append(pred_id)
                
            if image_id in pred_ids:
                txt2img_recall_at_k.append(1)
            else:
                txt2img_recall_at_k.append(0)

            # print('Current idx:', i , 'imgs_ids:', image_id)
            # print('Current idx:', i , 'imgs_pred_ids:', pred_ids)
            
    img2txt_recall_at_k = statistics.mean(img2txt_recall_at_k)
    txt2img_recall_at_k = statistics.mean(txt2img_recall_at_k)

    return img2txt_recall_at_k, txt2img_recall_at_k

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
    print("Creating retrieval dataset")
    index_dataset, train_dataset, val_dataset, test_dataset = create_dataset('re', config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([index_dataset, train_dataset], [True, True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None, None]
    
    index_loader, train_loader, val_loader, test_loader = create_loader([index_dataset, train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*3,
                                                          num_workers=[4,4,4,4],
                                                          is_trains=[True, True, False, False], 
                                                          collate_fns=[None,None,retrieval_collate_fn, retrieval_collate_fn])   
       
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    ids_tokenizer = Dec_Tokenizer.from_pretrained(args.ids_tokenizer)
    
    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, ids_tokenizer=ids_tokenizer, tokenizer=tokenizer)

    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        # state_dict = checkpoint
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        # m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        # state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        
        # for key in list(state_dict.keys()):
        #     if 'bert' in key:
        #         encoder_key = key.replace('bert.','')         
        #         state_dict[encoder_key] = state_dict[key] 
        #         del state_dict[key]                
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


    print("Start training")
    for epoch in range(0, max_epoch):
        start_time = time.time()    
        
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, index_loader, optimizer, tokenizer, ids_tokenizer, epoch, warmup_steps, device, lr_scheduler, config)

            if epoch % 5 == 0:
                train_i2t_rk, train_t2i_rk = evaluation(model, train_loader, tokenizer, ids_tokenizer, device, config, k=10, max_batches=51)
                print('Epoch: ', epoch, 'i2t: ', train_i2t_rk, 't2i:', train_t2i_rk)
                img2txt_recall_at_k, txt2img_recall_at_k = evaluation(model, val_loader, tokenizer, ids_tokenizer, device, config, k=10)
                print('Eval Epoch: ', epoch, 'i2t: ', img2txt_recall_at_k, 't2i:', txt2img_recall_at_k)
                
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
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  

           
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()     
        torch.cuda.empty_cache()

    img2txt_recall_at_k, txt2img_recall_at_k = evaluation(model, val_loader, tokenizer, ids_tokenizer, device, config, k=10)
    print(img2txt_recall_at_k, txt2img_recall_at_k)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='./output/Retrieval_flickr')        
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
