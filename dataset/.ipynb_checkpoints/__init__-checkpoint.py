import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset, index_dataset
from dataset.dataset_extr_fea import extract_train_dataset, extract_flickr_dataset
from dataset.fine_tune_dataset import fine_tune_dataset
from dataset.gen_id_dataset import genid_dataset
from dataset.randaugment import RandomAugment

def create_dataset(dataset, config):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    if dataset=='pretrain':
        dataset = pretrain_dataset(config['train_file'], pretrain_transform)                  
        return dataset      
               
    elif dataset=='re':    
        index_datasets = index_dataset(config['index_file'], train_transform, config['image_root'], config['img_caption_ids'])
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'], config['train_img_caption_ids'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'], config['val_img_caption_ids'])  
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'], config['test_img_caption_ids'])                
        return index_datasets, train_dataset, val_dataset, test_dataset   
     
    elif dataset=='extract':
        train_dataset = extract_flickr_dataset('Flicker/flickr_annotations_30k.csv', train_transform, './Flicker/flickr30k-images')
        # train_dataset = extract_train_dataset(config['index_file'], train_transform, config['image_root'])
        return train_dataset
    
    elif dataset=='fine-tune':
        fine_tune_datasets = fine_tune_dataset(config['content'], config['summary'], config['docid'])
        return fine_tune_datasets
    
    elif dataset=='generate':
        gen_id_dataset = genid_dataset(config['content'], config['summary'])
        return gen_id_dataset
    
def retrieval_collate_fn(batch):
    image_list, captions_list, img_id_list, caption_ids_list = [], [], [], []
    for image, captions, img_id, caption_ids in batch:
        image_list.append(image)
        captions_list.extend(captions)
        img_id_list.extend(img_id)
        caption_ids_list.append(caption_ids)
    return torch.stack(image_list,dim=0), captions_list, img_id_list, caption_ids_list


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
