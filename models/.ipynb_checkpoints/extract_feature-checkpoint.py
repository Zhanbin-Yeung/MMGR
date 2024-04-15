from functools import partial
from models.image_encoder import VisionTransformer
from models.text_encoder import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

class ALBEF(nn.Module):
    def __init__(self,                 
                 config = None,     
                 ):
        super().__init__()
    
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

  
    def forward(self, image):
        with torch.no_grad():
            image_embeds = self.visual_encoder(image) 
            image_feats = image_embeds[:,0,:]

            return image_feats

         
        

         
        

  

