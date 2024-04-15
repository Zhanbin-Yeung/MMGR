from functools import partial
from models.image_encoder import VisionTransformer
from models.text_encoder import BertConfig, BertModel
from models.decoder import BertLMHeadModel

import torch
from torch import nn
import torch.nn.functional as F

class ALBEF2(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 ids_tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.ids_tokenizer = ids_tokenizer 
        self.distill = config['distill']
        embed_dim = config['embed_dim']        
        vision_width = config['vision_width']  
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        config_decoder = BertConfig.from_json_file(config['bert_config_decode'])
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel(config_decoder)
        

    def forward(self,
                content=None, summary=None,
                index_id=None, retrieval_id=None,
                alpha=None, 
                idx=None, 
                train=True,
                k=10, 
                special_token=None):
            
        if train:
            index_loss_lm = 0
            with torch.no_grad():
                content_output = self.text_encoder(content.input_ids, attention_mask = content.attention_mask,                      
                                                    return_dict = True, mode = 'text')            
                content_embeds = content_output.last_hidden_state
                summary_output = self.text_encoder(summary.input_ids, attention_mask = summary.attention_mask,                      
                                                    return_dict = True, mode = 'text')            
                summary_embeds = summary_output.last_hidden_state
            # index loss
            index_id_targets = index_id.input_ids.masked_fill(index_id.input_ids == self.ids_tokenizer.pad_token_id, -100)


            index_id_output = self.text_decoder(index_id.input_ids,
                                            attention_mask = index_id.attention_mask, 
                                            encoder_hidden_states = content_embeds,
                                            encoder_attention_mask = content.attention_mask,                  
                                            labels = index_id_targets,
                                            return_dict = True,   
                                            reduction = 'none',
                                            ) 

            Index_loss_lm =  index_id_output.loss
            index_loss_lm = Index_loss_lm.sum() / content.input_ids.size(0) 

            
            retrieval_loss_lm = 0
            # retrieval loss
            retrieval_id_targets = retrieval_id.input_ids.masked_fill(retrieval_id.input_ids == self.ids_tokenizer.pad_token_id, -100)

            retrieval_id_output = self.text_decoder(retrieval_id.input_ids,
                                            attention_mask = retrieval_id.attention_mask, 
                                            encoder_hidden_states = summary_embeds,
                                            encoder_attention_mask = summary.attention_mask,                  
                                            labels = retrieval_id_targets,
                                            return_dict = True,   
                                            reduction = 'none',
                                            ) 

            Retrieval_loss_lm =  retrieval_id_output.loss
            retrieval_loss_lm = Retrieval_loss_lm.sum() / content.input_ids.size(0) 

            return index_loss_lm, retrieval_loss_lm
        
        else:
            with torch.no_grad():
                summary_output = self.text_encoder(summary.input_ids, attention_mask = summary.attention_mask,                      
                                                    return_dict = True, mode = 'text')            
                summary_embeds = summary_output.last_hidden_state
                
            # caption to image retrieval
            t2i_outputs = self.text_decoder.generate(inputs=retrieval_id, 
                                       num_beams=k,
                                       num_return_sequences=k, 
                                       max_length=10,
                                       encoder_hidden_states = summary_embeds,
                                       encoder_attention_mask = summary.attention_mask,
                                       bos_token_id=special_token['BOS'],
                                       eos_token_id=special_token['EOS'],
                                       pad_token_id=special_token['PAD']
                                       )
            
            return t2i_outputs
        

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idx)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr  
        

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output        






        # if train:
#             image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
#             text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 

#             idx = idx.view(-1,1)
#             idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
#             pos_idx = torch.eq(idx, idx_all).float()       
#             sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)     

#             with torch.no_grad():
#                 self._momentum_update()
#                 image_embeds_m = self.visual_encoder_m(image) 
#                 image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
#                 image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                                         
#                 text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,             
#                                                     return_dict = True, mode = 'text')    
#                 text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
#                 text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

#                 if self.distill:               
#                     sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
#                     sim_t2i_m = text_feat_m @ image_feat_all / self.temp   

#                     sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
#                     sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets 

#             sim_i2t = image_feat @ text_feat_all / self.temp 
#             sim_t2i = text_feat @ image_feat_all / self.temp           

#             if self.distill:
#                 loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
#                 loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 
#             else:
#                 loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
#                 loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()   

#             loss_ita = (loss_i2t+loss_t2i)/2

#             self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

#             ###=================================###
#             # forward the positve image-text pair
#             output_pos = self.text_encoder(encoder_embeds = text_embeds, 
#                                             attention_mask = text.attention_mask,
#                                             encoder_hidden_states = image_embeds,
#                                             encoder_attention_mask = image_atts,      
#                                             return_dict = True,
#                                             mode = 'fusion',
#                                         )            
#             with torch.no_grad():
#                 bs = image.size(0)      
#                 weights_i2t = F.softmax(sim_i2t[:,:bs]+1e-4,dim=1)
#                 weights_t2i = F.softmax(sim_t2i[:,:bs]+1e-4,dim=1)

#                 mask = torch.eq(idx, idx.T)
#                 weights_i2t.masked_fill_(mask, 0)
#                 weights_t2i.masked_fill_(mask, 0) 

#             # select a negative image for each text
#             image_embeds_neg = []    
#             for b in range(bs):
#                 neg_idx = torch.multinomial(weights_t2i[b], 1).item()
#                 image_embeds_neg.append(image_embeds[neg_idx])
#             image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

#             # select a negative text for each image
#             text_embeds_neg = []
#             text_atts_neg = []
#             for b in range(bs):
#                 neg_idx = torch.multinomial(weights_i2t[b], 1).item()
#                 text_embeds_neg.append(text_embeds[neg_idx])
#                 text_atts_neg.append(text.attention_mask[neg_idx])
#             text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
#             text_atts_neg = torch.stack(text_atts_neg,dim=0)      

#             text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
#             text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

#             image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
#             image_atts_all = torch.cat([image_atts,image_atts],dim=0)

#             output_neg = self.text_encoder(encoder_embeds = text_embeds_all, 
#                                             attention_mask = text_atts_all,
#                                             encoder_hidden_states = image_embeds_all,
#                                             encoder_attention_mask = image_atts_all,      
#                                             return_dict = True,
#                                             mode = 'fusion',
#                                         )                         

#             vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
#             vl_output = self.itm_head(vl_embeddings)            

#             itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
#                                 dim=0).to(image.device)
#             loss_itm = F.cross_entropy(vl_output, itm_labels)     from functools import partial
