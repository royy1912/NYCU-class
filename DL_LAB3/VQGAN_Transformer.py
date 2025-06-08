import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import torch.nn.functional as F
import random
#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs']) 
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])
        self.vocab_size = configs['num_codebook_vectors'] + 1  
        self.mask_token_id = self.vocab_size - 1               

        with torch.no_grad():
            old_weight = self.vqgan.codebook.embedding.weight  
            new_embed = nn.Embedding(self.vocab_size, old_weight.shape[1])
            new_embed.weight[:-1] = old_weight  
            new_embed.weight[-1].zero_()        
            self.vqgan.codebook.embedding = new_embed

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        cfg['model_param']['image_channels'] = int(cfg['model_param']['image_channels'])  
        cfg['model_param']['enc_channels'] = [int(x) for x in cfg['model_param']['enc_channels']]
        cfg['model_param']['dec_channels'] = [int(x) for x in cfg['model_param']['dec_channels']]
        cfg['model_param']['latent_dim'] = int(cfg['model_param']['latent_dim'])
        cfg['model_param']['img_resolution'] = int(cfg['model_param']['img_resolution'])

        cfg['model_param']['latent_resolution'] = int(cfg['model_param']['latent_resolution'])
        cfg['model_param']['num_codebook_vectors'] = int(cfg['model_param']['num_codebook_vectors'])
        
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        _, z_indices, _ = self.vqgan.encode(x)       
        B = x.shape[0]
        H, W = self.vqgan.encoder(x).shape[2:]       
        z_indices = z_indices.view(B, H * W)         
        return z_indices   
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: 1.0 - r
        elif mode == "cosine":
            return lambda r: math.cos(r * math.pi * 0.5)
        elif mode == "square":
            return lambda r: 1.0 - r ** 2
        elif mode == "logarithm":
            return lambda r: math.log(1 + r * 9) / math.log(10)  
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        
        z_indices = self.encode_to_z(x)  
        B, N = z_indices.shape
        device = z_indices.device
        mask_ratio = random.uniform(0.4, 0.55)
        mask = torch.rand((B, N), device=device) < mask_ratio

        x_input = z_indices.clone()
        x_input[mask] = self.mask_token_id

        logits = self.transformer(x_input)  

        return logits, z_indices, mask
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_bc, t, T):
        B, N = z_indices.shape
        device = z_indices.device

        assert z_indices.shape == mask_bc.shape, f"Shape mismatch: z_indices {z_indices.shape}, mask_bc {mask_bc.shape}"

        z_pred = z_indices.clone()
        z_pred = z_pred.masked_fill(mask_bc, self.mask_token_id)  

        logits = self.transformer(z_pred)  
        probs = torch.softmax(logits, dim=-1)
        pred_probs, pred_ids = torch.max(probs, dim=-1) 

        print("vocab size:", self.vocab_size)
        print("pred_ids max:", pred_ids.max().item())
        assert pred_ids.max() < self.vocab_size, f"pred_ids overflow! max={pred_ids.max().item()}, vocab size={self.vocab_size}"
        pred_ids = torch.clamp(pred_ids, 0, self.vocab_size - 1)

        # Gumbel Noise
        gumbel_noise = -torch.empty_like(pred_probs).exponential_().log()
        
        temperature = self.choice_temperature * (1 - t / T)
        confidence = pred_probs + temperature * gumbel_noise

        
        _, sorted_indices = torch.sort(confidence, dim=-1)
        gamma = self.gamma(t / T)
        num_mask = int(gamma * N)

        new_mask = torch.ones_like(mask_bc, dtype=torch.bool)
        for b in range(B):
            keep_idx = sorted_indices[b, num_mask:]
            keep_idx = keep_idx.clamp(0, N-1).long()
            new_mask[b, keep_idx] = False 

        z_pred = torch.where(mask_bc, pred_ids, z_pred) 
        return z_pred, new_mask
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
