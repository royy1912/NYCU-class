import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import argparse
from utils import LoadTestData, LoadMaskData
from torch.utils.data import Dataset,DataLoader
from torchvision import utils as vutils
import os
from models import MaskGit as VQGANTransformer
import yaml
import torch.nn.functional as F
from torchvision.utils import make_grid
class MaskGIT:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.model.load_transformer_checkpoint(args.load_transformer_ckpt_path)
        self.model.eval()
        self.total_iter=args.total_iter
        self.mask_func=args.mask_func
        self.sweet_spot=args.sweet_spot
        self.device=args.device
        self.prepare()

    @staticmethod
    def prepare():
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("mask_scheduling", exist_ok=True)
        os.makedirs("imga", exist_ok=True)

##TODO3 step1-1: total iteration decoding  
#mask_b: iteration decoding initial mask, where mask_b is true means mask
    
    def inpainting(self,image,mask_b,i):
        B = image.size(0)
        maska = torch.zeros(B, self.total_iter + 1, 3, 16, 16, device=self.device) 
        imga = torch.zeros(B, self.total_iter+1, 3, 64, 64, device=self.device)
        mean = torch.tensor([0.4868, 0.4341, 0.3844],device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2620, 0.2527, 0.2543],device=self.device).view(1, 3, 1, 1)

        ori = image * std + mean
        imga[:,0] = ori

        self.model.eval()
        with torch.no_grad():
            z_indices = self.model.encode_to_z(image)
            z_indices_predict = z_indices.clone()
            mask_bc = mask_b.to(device=self.device)
            original_mask = mask_b.to(self.device).view(B, 16, 16)

            initial_mask = (~original_mask).float().unsqueeze(1).expand(-1, 3, -1, -1)
            maska[:, 0] = initial_mask

            for step in range(self.total_iter):
                if step == self.sweet_spot:
                    break
                z_indices_predict, mask_bc = self.model.inpainting(z_indices_predict, mask_bc, step, self.total_iter)

                current_mask = (original_mask & mask_bc.view(B, 16, 16))
                mask_image = (~current_mask).float().unsqueeze(1).expand(-1, 3, -1, -1)
                maska[:, step + 1] = mask_image

                z_q_origin = self.model.vqgan.codebook.embedding(z_indices).view(B, 16, 16, 256)
                z_q_predict = self.model.vqgan.codebook.embedding(z_indices_predict).view(B, 16, 16, 256)

                mask_reshape = original_mask.unsqueeze(-1)
                z_q = torch.where(mask_reshape, z_q_predict, z_q_origin).permute(0, 3, 1, 2)

                decoded_img = self.model.vqgan.decode(z_q)
                dec_img_ori = torch.clamp(decoded_img * std + mean, 0.0, 1.0)
                imga[:, step+1] = dec_img_ori

            for b in range(B):
                masks = maska[b]  
                masks_gray = torch.cat([masks[0:1], masks[1:].flip(0)], dim=0)[:, 0:1, :, :]  
                mask_grid = make_grid(masks_gray, nrow=self.total_iter + 1, normalize=True, padding=2)
                vutils.save_image(mask_grid, os.path.join("mask_scheduling", f"image_mask_{i + b:03d}.png"))
                vutils.save_image(dec_img_ori[b], os.path.join("test_results", f"image_{i + b:03d}.png"), nrow=1)
                vutils.save_image(imga[b], os.path.join("imga", f"test_{i + b}.png"), nrow=7)


class MaskedImage:
    def __init__(self, args):
        mi_ori=LoadTestData(root= args.test_maskedimage_path, partial=args.partial)
        self.mi_ori =  DataLoader(mi_ori,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)
        mask_ori =LoadMaskData(root= args.test_mask_path, partial=args.partial)
        self.mask_ori =  DataLoader(mask_ori,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)
        self.device=args.device


    def get_mask_latent(self,mask):    
        downsampled1 = torch.nn.functional.avg_pool2d(mask, kernel_size=2, stride=2)
        resized_mask = torch.nn.functional.avg_pool2d(downsampled1, kernel_size=2, stride=2)
        resized_mask[resized_mask != 1] = 0
        mask_tokens=(resized_mask[:,0]//1).flatten(start_dim=1)
        mask_b = torch.zeros_like(mask_tokens, dtype=torch.bool, device=self.device)
        mask_b |= (mask_tokens == 0)
        return mask_b

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT for Inpainting")
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')#cuda
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for testing.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker')
    
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for MaskGIT')
    
    
#TODO3 step1-2: modify the path, MVTM parameters
    parser.add_argument('--load-transformer-ckpt-path', type=str, default='./transformer_checkpoints/epoch50_transformer_only_4.pt', help='load ckpt')
    
    #dataset path
    parser.add_argument('--test-maskedimage-path', type=str, default='./cat_face/masked_image', help='Path to testing image dataset.')
    parser.add_argument('--test-mask-path', type=str, default='./cat_face/mask64', help='Path to testing mask dataset.')
    #MVTM parameter
    parser.add_argument('--sweet-spot', type=int, default=8, help='sweet spot: the best step in total iteration')
    parser.add_argument('--total-iter', type=int, default=8, help='total step for mask scheduling')
    parser.add_argument('--mask-func', type=str, default='logarithm', help='mask scheduling function')

    args = parser.parse_args()

    t=MaskedImage(args)
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    maskgit = MaskGIT(args, MaskGit_CONFIGS)
    maskgit.model.gamma = maskgit.model.gamma_func(args.mask_func)
    i=0
    for image, mask in zip(t.mi_ori, t.mask_ori):
        image=image.to(device=args.device)
        mask=mask.to(device=args.device)
        mask_b=t.get_mask_latent(mask)       
        maskgit.inpainting(image,mask_b,i)
        i+=1
        


