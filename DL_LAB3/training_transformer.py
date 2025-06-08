import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        if os.path.exists(args.checkpoint_path):
            print(f">> Loading transformer weights from {args.checkpoint_path}")
            self.model.transformer.load_state_dict(torch.load(args.checkpoint_path))
        self.vocab_size = MaskGit_CONFIGS["model_param"]['num_codebook_vectors'] + 1  
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)


    def train_one_epoch(self, train_loader, epoch, device):
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            x = data.to(device)
            logits, target, mask = self.model(x)  
            assert (target < self.vocab_size - 1).all(), 
            
            logits = logits[mask]          
            target = target[mask]         

            loss = F.cross_entropy(logits, target)

            loss.backward()

            if (i + 1) % self.args.accum_grad == 0:
                
                self.optim.step()
                self.scheduler.step()
                self.optim.zero_grad()

            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

        return total_loss / len(train_loader)


    @torch.no_grad()
    def eval_one_epoch(self, val_loader, epoch, device):
        self.model.eval()
        total_loss = 0

        pbar = tqdm(val_loader)
        for i, data in enumerate(pbar):
            x = data.to(device)
            logits, target, mask = self.model(x)  

            logits = logits[mask]
            target = target[mask]

            loss = F.cross_entropy(logits, target)
            total_loss += loss.item()

            pbar.set_description(f"[Eval] Epoch {epoch} Loss {loss.item():.4f}")

            z_indices = self.model.encode_to_z(x)
            z_masked = z_indices.clone()
            z_masked[mask] = self.model.mask_token_id

           
            logits = self.model.transformer(z_masked)
            pred_ids = torch.argmax(logits, dim=-1)

            
            z_q = self.model.vqgan.codebook.embedding(pred_ids)
            B, N, C = z_q.shape  
            h = w = int(N ** 0.5)
            z_q = z_q.permute(0, 2, 1).view(B, C, h, w)  
            recon = self.model.vqgan.decode(z_q)  

            
            mean = torch.tensor([0.4868, 0.4341, 0.3844], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.2620, 0.2527, 0.2543], device=device).view(1, 3, 1, 1)
            ori = x * std + mean
            recon = recon * std + mean

            
            for b in range(x.size(0)):
                vutils.save_image(ori[b], f"./test/output/orig_{i*self.args.batch_size+b}.png")
                vutils.save_image(recon[b], f"./test/output/recon_{i*self.args.batch_size+b}.png")

        return total_loss / len(val_loader)

    def configure_optimizers(self, steps_per_epoch):
        optimizer = torch.optim.AdamW(
            self.model.transformer.parameters(),  
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )

        total_steps = self.args.epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)  

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return cosine_decay * (1.0 - 0.5) + 0.5  

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./transformer_checkpoints/epoch120_transformer_only_3.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=8, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=10, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)
    steps_per_epoch = len(train_loader)
    train_transformer.optim, train_transformer.scheduler = train_transformer.configure_optimizers(steps_per_epoch)
    
    
    
    
    
    train_losses = []
    val_losses = []
    
#TODO2 step1-5:    
    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        avg_loss = train_transformer.train_one_epoch(train_loader, epoch, args.device)
        
        val_loss = train_transformer.eval_one_epoch(val_loader, epoch, args.device)
        
        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        print(f"[Eval Epoch {epoch}] Val Loss: {val_loss:.4f}")
        if epoch % args.save_per_epoch == 0:
            
            torch.save(train_transformer.model.transformer.state_dict(), f'transformer_checkpoints/epoch{epoch}_transformer_only_f.pt')

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('transformer_checkpoints/loss_trend_10.png')
    plt.close()