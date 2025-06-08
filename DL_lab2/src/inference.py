import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from models.unet import UNet
from oxford_pet import load_dataset
from evaluate import evaluate 
from models.resnet34_unet import ResNet34UNet
def load_model(model_path, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  
    return model



def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path, device)

    test_loader = load_dataset(args.data_path, mode="test", batch_size=args.batch_size)

    print("Evaluating on Test Set...")
    test_iou, test_acc = evaluate(model, test_loader, device)
    print(f"Test dice_score: {test_iou:.4f}, Pixel Accuracy: {test_acc:.4f}")

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weight')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
