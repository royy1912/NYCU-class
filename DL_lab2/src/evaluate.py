import torch
from utils import dice_score


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()

    total_dice = 0.0
    total_acc = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            outputs = model(images)
            
            outputs = torch.sigmoid(outputs)  
            preds = (outputs > 0.5).float()


            
            dice = dice_score(preds, masks)
            total_dice += dice

            
            correct_pixels = (preds == masks).sum().item()
            total_pixels = masks.numel()
            pixel_acc = correct_pixels / total_pixels
            total_acc += pixel_acc

            count += 1

    avg_dice = total_dice / count
    avg_acc = total_acc / count

    print(f"Average Dice Score: {avg_dice:.4f}, Pixel Accuracy: {avg_acc:.4f}")
    return avg_dice, avg_acc

def compute_metrics(preds, masks):
    
    return dice_score(preds, masks)
