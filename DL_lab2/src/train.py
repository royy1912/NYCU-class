import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.unet import UNet 
from oxford_pet import load_dataset 
from evaluate import evaluate , dice_score ,compute_metrics
from models.resnet34_unet import ResNet34UNet
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds) 
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth)
        return 1 - dice.mean() 
    
criterion_bce = nn.BCEWithLogitsLoss()
criterion_dice = DiceLoss()    
def loss_fn(preds, targets):
    return 1 * criterion_bce(preds, targets) + 0 * criterion_dice(preds, targets) 
def train(args):
    """ 訓練 UNet 模型 """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = load_dataset(args.data_path, mode="train", batch_size=args.batch_size)
    val_loader = load_dataset(args.data_path, mode="valid", batch_size=args.batch_size)

    model = UNet(in_channels=3, out_channels=1).to(device)
    

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate ,  weight_decay=1e-4)

    loss_history = []
    val_loss_history = []
    dice_scores = []

    for batch in train_loader:
        masks = batch["mask"]
        image = batch["image"]
        print(image.shape)
        print(torch.unique(masks))
        break  

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            images, masks = batch["image"].to(device), batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks.float()) 
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            dice_score, pixel_acc = evaluate(model, val_loader, device)  
            dice_scores.append(dice_score)

            val_batches = 0
            for batch in val_loader:
                images, masks = batch["image"].to(device), batch["mask"].to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks.float())
                val_loss += loss.item()
                val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_loss_history.append(avg_val_loss)

        print(f"Validation Loss: {avg_val_loss:.4f}, Dice Score: {dice_score:.4f}, Pixel Accuracy: {pixel_acc:.4f}")

    torch.save(model.state_dict(), "../saved_models/resnet34unet_8.pth")
    print("Model training completed and saved!")


    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.epochs + 1), loss_history, marker='o', linestyle='-', color='b', label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.savefig("../saved_models/loss_curve.png")
    print("Loss curve saved as 'loss_curve.png'")


    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.epochs + 1), val_loss_history, marker='o', linestyle='-', color='r', label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("../saved_models/validation_loss_curve.png")
    print("Validation Loss curve saved as 'validation_loss_curve.png'")


    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.epochs + 1), dice_scores, marker='o', linestyle='-', color='g', label="Validation Dice Score")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Score Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("../saved_models/dice_score_curve.png")
    print("Dice Score curve saved as 'dice_score_curve.png'")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)