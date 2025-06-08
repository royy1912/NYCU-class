def dice_score(preds, masks, smooth=1e-6):
    """ 計算 Dice Score """
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + smooth)
    return dice.mean().item()

