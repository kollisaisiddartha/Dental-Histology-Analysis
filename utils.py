import torch
import torchvision
from dataset import UnetDataset
from torch.utils.data import DataLoader
import os
import torchmetrics

#######################################################################################################################

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint\n")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint\n")
    model.load_state_dict(checkpoint["state_dict"])

#######################################################################################################################

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    test_dir,
    test_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True):

    train_ds = UnetDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True)

    val_ds = UnetDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False)

    test_ds = UnetDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=val_transform)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False)

    return train_loader, val_loader, test_loader

#######################################################################################################################

def apply_color_map(predictions, color_map):
    """
    Args:
        predictions (torch.Tensor): Predicted masks of shape (N, H, W).
        color_map (dict): A dictionary mapping class indices to RGB tuples.

    Returns:
        torch.Tensor: Colorized masks of shape (N, 3, H, W).
    """
    N, H, W = predictions.size()
    colorized_masks = torch.zeros((N, 3, H, W), dtype=torch.uint8, device=predictions.device)

    for class_index, color in color_map.items():
        # Create a mask for the current class
        mask = (predictions == class_index)  # Shape (N, H, W)
        
        # Color the corresponding pixels in the colorized masks
        colorized_masks[:, 0, :, :] += mask * color[0]  # Red channel
        colorized_masks[:, 1, :, :] += mask * color[1]  # Green channel
        colorized_masks[:, 2, :, :] += mask * color[2]  # Blue channel

    return colorized_masks

#######################################################################################################################

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    
    os.makedirs(folder, exist_ok=True)  # Ensure the output directory exists
    
    color_map = {
        0: (255, 0, 0),  # Class 0 - Red
        1: (0, 255, 0),  # Class 1 - Green
        2: (0, 0, 255),  # Class 2 - Blue
        # Add more classes and colors as needed
    }

    with torch.no_grad():  # Disable gradient calculation for inference
        for idx, (x, _) in enumerate(loader):
            x = x.to(device=device)  # Move input to GPU
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()  # Apply thresholding for binary mask

            # Convert predictions to class indices
            predictions = preds.argmax(dim=1)  # Shape (N, H, W)

            # Apply color map to the predictions
            colorized_masks = apply_color_map(predictions, color_map)

            # Process all images in the batch
            for i in range(x.size(0)):
                original_image = x[i].cpu()  # Move original image to CPU for visualization
                predicted_image = colorized_masks[i].cpu()  # Move predicted mask to CPU

                # Concatenate original and predicted images along the width (dim=2)
                combined_image = torch.cat((original_image, predicted_image), dim=2)

                # Save the combined image with a unique filename
                torchvision.utils.save_image(combined_image, f"{folder}/combined_{idx * loader.batch_size + i}.png")

    model.train()

#######################################################################################################################

def calculate_metrics(loader, model, iou_metric, accuracy_metric, dice_metric, ap_metric, device='cuda'):
    model.eval()  # Set the model to evaluation mode

    iou_metric = torchmetrics.JaccardIndex(num_classes=3, task="multiclass").to(device)
    accuracy_metric = torchmetrics.Accuracy(num_classes=3, task="multiclass").to(device)
    dice_metric = torchmetrics.Dice(num_classes=3, average="macro").to(device)
    ap_metric = torchmetrics.AveragePrecision(num_classes=3, task="multiclass").to(device)

    ap_values = []

    with torch.no_grad():  # Disable gradient calculation for validation
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device).long()

            # Forward pass to get predictions
            predictions = model(data)
            preds = torch.softmax(predictions, dim=1)  # Prob from predictions used for Dice

            # Update metrics
            iou_metric.update(predictions.argmax(dim=1), targets.long())  
            accuracy_metric.update(predictions.argmax(dim=1), targets.long()) 
            dice_metric.update(preds, targets.long()) 
            ap_metric.update(preds, targets.long())

    # Compute metrics for the entire set
    iou = iou_metric.compute()
    accuracy = accuracy_metric.compute()
    dice = dice_metric.compute()
    ap = ap_metric.compute()

    # Compute mAP
    ap_values.append(ap)
    map = torch.mean(torch.stack(ap_values))
    
    # Reset the metrics for the next validation run
    iou_metric.reset()
    accuracy_metric.reset()
    dice_metric.reset()
    ap_metric.reset()

    model.train()
    return iou, accuracy, dice, map

