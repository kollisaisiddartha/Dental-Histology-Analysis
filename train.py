import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from itertools import product
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    calculate_metrics,
    save_predictions_as_imgs
)


# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 0
NUM_WORKERS = 2
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "UNET Dataset/train/images/"
TRAIN_MASK_DIR = "UNET Dataset/train/masks/"
VAL_IMG_DIR = "UNET Dataset/valid/images/"
VAL_MASK_DIR = "UNET Dataset/valid/masks/"
TEST_IMG_DIR = "UNET Dataset/test/images/"
TEST_MASK_DIR = "UNET Dataset/test/masks/"

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Class weights for CrossEntropyLoss (adjust based on class distribution)
# class_weights = torch.tensor([1.0, 2.0, 3.0]).to(DEVICE)  # Example weights for 3 classes

#######################################################################################################################

def train_func(loader, model, optimizer, loss_fn, scalar, iou_metric, accuracy_metric, dice_metric):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # Forward pass
        with torch.amp.autocast(device_type='cuda'):
            # Get predictions
            predictions = model(data)

            # Compute loss using loss_fn
            crossentropy_loss = loss_fn(predictions, targets.long())
            dice_loss = 1 - dice_metric(predictions.softmax(dim=1), targets.long())
            # Compute Train IoU metric
            iou_metric.update(predictions, targets.long())
            # Compute Train Accuracy
            accuracy_metric.update(predictions.argmax(dim=1), targets.long())

            total_loss = 0.5 * crossentropy_loss + 0.5 * dice_loss;            

        # Backward pass
        optimizer.zero_grad()
        scalar.scale(total_loss).backward()
        scalar.step(optimizer)
        scalar.update()

        # Update tqdm loop
        loop.set_postfix(loss=total_loss.item())

#######################################################################################################################

def grid_search_class_weights(model, train_loader, val_loader, num_classes, num_epochs=2):
    # Define a grid of possible class weights
    weight_grid = {
        'class_1': torch.arange(0.2, 2.0, 0.4).tolist(),
        'class_2': torch.arange(0.2, 2.0, 0.4).tolist(),
        'class_3': torch.arange(0.2, 2.0, 0.4).tolist(),
    }

    iou_metric = torchmetrics.JaccardIndex(num_classes=3, task="multiclass").to(DEVICE)
    accuracy_metric = torchmetrics.Accuracy(num_classes=3, task="multiclass").to(DEVICE)
    dice_metric = torchmetrics.Dice(num_classes=3, average="macro").to(DEVICE)
    ap_metric = torchmetrics.AveragePrecision(num_classes=3, task="multiclass").to(DEVICE)
    scaler = torch.amp.GradScaler()

    # Create all combinations of weights
    weight_combinations = list(product(weight_grid['class_1'], weight_grid['class_2'], weight_grid['class_3']))

    best_loss = float('inf')
    best_weights = None

    for weights in weight_combinations:
        print(f'Trying class weights: {weights}')
        class_weights_tensor = torch.tensor(weights).float().to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training the model with the current weights
        for epoch in range(num_epochs):
            train_func(train_loader, model, optimizer, loss_fn, scaler, iou_metric, accuracy_metric, dice_metric)

        # Evaluate on the validation set
        val_iou, val_accuracy, val_dice, val_map = calculate_metrics(val_loader, model, iou_metric, accuracy_metric, dice_metric, ap_metric, DEVICE)
        current_loss = 1 - val_dice  

        if current_loss < best_loss:
            best_loss = current_loss
            best_weights = weights
            print(f'Best class weights: {best_weights} with loss: {best_loss}')

    print(f'Final Best class weights: {best_weights} with loss: {best_loss}')
    return best_weights

#######################################################################################################################

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.2),
            A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.5),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ])

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ])


    # Creating a model and initializing all parameters and metrics
    model = UNET(in_channels=3, out_channels=3).to(DEVICE)  # out_channels == the number of classes
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    iou_metric = torchmetrics.JaccardIndex(num_classes=3, task="multiclass").to(DEVICE)
    accuracy_metric = torchmetrics.Accuracy(num_classes=3, task="multiclass").to(DEVICE)
    dice_metric = torchmetrics.Dice(num_classes=3, average="macro").to(DEVICE)
    ap_metric = torchmetrics.AveragePrecision(num_classes=3, task="multiclass").to(DEVICE)

    # Load the dataset from util get_loaders()
    train_loader, val_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    # Perform Grid Search for Class weights
    #best_weights = grid_search_class_weights(model, train_loader, val_loader, num_classes=3, num_epochs=2)
    #class_weights = torch.tensor(best_weights).float().to(DEVICE)
    class_weights = torch.tensor([1.8, 1.8, 1.8]).to(DEVICE) # Already executed grid search and found these with best loss.
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)  # Uses class weights

    # Loading model if LOAD_MODEL is set to "true"
    if LOAD_MODEL:
        load_checkpoint(torch.load("epoch50_size640.tar"), model)

    # Sanity Check (i.e., when a model is loaded)
    val_iou, val_accuracy, val_dice, val_map = calculate_metrics(val_loader, model, iou_metric, accuracy_metric, dice_metric, ap_metric, DEVICE)
    print(f"Validation IoU: {val_iou:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Dice Score: {val_dice:.4f}, Validation mAP Score: {val_map:.4f}\n")
    scaler = torch.amp.GradScaler()


    # Start of epoch loop
    for epoch in range(NUM_EPOCHS):
        print(f"EPOCH: {epoch + 1}/{NUM_EPOCHS}")  # Prints the current EPOCH number

        # Training for one epoch
        train_func(train_loader, model, optimizer, loss_fn, scaler, iou_metric, accuracy_metric, dice_metric)

        # Compute training IoU and accuracy after the epoch
        train_iou = iou_metric.compute()
        train_accuracy = accuracy_metric.compute()
        print(f"Training IoU: {train_iou:.4f}, Training Accuracy: {train_accuracy:.4f}\n")
        iou_metric.reset()
        accuracy_metric.reset()

        # Calculate validation metrics (IoU, accuracy, and dice score)
        val_iou, val_accuracy, val_dice, val_map = calculate_metrics(val_loader, model, iou_metric, accuracy_metric, dice_metric, ap_metric, DEVICE)
        print(f"Validation IoU: {val_iou:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Dice Score: {val_dice:.4f}, Validation mAP Score: {val_map:.4f}\n")

        # Step the scheduler
        current_lr = scheduler.get_last_lr()[0]  
        print(f"Current Learning Rate: {current_lr}")
        scheduler.step(val_dice)

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

    # Test metrics
    test_iou, test_accuracy, test_dice, test_map = calculate_metrics(test_loader, model, iou_metric, accuracy_metric, dice_metric, ap_metric, DEVICE)
    print(f"Test IoU: {test_iou:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Dice Score: {test_dice:.4f}, Test mAP Score: {test_map:.4f}\n")

    # Store output concated with input
    save_predictions_as_imgs(test_loader, model, folder="saved_images/", device=DEVICE)


if __name__ == "__main__":
    main()
