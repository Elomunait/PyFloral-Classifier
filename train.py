# Import necessary libraries
# load libraries
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms, models
import os
from collections import OrderedDict
from os import listdir
import time
import copy
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a deep learning model on an image dataset.')
    parser.add_argument('data_dir', type=str, help='Location of the directory with training and validation data')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'alexnet', 'densenet121'], help='Choose the architecture (vgg16, alexnet, densenet121)')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units for the first layer')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save the trained model')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()
    return args

def check_gpu():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_model(arch='vgg16', hidden_units=512, learning_rate=0.001):
    # Function builds model
    model = getattr(models, arch)(pretrained=True)
    in_features = model.classifier[0].in_features

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features, hidden_units)),
        ('ReLu1', nn.ReLU()),
        ('Dropout1', nn.Dropout(p=0.15)),
        ('fc2', nn.Linear(hidden_units, 512)),
        ('ReLu2', nn.ReLU()),
        ('Dropout2', nn.Dropout(p=0.15)),
        ('fc3', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1, last_epoch=-1)

    return model, criterion, optimizer, scheduler

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, epochs=10):
    # Function that trains pretrained model and classifier on an image dataset and validates.
    since = time.time()
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

def save_model(model, args, image_datasets):
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    save_dir = args.save_dir
    checkpoint = {
        'arch': args.arch,
        'hidden_units': args.hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
    }

    torch.save(checkpoint, save_dir)
    print(f"Model checkpoint saved to {save_dir}")

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Check for GPU
    device = check_gpu() if args.gpu else torch.device("cpu")
    print(f"Data directory: {args.data_dir}")

    # Create model, criterion, optimizer, and scheduler
    model, criterion, optimizer, scheduler = create_model(args.arch, args.hidden_units, args.learning_rate)

    # Define data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'train')  # Fix the path here
    valid_dir = os.path.join(data_dir, 'valid')  # Fix the path here

    print(f"Train directory: {train_dir}")
    print(f"Valid directory: {valid_dir}")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'valid']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    
    # Train the model
    model_trained = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, args.epochs)

    # Save the trained model
    save_model(model_trained, args, image_datasets)

if __name__ == "__main__":
    main()

# python /home/workspace/ImageClassifier/train.py /home/workspace/ImageClassifier/flowers --arch vgg16 --hidden_units 512 --learning_rate 0.001 --epochs 10 --save_dir checkpoint.pth --gpu
