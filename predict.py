#load libraries
import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image
import json
import argparse

def load_model(checkpoint_path, gpu):
    checkpoint = torch.load(checkpoint_path)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 25088
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = 9216
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = 1024
        for param in model.parameters():
            param.requires_grad = False
    else:
        print('Sorry base architecture not recognized')
        return None
    
    model.class_to_idx = checkpoint['class_to_idx']
    hidden_units = checkpoint['hidden_units']
    
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
    model.load_state_dict(checkpoint['state_dict'])

    if gpu:
        model = model.cuda()

    return model

def process_image(image_path):
    size = 256, 256
    crop_size = 224
    
    im = Image.open(image_path)
    
    im.thumbnail(size)
    left = (size[0] - crop_size) / 2
    top = (size[1] - crop_size) / 2
    right = (left + crop_size)
    bottom = (top + crop_size)

    im = im.crop((left, top, right, bottom))
    
    np_image = np.array(im)
    np_image = np_image / 255
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    np_image = (np_image - means) / stds
    pytorch_np_image = np_image.transpose(2, 0, 1)
    
    return pytorch_np_image

def predict(image_path, model, topk=5, gpu=False):
    pytorch_np_image = process_image(image_path)
    pytorch_tensor = torch.tensor(pytorch_np_image).float()
    pytorch_tensor = pytorch_tensor.unsqueeze(0)
    
    if gpu:
        model = model.cuda()
        pytorch_tensor = pytorch_tensor.cuda()

    model.eval()
    LogSoftmax_predictions = model.forward(pytorch_tensor)
    predictions = torch.exp(LogSoftmax_predictions)
    
    # Move the predictions and labels to CPU before converting to NumPy
    top_preds, top_labs = predictions.topk(topk)
    top_preds = top_preds.cpu().detach().numpy().tolist()
    top_labs = top_labs.cpu().tolist()
    
    labels = pd.DataFrame({'class': pd.Series(model.class_to_idx), 'flower_name': pd.Series(cat_to_name)})
    labels = labels.set_index('class')
    labels = labels.iloc[top_labs[0]]
    labels['predictions'] = top_preds[0]
    
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict flower name from an image with checkpoint')
    parser.add_argument('image_path', type=str, help='Location of image to predict e.g. flowers/test/class/image')
    parser.add_argument('checkpoint', type=str, help='Name of trained model checkpoint to be loaded and used for predictions.')
    parser.add_argument('--top_k', type=int, default=5, help='Select number of classes you wish to see in descending order.')
    parser.add_argument('--category_names', type=str, help='Define name of JSON file holding class names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()
    
    model = load_model(args.checkpoint, args.gpu)

    if model is not None:
        if args.category_names:
            with open(args.category_names, 'r') as f:
                cat_to_name = json.load(f)

        labels = predict(args.image_path, model, args.top_k, args.gpu)
        
        print('-' * 40)
        print(labels)
        print('-' * 40)


# python /home/workspace/ImageClassifier/predict.py /home/workspace/ImageClassifier/flowers/test/72/image_03624.jpg checkpoint.pth --top_k 5 --category_names /home/workspace/ImageClassifier/cat_to_name.json --gpu
