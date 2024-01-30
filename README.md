# PyFloral Image Classifier Project
Neural network-based image classifier for accurate and fast flower identification.

## Overview

This repository contains the code and assets for a deep learning project that builds an image classifier using PyTorch and torchvision. The project includes scripts for training a model on an image dataset and making predictions on new images.

## Project Structure

The project is organized into the following components:

- **`train.py`**: Script for training a deep learning model on an image dataset.
- **`predict.py`**: Script for making predictions using a trained model checkpoint.
- **`cat_to_name.json`**: JSON file mapping category indices to flower names.
- **`checkpoint.pth`**: Example trained model checkpoint.
- **`/flowers`**: Sample image dataset directory containing training and validation sets.

## How to Use

### Training the Model

To train the model on your own dataset, use the following command:

```bash
python train.py /path/to/dataset --arch vgg16 --hidden_units 512 --learning_rate 0.001 --epochs 10 --save_dir checkpoint.pth --gpu
```

Replace `/path/to/dataset` with the path to your image dataset.

### Making Predictions

To make predictions on a new image, use the following command:

```bash
python predict.py /path/to/image checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```

Replace `/path/to/image` with the path to the image you want to classify.

## Dependencies

Make sure you have the required dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- This project was built using PyTorch and torchvision.
- The image dataset used for training is based on the [Flower Recognition Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

