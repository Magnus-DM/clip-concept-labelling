import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import sys
import os
import clip
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from PIL import Image
import random
import matplotlib.pyplot as plt
import re
from sklearn.metrics import f1_score
#import torch_directml # Uncomment if using AMD GPU

# Import from helper functions
from datasets import celebADataset,VOC2012Dataset,CUB200Dataset, PreprocessedDataset, load_datasets, save_datasets, get_clip_image_embedding, get_clip_text_embedding
from sampling import oversample_minority_class, undersample_majority_class, calculate_0_1_label_ratio
from train import train_model
from visualization import plot_losses, plot_metrics
from model import AttributeClassifier

# Step 1: Choose GPU device if available
device = "cpu"
#device = torch_directml.device()  # Uses DirectML for AMD GPUs
print(f"Using device: {device}")

# ============ Define all parameters ===========================================================
preprocessing = False # Should the model use metrics (preprocessing = True) or raw embeddings (preprocessing = False)
negative_concepts = False # Should the model inlude the use of negative concepts
double_nn = False # Are positive and negative concepts inputted in 1 neural network or in 2 separate ones
double_input = False # Give the positive concepts and negative concepts as separate datapoints to the neural network

# The root_cub directory of the CUB200 dataset
root_cub = "./data/CUB_200_2011"
root_voc2012 = "./data/VOC2012"
root_celeba = "./data/celebA"

# Variables that determine the percentage of images and attributes to use
# Both should be a divisor of 100
images_percentage = 1
attributes_percentage = 100

# Instead of using a percentage of the attributes, you can also specify which attributes 
# should be picked by providing a list with the numbers of the attributes (1 to 312)
attribute_list = None # e.g. [55, 56, 134]

use_processed_concepts = False # Variable that states if the processed concepts should be used

image_split_list = [0.8, 0.1, 0.1] # Variable that states in which ratio the images should be split into training, validation, and test sets

# Variable that states if the attributes should be split into training, validation, and test sets and in which ratio if so
split_attributes = True
attribute_split_list = [0.4, 0.3, 0.3]

# Variable that states if the dataset should be attempted to be loaded from disk
attempt_load = True

# Variables that state if oversampling of the minority class/undersampling of the majority class should be applied
oversample_training = False
undersample_training = False
oversample_validation = False
undersample_validation = False

learning_rate = 0.002 # Learning rate for neural network
regularization = False # Using regularization or not in the neural network and optimizer
weight_decay = 0.0001 # If regularization is used, the weight decay for the optimizer
dropout_rate = 0.5 # If regularization is used, the dropout rate for the neural network
batch_size = 2048 # Batch size for neural network
epochs = 10 # Amount of epochs used
weighted_loss = True # Whether to use weighted loss or not (as compensation for class imbalance)
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # Thresholds to apply to the output of the neural network to determine the prediction

# Preprocessing of some parameters to ensure no weird things happen

# If negative_concepts is not used, double_nn should be set to False
if not negative_concepts:
    double_nn = False
    double_input = False

assert not (double_nn and double_input), "Double input should not be used with double nn"

# Each element of the attribute_list is reduced by 1, to correspond to the correct index in the dataset
if attribute_list is not None:
    attribute_list = [attribute - 1 for attribute in attribute_list]

# Don't attempt to load the dataset if the attribute_list is not None, as it will be incorrectly loaded
if attribute_list is not None:
    attempt_load = False

# ============ End of parameter definition =====================================================



# Step 2: Load CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
# Step 3: Load CUB200 dataset
dataset = celebADataset(root_celeba, device, clip_model, preprocess, preprocessing, negative_concepts, images_percentage, attributes_percentage, 
                        attribute_list, use_processed_concepts, image_split_list, split_attributes, attribute_split_list, double_input)

dataset.get_image_name_with_labels([1, 2, 3])
#print(dataset.calculate_label_ratio())