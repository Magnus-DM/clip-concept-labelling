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
preprocessing = True # Should the model use metrics (preprocessing = True) or raw embeddings (preprocessing = False)
negative_concepts = False # Should the model inlude the use of negative concepts
double_nn = True # Are positive and negative concepts inputted in 1 neural network or in 2 separate ones
double_input = False # Give the positive concepts and negative concepts as separate datapoints to the neural network

# The root_cub directory of the CUB200 dataset
dataset_used = ["CUB200", "VOC2012", "celebA"][1] # Choose the dataset to use

root_cub = "./data/CUB_200_2011"
root_voc2012 = "./data/VOC2012"
root_celebA = "./data/celebA"

# Variable that states if the dataset should be attempted to be loaded from disk
attempt_load = True

# Variables that determine the percentage of images and attributes to use
# Both should be a divisor of 100
images_percentage = 1
attributes_percentage = 100

# Instead of using a percentage of the attributes, you can also specify which attributes 
# should be picked by providing a list with the numbers of the attributes
# these values will be reduced by 1 to correspond to the correct index in the dataset
# For VOC2012, the attribute numbers are 1 to 20
# For celebA, the attribute numbers are 1 to 40
# For CUB200, the attribute numbers are 1 to 312
attribute_list = None # e.g. [55, 56, 134]

use_processed_concepts = False # Variable that states if the processed concepts should be used

split_images = False # Variable that states if different images should be used for training, validation, and test sets
image_split_list = [0.8, 0.1, 0.1] # Variable that states in which ratio the images should be split into training, validation, and test sets

# Variable that states if the attributes should be split into training, validation, and test sets and in which ratio if so
split_attributes = True
attribute_split_list = [0.4, 0.3, 0.3]


# Variables that state if oversampling of the minority class/undersampling of the majority class should be applied
oversample_training = False
undersample_training = False
oversample_validation = False
undersample_validation = False

learning_rate = 0.002 # Learning rate for neural network
regularization = True # Using regularization or not in the neural network and optimizer
weight_decay = 0.0001 # If regularization is used, the weight decay for the optimizer
dropout_rate = 0.5 # If regularization is used, the dropout rate for the neural network
batch_size = 2048 # Batch size for neural network
epochs = 35 # Amount of epochs used
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
if dataset_used == "VOC2012":
    dataset = VOC2012Dataset(root_voc2012, device, clip_model, preprocess, preprocessing, negative_concepts, images_percentage, attributes_percentage, 
                        attribute_list, False, split_images, image_split_list, split_attributes, attribute_split_list, double_input)
elif dataset_used == "CUB200":
    dataset = CUB200Dataset(root_cub, device, clip_model, preprocess, preprocessing, negative_concepts, images_percentage, attributes_percentage, 
                       attribute_list, use_processed_concepts, split_images, image_split_list, split_attributes, attribute_split_list, double_input)
elif dataset_used == "celebA":
    dataset = celebADataset(root_celebA, device, clip_model, preprocess, preprocessing, negative_concepts, images_percentage, attributes_percentage, 
                       attribute_list, use_processed_concepts, split_images, image_split_list, split_attributes, attribute_split_list, double_input)

print(f"Size of the {dataset_used} dataset before preprocessing: ", dataset.size())

# Step 4: Preprocess dataset such that it can be used for training a neural network

# Define the strings for the file names
if preprocessing:
    type_string = f"{dataset_used}_preprocessing"
else:
    type_string = f"{dataset_used}_original"

if negative_concepts:
    type_string += "_negative"

    if double_nn:
        type_string += "_double_nn"

    if double_input:
        type_string += "_double_input"

if split_attributes:
    split_attributes_string = "split_on_attributes"
else:
    split_attributes_string = "no_split_on_attributes"

if use_processed_concepts:
    processed_concepts_string = "processed_"
else:
    processed_concepts_string = "original_"

sampling_string = ""
if oversample_training:
    sampling_string += "over_train_"
if undersample_training:
    sampling_string += "under_train_"
if oversample_validation:
    sampling_string += "over_val_"
if undersample_validation:
    sampling_string += "under_val_"

if sampling_string == "":
    sampling_string = "nothing_"

if attribute_list is None:
    percentage_string = f"{images_percentage}_{attributes_percentage}"
else:
    percentage_string = "custom_attributes"

pkl_training_filename = f"./data/{type_string}/training_set/{split_attributes_string}/{processed_concepts_string}concepts_{sampling_string}{percentage_string}.pkl"
pkl_validation_filename = f"./data/{type_string}/validation_set/{split_attributes_string}/{processed_concepts_string}concepts_{sampling_string}{percentage_string}.pkl"
pkl_test_filename = f"./data/{type_string}/test_set/{split_attributes_string}/{processed_concepts_string}concepts_{sampling_string}{percentage_string}.pkl"

# Check if the datasets exist and load them, otherwise preprocess and save them
if attempt_load and os.path.exists(pkl_training_filename) and os.path.exists(pkl_validation_filename) and os.path.exists(pkl_test_filename):
    training_set, validation_set, test_set = load_datasets(pkl_training_filename, pkl_validation_filename, pkl_test_filename)
    print("Training set ratio of 0's to 1's: ", calculate_0_1_label_ratio(training_set.preprocessed_data))
    print("Validation set ratio of 0's to 1's: ", calculate_0_1_label_ratio(validation_set.preprocessed_data))
    print("Test set ratio of 0's to 1's: ", calculate_0_1_label_ratio(test_set.preprocessed_data))
else:
    # Preprocess the datasets
    training_list, validation_list, test_list = dataset.preprocess_dataset()
    print(f"Size of the {dataset_used} training set after preprocessing: ", len(training_list))
    print(f"Size of the {dataset_used} validation set after preprocessing: ", len(validation_list))
    print(f"Size of the {dataset_used} test set after preprocessing: ", len(test_list))
    # Apply oversampling or undersampling if needed
    if oversample_training:
        training_list = oversample_minority_class(training_list)
        print("New ratio of 0s to 1s in the training set, after minority oversampling: ", calculate_0_1_label_ratio(training_list))

    if oversample_validation:
        validation_list = oversample_minority_class(validation_list)
        print("New ratio of 0s to 1s in the validation set, after minority oversampling: ", calculate_0_1_label_ratio(validation_list))

    if undersample_training:
        training_list = undersample_majority_class(training_list)
        print("New ratio of 0s to 1s in the training set, after majority undersampling: ", calculate_0_1_label_ratio(training_list))

    if undersample_validation:
        validation_list = undersample_majority_class(validation_list)
        print("New ratio of 0s to 1s in the validation set, after majority undersampling: ", calculate_0_1_label_ratio(validation_list))

    if not oversample_training and not undersample_training:
        print("Training set ratio of 0's to 1's (no oversampling or undersampling applied): ", calculate_0_1_label_ratio(training_list))

    if not oversample_validation and not undersample_validation:
        print("Validation set ratio of 0's to 1's (no oversampling or undersampling applied): ", calculate_0_1_label_ratio(validation_list))

    print("Test set ratio of 0's to 1's (no oversampling or undersampling applied): ", calculate_0_1_label_ratio(test_list))

    assert training_list != []
    assert validation_list != []
    assert test_list != []

    training_set = PreprocessedDataset(training_list)
    validation_set = PreprocessedDataset(validation_list)
    test_set = PreprocessedDataset(test_list)

    save_datasets(training_set, validation_set, test_set, pkl_training_filename, pkl_validation_filename, pkl_test_filename)

print("Size of the training set: ", len(training_set))

print("Size of the validation set: ", len(validation_set))

print("Size of the test set: ", len(test_set))

# Step 5: Define the neural network and train it
# Calculate the input size for the neural network
training_list = training_set.preprocessed_data
training_tuple = training_list[0]
input_list = list(training_tuple[:-1])
if len(input_list) == 1:
    input_size = input_list[0].shape[0]
    if double_nn:
        input_size_pos = input_size_neg = input_size // 2
else:
    input_size = sum([input_tensor.shape[1] for input_tensor in input_list])
    if double_nn:
        input_list_positive = [input_list[0]] + [input_list[i] for i in range(1, len(input_list)) if i % 2 != 0]
        input_list_negative = [input_list[0]] + [input_list[i] for i in range(1, len(input_list)) if i % 2 == 0]
        input_size_pos = sum([input_tensor.shape[1] for input_tensor in input_list_positive])
        input_size_neg = sum([input_tensor.shape[1] for input_tensor in input_list_negative])

# Initialize model and optimizer
if not double_nn:
    net_list = [AttributeClassifier(preprocessing=preprocessing, input_size=input_size, regularization=regularization, dropout_rate=dropout_rate).to(device)]
else:
    net_list = [AttributeClassifier(preprocessing=preprocessing, input_size=input_size_pos, regularization=regularization, dropout_rate=dropout_rate).to(device), 
                AttributeClassifier(preprocessing=preprocessing, input_size=input_size_neg, regularization=regularization, dropout_rate=dropout_rate).to(device)]
if regularization:
    optimizer_list = [optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay) for net in net_list]
else:
    optimizer_list = [optim.Adam(net.parameters(), lr=learning_rate) for net in net_list]


# Calculate the majority label and the ratio of the majority to the minority label in the training, validation, and test sets
training_set_maj_min_label_ratio = calculate_0_1_label_ratio(training_set.preprocessed_data)
if training_set_maj_min_label_ratio == 0:
    training_set_maj_min_label_ratio = 1/len(training_set.preprocessed_data)
if training_set_maj_min_label_ratio > 1:
    training_majority_label = 0
else:
    training_set_maj_min_label_ratio = 1/training_set_maj_min_label_ratio
    training_majority_label = 1

validation_set_maj_min_label_ratio = calculate_0_1_label_ratio(validation_set.preprocessed_data)
if validation_set_maj_min_label_ratio == 0:
    validation_set_maj_min_label_ratio = 1/len(validation_set.preprocessed_data)
if validation_set_maj_min_label_ratio > 1:
    validation_majority_label = 0
else:
    validation_set_maj_min_label_ratio = 1/validation_set_maj_min_label_ratio
    validation_majority_label = 1

test_set_maj_min_label_ratio = calculate_0_1_label_ratio(test_set.preprocessed_data)
if test_set_maj_min_label_ratio == 0:
    test_set_maj_min_label_ratio = 1/len(test_set.preprocessed_data)
if test_set_maj_min_label_ratio > 1:
    test_majority_label = 0
else:
    test_set_maj_min_label_ratio = 1/test_set_maj_min_label_ratio
    test_majority_label = 1

# Wrap dataset into DataLoader

if double_input:
    # Group the training_set into pairs and shuffle the pairs
    paired_training_set = [training_set.preprocessed_data[i:i+2] for i in range(0, len(training_set.preprocessed_data), 2)]
    random.shuffle(paired_training_set)

    # Flatten the shuffled pairs back into a single list
    shuffled_training_data = [item for pair in paired_training_set for item in pair]

    # Update the training_set with the shuffled data
    training_set.preprocessed_data = shuffled_training_data

    # Create the DataLoader
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=False) # shuffle is False because the data is already shuffled in pairs

else:
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Train the model
train_model(device, 
            net_list=net_list, optimizer_list=optimizer_list, 
            preprocessing=preprocessing, # Should the model use metrics (preprocessing = True) or raw embeddings (preprocessing = False)
            negative_concepts=negative_concepts, # Should the model inlude the use of negative concepts

            train_loader=train_loader,
            validation_loader=validation_loader, 
            test_loader=test_loader, 
            epochs=epochs, # amount of epochs used
            weighted_loss=True, # Whether to use weighted loss or not (as compensation for class imbalance)
            thresholds=[0.3,0.4,0.5,0.6,0.7],# Thresholds to apply to the output of the neural network to determine the prediction

            training_majority_label=training_majority_label, 
            training_set_maj_min_label_ratio=training_set_maj_min_label_ratio, 
            validation_majority_label=validation_majority_label, 
            validation_set_maj_min_label_ratio=validation_set_maj_min_label_ratio, 
            test_majority_label=test_majority_label, 
            test_set_maj_min_label_ratio=test_set_maj_min_label_ratio, 
            
            images_percentage=images_percentage, # Variables that determine the percentage of images and attributes to use
            attributes_percentage=attributes_percentage, # Both should be a divisor of 100
            oversample_training=False, 
            undersample_training=False, 
            oversample_validation=False,
            undersample_validation=False, 
            use_processed_concepts=use_processed_concepts, 
            split_attributes = split_attributes, # Variable that states if the attributes should be split into training, validation, and test sets and in which ratio if so
            attribute_list=attribute_list, # Give attribute names if want selection f.e ["aeroplane", "bicycle"] 
            double_input=double_input, # Give the positive concepts and negative concepts as separate datapoints to the neural network
            regularization=regularization # Using regularization or not in the neural network and optimizer

            )


