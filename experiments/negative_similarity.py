import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import clip
import os
import re
from sklearn.metrics import balanced_accuracy_score, f1_score
from collections import defaultdict
from PIL import Image

from datasets import CUB200Dataset, PreprocessedDataset, VOC2012Dataset, load_datasets, save_datasets
from sampling import calculate_0_1_label_ratio, oversample_minority_class, undersample_majority_class

device = "mps"  # Change to "cuda" if using NVIDIA GPU

device = "mps"
#device = torch_directml.device()  # Uses DirectML for AMD GPUs
print(f"Using device: {device}")

# ============ Define all parameters ===========================================================
preprocessing = True # Should the model use metrics (preprocessing = True) or raw embeddings (preprocessing = False)
negative_concepts = True # Should the model inlude the use of negative concepts
double_nn = False # Are positive and negative concepts inputted in 1 neural network or in 2 separate ones
double_input = False # Give the positive concepts and negative concepts as separate datapoints to the neural network

# The root_cub directory of the CUB200 dataset
root_cub = "./data/CUB_200_2011"
root_voc2012 = "./data/VOC2012"

# Variables that determine the percentage of images and attributes to use
# Both should be a divisor of 100
images_percentage = 100
attributes_percentage = 100

# Instead of using a percentage of the attributes, you can also specify which attributes 
# should be picked by providing a list with the numbers of the attributes (1 to 312)
attribute_list = None # e.g. [55, 56, 134]

use_processed_concepts = True # Variable that states if the processed concepts should be used

image_split_list = [0.8, 0.1, 0.1] # Variable that states in which ratio the images should be split into training, validation, and test sets

# Variable that states if the attributes should be split into training, validation, and test sets and in which ratio if so
split_attributes = True
attribute_split_list = [0.4, 0.3, 0.3]

# Variable that states if the dataset should be attempted to be loaded from disk
attempt_load = False

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
#dataset = VOC2012Dataset(root_voc2012, device, clip_model, preprocess, preprocessing, negative_concepts, images_percentage, attributes_percentage, 
#                        attribute_list, False, image_split_list, split_attributes, attribute_split_list, double_input)
dataset = CUB200Dataset(root_cub, device, clip_model, preprocess, preprocessing, negative_concepts, images_percentage, attributes_percentage, 
                      attribute_list, use_processed_concepts, image_split_list, split_attributes, attribute_split_list, double_input)


print("Size of the CUB200 dataset before preprocessing: ", dataset.size())

# Step 4: Preprocess dataset such that it can be used for training a neural network

# Define the strings for the file names
if preprocessing:
    type_string = "preprocessing"
else:
    type_string = "original"

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
    
else:
    # Preprocess the datasets
    training_list, validation_list, test_list = dataset.preprocess_dataset()
    print("Size of the CUB200 training set after preprocessing: ", len(training_list))
    print("Size of the CUB200 validation set after preprocessing: ", len(validation_list))
    print("Size of the CUB200 test set after preprocessing: ", len(test_list))

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


    assert training_list != []
    assert validation_list != []
    assert test_list != []

    training_set = PreprocessedDataset(training_list)
    validation_set = PreprocessedDataset(validation_list)
    test_set = PreprocessedDataset(training_list)
    preprocessed_data = training_set
    print(training_set)
save_datasets(training_set, validation_set, test_set, pkl_training_filename, pkl_validation_filename, pkl_test_filename)

combined_data = (
    training_set.preprocessed_data + 
    validation_set.preprocessed_data + 
    test_set.preprocessed_data
)
# Calculate metrics across all attributes (global metrics)
true_labels = []
predictions = []

for features, label in combined_data:
    pos_sim = features[2].item()  # Index 2 = positive cosine similarity
    neg_sim = features[3].item()  # Index 3 = negative cosine similarity
    marge1 = 0
    marge2 = 0.22
    pred = 1 if pos_sim > marge2 else 0
    true_labels.append(label)
    predictions.append(pred)

# Calculate metrics
f1 = f1_score(true_labels, predictions)
acc = balanced_accuracy_score(true_labels, predictions)
print(f"F1 Score: {f1:.4f}")
print(f"Balanced Accuracy: {acc:.4f}")
# Calculate balanced accuracy
tp = sum((t == 1 and p == 1) for t, p in zip(true_labels, predictions))
tn = sum((t == 0 and p == 0) for t, p in zip(true_labels, predictions))
fp = sum((t == 0 and p == 1) for t, p in zip(true_labels, predictions))
fn = sum((t == 1 and p == 0) for t, p in zip(true_labels, predictions))
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f"TPR: {sensitivity}, TNR: {specificity}")
balanced_acc = (sensitivity + specificity) / 2

# Print single result table
print("\n{:=^80}".format(" Global Metrics "))
print("{:<15} {:<15} {:<15} {:<15}".format(
    "F1 Score", "Balanced Acc", "Samples", "Total Pairs"))
print("{:<15.4f} {:<15.4f} {:<15} {:<15}".format(
    f1,
    balanced_acc,
    len(true_labels),
    len(combined_data)
))
print("="*80)