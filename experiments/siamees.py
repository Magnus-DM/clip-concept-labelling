import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import clip
import os
import re
from sklearn.metrics import f1_score
from collections import defaultdict
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np

from datasets import CUB200Dataset, PreprocessedDataset, VOC2012Dataset, load_datasets
from sampling import calculate_0_1_label_ratio, oversample_minority_class, undersample_majority_class


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=512, embedding_dim=256):
        """
        Dit netwerk heeft twee branches: één voor de afbeelding en één voor de concepten.
        Beide branches verwerken 512-dimensionale vectoren tot een embedding van grootte 'embedding_dim'.
        """
        super(SiameseNetwork, self).__init__()
        # Branch voor image input
        self.image_branch = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        # Branch voor concept input (gebruik dezelfde architectuur voor zowel positief als negatief)
        self.concept_branch = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, ima, concept):
        """
        Geeft de L2-genormaliseerde embeddings voor de image en de concept input.
        """
        image_emb = self.image_branch(ima)
        concept_emb = self.concept_branch(concept)
        # Normaliseer zodat de cosine similarity gelijk is aan de dot-product
        image_emb = F.normalize(image_emb, p=2, dim=1)
        concept_emb = F.normalize(concept_emb, p=2, dim=1)
        return image_emb, concept_emb

device = "cuda"
#device = torch_directml.device()  # Uses DirectML for AMD GPUs
print(f"Using device: {device}")

# ============ Define all parameters ===========================================================
preprocessing = False # Should the model use metrics (preprocessing = True) or raw embeddings (preprocessing = False)
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
attempt_load = True

# Variables that state if oversampling of the minority class/undersampling of the majority class should be applied
oversample_training = True
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
dataset2 = VOC2012Dataset(root_voc2012, device, clip_model, preprocess, preprocessing, negative_concepts, images_percentage, attributes_percentage, 
                        attribute_list, False, image_split_list, split_attributes, attribute_split_list, double_input)

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
    print("Training set ratio of 0's to 1's: ", calculate_0_1_label_ratio(training_set.preprocessed_data))
    print("Validation set ratio of 0's to 1's: ", calculate_0_1_label_ratio(validation_set.preprocessed_data))
    print("Test set ratio of 0's to 1's: ", calculate_0_1_label_ratio(test_set.preprocessed_data))
else:
    # Preprocess the datasets
    training_list, validation_list, test_list = dataset.preprocess_dataset()
    print("Size of the CUB200 training set after preprocessing: ", len(training_list))
    print("Size of the CUB200 validation set after preprocessing: ", len(validation_list))
    print("Size of the CUB200 test set after preprocessing: ", len(test_list))
    training_list2, validation_list2, test_list2 = dataset2.preprocess_dataset()

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
    test_set = PreprocessedDataset(training_list)
    training_set2 = PreprocessedDataset(training_list2)
    validation_set2 = PreprocessedDataset(validation_list2)
    test_set2 = PreprocessedDataset(training_list2)

 
 # Maak trainings-, validatie- en testdata aan

train_data = training_set.preprocessed_data

val_data   = validation_set.preprocessed_data
test_data  = test_set.preprocessed_data

CUB_data = (
    training_set2.preprocessed_data + 
    validation_set2.preprocessed_data + 
    test_set2.preprocessed_data
)
VOC_data = (
    training_set.preprocessed_data + 
    validation_set.preprocessed_data + 
    test_set.preprocessed_data
)





class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=512, embedding_dim=256):
        super(SiameseNetwork, self).__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        self.concept_branch = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward_image(self, x):
        x = self.image_branch(x)
        return F.normalize(x, p=2, dim=1)

    def forward_concept(self, x):
        x = self.concept_branch(x)
        return F.normalize(x, p=2, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
margin = 0.2  
num_epochs = 10


for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch in VOC_data:
        images, pos_concepts, neg_concepts, labels = batch
        

        img_emb = model.forward_image(images)
        pos_emb = model.forward_concept(pos_concepts)
        neg_emb = model.forward_concept(neg_concepts)
        

        sim_pos = F.cosine_similarity(img_emb, pos_emb)
        sim_neg = F.cosine_similarity(img_emb, neg_emb)
        
        loss_positive = F.relu(margin + sim_neg - sim_pos)
        loss_negative = F.relu(margin + sim_pos - sim_neg)
        
 
        loss = (labels * loss_positive + (1 - labels) * loss_negative).mean()
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_data):.4f}")


    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_data:
            images, pos_concepts, neg_concepts, labels = batch
            
            img_emb = model.forward_image(images)
            pos_emb = model.forward_concept(pos_concepts)
            neg_emb = model.forward_concept(neg_concepts)
            
            sim_pos = F.cosine_similarity(img_emb, pos_emb)
            sim_neg = F.cosine_similarity(img_emb, neg_emb)
            
            loss_positive = F.relu(margin + sim_neg - sim_pos)
            loss_negative = F.relu(margin + sim_pos - sim_neg)
            loss = (labels * loss_positive + (1 - labels) * loss_negative).mean()
            
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1} - Val Loss: {val_loss/len(val_data):.4f}")


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for sample in CUB_data:
        ima = sample[0]    
        pos = sample[1]    
        neg = sample[2]    
        label = sample[3]  
        

        img_emb = model.forward_image(ima)
        pos_emb = model.forward_concept(pos)
        neg_emb = model.forward_concept(neg)
        

        sim_pos = F.cosine_similarity(img_emb, pos_emb)
        sim_neg = F.cosine_similarity(img_emb, neg_emb)
        

        pred = 1 if sim_pos.item() > sim_neg.item() else 0

        all_preds.append(pred)
        all_labels.append(label) 


TP = sum((np.array(all_labels) == 1) & (np.array(all_preds) == 1))
TN = sum((np.array(all_labels) == 0) & (np.array(all_preds) == 0))
FP = sum((np.array(all_labels) == 0) & (np.array(all_preds) == 1))
FN = sum((np.array(all_labels) == 1) & (np.array(all_preds) == 0))

sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
balanced_acc = (sensitivity + specificity) / 2

print(f"Balanced Accuracy: {balanced_acc:.4f}")