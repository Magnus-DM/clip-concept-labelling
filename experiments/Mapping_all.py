import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import clip
import os
import pickle
import torch_directml
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from PIL import Image
import random
import matplotlib.pyplot as plt
import re
from sklearn.metrics import f1_score

class PreprocessedDataset(Dataset):
    def __init__(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]

class CUB200Dataset(Dataset):
    def __init__(self, root, device, model, preprocess, preprocessing, negative_concepts, 
                 images_percentage, attributes_percentage, attribute_list=None, 
                 use_processed_concepts=False, image_split_list=[0.8, 0.1, 0.1], 
                 split_attributes=False, attribute_split_list=[0.8, 0.1, 0.1],
                 double_input=False):

        self.root = root
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.preprocessing = preprocessing
        self.negative_concepts = negative_concepts
        self.image_split_list = image_split_list
        self.split_attributes = split_attributes
        self.attribute_split_list = attribute_split_list
        self.double_input = double_input

        self.image_dir = os.path.join(root, "images")
        self.attribute_file = os.path.join(root, "attributes", "image_attribute_labels.txt")
        self.image_file = os.path.join(root, "images.txt")
        self.attribute_names_file = os.path.join(root, "attributes", "attributes.txt")
        self.processed_attribute_names_file = os.path.join(root, "attributes", "processed_attributes.txt")
        self.negative_processed_attribute_names_file = os.path.join(root, "attributes", "negative_processed_attributes.txt")

        # Load image paths
        images_divider = 100 // images_percentage
        i = 0
        self.image_ids = []
        self.image_paths = []
        with open(self.image_file, "r") as f:
            for line in f:
                if i % images_divider == 0:
                    parts = line.strip().split()
                    image_id = int(parts[0])
                    image_rel_path = parts[1]
                    self.image_ids.append(image_id)
                    self.image_paths.append(os.path.join(self.image_dir, image_rel_path))
                i += 1

        # Load attribute names and attribute values
        attributes_divider = 100 // attributes_percentage
        self.all_attribute_names = []
        self.attribute_names = []
        self.attribute_numbers = []
        self.processed_attribute_names = []
        self.negative_processed_attribute_names = []

        with open(self.attribute_names_file, "r") as f:
            lines = f.readlines()
        self.all_attribute_names = [line.strip().split()[1] for line in lines]

        with open(self.attribute_names_file, "r") as f:
            lines = f.readlines()
        i = 0
        for line in lines:
            if (attribute_list is None and i % attributes_divider == 0) or (attribute_list is not None and i in attribute_list):
                parts = line.strip().split()
                self.attribute_numbers.append(int(parts[0]))
                self.attribute_names.append(parts[1])
            i += 1

        with open(self.processed_attribute_names_file, "r") as f:
            lines = f.readlines()
        i = 0
        for line in lines:
            if (attribute_list is None and i % attributes_divider == 0) or (attribute_list is not None and i in attribute_list):
                self.processed_attribute_names.append(re.sub(r'^\d+\.\s*', '', line.strip()))
            i += 1

        self.used_attribute_names = self.processed_attribute_names if use_processed_concepts else self.attribute_names

        with open(self.negative_processed_attribute_names_file, "r") as f:
            lines = f.readlines()
        i = 0
        for line in lines:
            if (attribute_list is None and i % attributes_divider == 0) or (attribute_list is not None and i in attribute_list):
                self.negative_processed_attribute_names.append(re.sub(r'^\d+\.\s*', '', line.strip()))
            i += 1

        self.image_attributes = {}
        with open(self.attribute_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                image_id = int(parts[0])
                attribute_id = int(parts[1])
                is_present = int(parts[2])
                if image_id in self.image_ids and attribute_id in self.attribute_numbers:
                    if image_id not in self.image_attributes:
                        self.image_attributes[image_id] = [-1] * len(self.all_attribute_names)
                    self.image_attributes[image_id][attribute_id - 1] = is_present

        # For each list in the dictionary, remove the -1 values
        for key in self.image_attributes:
            self.image_attributes[key] = [x for x in self.image_attributes[key] if x != -1]

        # print("Attributes: ", self.attribute_names)
        # print("--------------------")
        # print("Processed Attributes: ", self.processed_attribute_names)

    def size(self):
        """
        Returns the size of the dataset
        """
        return len(self.image_paths)

    def get_image_with_labels(self, idx):
        """
        Returns the image and its attributes at the given index
        """
        image_id = self.image_ids[idx]
        image_path = self.image_paths[idx]
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        attributes = self.image_attributes[image_id]
        return image, attributes
    
    def calculate_label_ratio(self):
        """
        Calculates the ratio of 0s to 1s in the dataset
        """
        count0 = 0
        count1 = 0
        for idx in range(self.size()):
            _, labels = self.get_image_with_labels(idx)
            for label in labels:
                if label == 0:
                    count0 += 1
                else:
                    count1 += 1
        return count0/count1
    
    def preprocess_dataset(self):
        """
        Preprocesses the dataset to create a set of tuples containing image encodings, attribute encodings, and labels
        """
        training_list = []
        validation_list = []
        test_list = []
        self.attribute_encodings = []
        self.negative_attribute_encodings = []
        for attr in self.used_attribute_names:
            self.attribute_encodings.append(get_clip_text_embedding(attr, self.model, self.device).float())
        for attr in self.negative_processed_attribute_names:
            self.negative_attribute_encodings.append(get_clip_text_embedding(attr, self.model, self.device).float())
        if self.split_attributes:
            attribute_test_size_1 = self.attribute_split_list[1] + self.attribute_split_list[2]
            attribute_test_size_2 = round(self.attribute_split_list[2] / attribute_test_size_1,1)
            attribute_indices = list(range(len(self.attribute_numbers)))
            attribute_train_indices, attribute_no_train_indices = train_test_split(attribute_indices, test_size=attribute_test_size_1, random_state=69)
            attribute_validation_indices, attribute_test_indices = train_test_split(attribute_no_train_indices, test_size=attribute_test_size_2, random_state=69)
        else:
            attribute_train_indices = []
            attribute_validation_indices = []
            attribute_test_indices = []

        print("Preprocessing dataset...")
        print("Encoding all images...")
        image_test_size_1 = self.image_split_list[1] + self.image_split_list[2]
        image_test_size_2 = round(self.image_split_list[2] / image_test_size_1,1)
        image_indices = list(range(self.size()))
        image_train_indices, image_no_train_indices = train_test_split(image_indices, test_size=image_test_size_1, random_state=69)
        image_validation_indices, image_test_indices = train_test_split(image_no_train_indices, test_size=image_test_size_2, random_state=69)
        print("Amount of training images: ", len(image_train_indices))
        print("Amount of validation images: ", len(image_validation_indices))
        print("Amount of test images: ", len(image_test_indices))
        if attribute_train_indices != [] and attribute_validation_indices != [] and attribute_test_indices != []:
            print("Amount of training attributes: ", len(attribute_train_indices))
            print("Amount of validation attributes: ", len(attribute_validation_indices))
            print("Amount of test attributes: ", len(attribute_test_indices))
        else:
            print("Amount of training attributes: ", len(self.attribute_numbers))
            print("Amount of validation attributes: ", len(self.attribute_numbers))
            print("Amount of test attributes: ", len(self.attribute_numbers))


        for idx in range(self.size()):
            if idx % 100 == 0:
                print("Currently at index ",idx)
            image, labels = self.get_image_with_labels(idx)
            image_encoding = get_clip_image_embedding(image, self.model).float()
            for attr_idx in range(len(self.attribute_numbers)):
                concept_encoding = self.attribute_encodings[attr_idx]
                negative_concept_encoding = self.negative_attribute_encodings[attr_idx]
                label = labels[attr_idx]
                if not self.preprocessing and not self.negative_concepts:
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((image_encoding, concept_encoding, label))
                    elif idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((image_encoding, concept_encoding, label))
                    elif idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((image_encoding, concept_encoding, label))
                elif self.preprocessing and not self.negative_concepts:
                    norm_text = torch.norm(concept_encoding)
                    norm_image = torch.norm(image_encoding)
                    norm_diff = torch.norm(image_encoding - concept_encoding)
                    cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, concept_encoding, dim=1)
                    angle = torch.acos(cosine_similarity)
                    dot_product = torch.dot(image_encoding.squeeze(), concept_encoding.squeeze())
                    inputs = torch.tensor([norm_text, norm_image, norm_diff, angle, dot_product])
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((inputs, label))
                    elif idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((inputs, label))
                    elif idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((inputs, label))
                elif not self.preprocessing and self.negative_concepts and not self.double_input:
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
                    elif idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
                    elif idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
                elif not self.preprocessing and self.negative_concepts and self.double_input:
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((image_encoding, concept_encoding, label))
                        training_list.append((image_encoding, negative_concept_encoding, 1 - label))
                    elif idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((image_encoding, concept_encoding, label))
                        validation_list.append((image_encoding, negative_concept_encoding, 1 - label))
                    elif idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((image_encoding, concept_encoding, label))
                        test_list.append((image_encoding, negative_concept_encoding, 1 - label))
                elif self.preprocessing and self.negative_concepts and not self.double_input:
                    norm_diff = torch.norm(image_encoding - concept_encoding)
                    negative_norm_diff = torch.norm(image_encoding - negative_concept_encoding)
                    cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, concept_encoding, dim=1)
                    negative_cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, negative_concept_encoding, dim=1)
                    angle = torch.acos(cosine_similarity)
                    negative_angle = torch.acos(negative_cosine_similarity)
                    dot_product = torch.dot(image_encoding.squeeze(), concept_encoding.squeeze())
                    negative_dot_product = torch.dot(image_encoding.squeeze(), negative_concept_encoding.squeeze())
                    inputs = torch.tensor([norm_diff, negative_norm_diff, cosine_similarity, negative_cosine_similarity, angle, negative_angle, dot_product, negative_dot_product])
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((inputs, label))
                    elif idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((inputs, label))
                    elif idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((inputs, label))
                elif self.preprocessing and self.negative_concepts and self.double_input:
                    norm_diff = torch.norm(image_encoding - concept_encoding)
                    negative_norm_diff = torch.norm(image_encoding - negative_concept_encoding)
                    cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, concept_encoding, dim=1)
                    negative_cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, negative_concept_encoding, dim=1)
                    angle = torch.acos(cosine_similarity)
                    negative_angle = torch.acos(negative_cosine_similarity)
                    dot_product = torch.dot(image_encoding.squeeze(), concept_encoding.squeeze())
                    negative_dot_product = torch.dot(image_encoding.squeeze(), negative_concept_encoding.squeeze())
                    inputs_pos = torch.tensor([norm_diff, cosine_similarity, angle, dot_product])
                    inputs_neg = torch.tensor([negative_norm_diff, negative_cosine_similarity, negative_angle, negative_dot_product])
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((inputs_pos, label))
                        training_list.append((inputs_neg, 1 - label))
                    elif idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((inputs_pos, label))
                        validation_list.append((inputs_neg, 1 - label))
                    elif idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((inputs_pos, label))
                        test_list.append((inputs_neg, 1 - label))
        return training_list, validation_list, test_list
    
class AttributeClassifier(nn.Module):
    def __init__(self, preprocessing, input_size, regularization, dropout_rate):
        super(AttributeClassifier, self).__init__()
        if not preprocessing:
            self.fc1 = nn.Linear(input_size, 512)
            if regularization:
                self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(512, 256)
            if regularization:
                self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(256, 1)
            self.activation = nn.Sigmoid()
        elif preprocessing:
            self.fc1 = nn.Linear(input_size, 10)
            if regularization:
                self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(10, 10)
            if regularization:
                self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(10, 1)
            self.activation = nn.Sigmoid()

    def forward(self, input_list, regularization):
        if len(input_list) == 1:
            x = torch.relu(self.fc1(input_list[0]))
            if regularization:
                x = self.dropout1(x)  # Apply dropout
            x = torch.relu(self.fc2(x))
            if regularization:
                x = self.dropout2(x)  # Apply dropout
            x = self.activation(self.fc3(x))
            return x
        else:
            x = input_list[0]
            for i in range(1, len(input_list)):
                x = torch.cat((x, input_list[i]), dim=2)
            x = torch.relu(self.fc1(x))
            if regularization:
                x = self.dropout1(x)  # Apply dropout
            x = torch.relu(self.fc2(x))
            if regularization:
                x = self.dropout2(x)  # Apply dropout
            x = self.activation(self.fc3(x))
            return x

    # def forward(self, input_list):
    #     if len(input_list) == 1:
    #         x = torch.relu(self.fc1(input_list[0]))
    #         x = torch.relu(self.fc2(x))
    #         x = self.activation(self.fc3(x))
    #         return x
    #     else:
    #         x = input_list[0]
    #         for i in range(1, len(input_list)):
    #             x = torch.cat((x, input_list[i]), dim=2)
    #         x = torch.relu(self.fc1(x))
    #         x = torch.relu(self.fc2(x))
    #         x = self.activation(self.fc3(x))
    #         return x

def get_clip_image_embedding(image, model):
    """
    Returns the image embedding from the CLIP model
    """
    with torch.no_grad():
        image_features = model.encode_image(image)
        return image_features
    
def get_clip_text_embedding(text, model, device):
    """
    Returns the text embedding from the CLIP model
    """
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([text]).to(device))
        return text_features
    
def calculate_0_1_label_ratio(dataset_list):
    """
    Calculates the ratio of 0s to 1s in the dataset
    """
    count0 = 0
    count1 = 0
    for tuple in dataset_list:
        if tuple[-1] == 0:
            count0 += 1
        else:
            count1 += 1
    if count0 == 0 or count1 == 0:
        print(f"Warning: count0 is {count0} and count1 is {count1}")
    return count0/count1 if count1 > 0 else count0

def load_datasets(training_file_path, validation_file_path, test_file_path):
    """
    Loads the datasets from disk
    """
    with open(training_file_path, 'rb') as f:
        training_set = pickle.load(f)
    with open(validation_file_path, 'rb') as f:
        validation_set = pickle.load(f)
    with open(test_file_path, 'rb') as f:
        test_set = pickle.load(f)
    return training_set, validation_set, test_set

def save_datasets(training_set, validation_set, test_set, training_file_path, validation_file_path, test_file_path):
    """
    Saves the datasets to disk, creating directories if they don't exist
    """
    os.makedirs(os.path.dirname(training_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(validation_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

    with open(training_file_path, 'wb') as f:
        pickle.dump(training_set, f)
    with open(validation_file_path, 'wb') as f:
        pickle.dump(validation_set, f)
    with open(test_file_path, 'wb') as f:
        pickle.dump(test_set, f)

def oversample_minority_class(dataset_list):
    """
    Oversamples the minority class in the dataset list
    """
    majority_minority_label_ratio = calculate_0_1_label_ratio(dataset_list)
    print("Ratio of 0s to 1s in the training set: ", majority_minority_label_ratio)

    if majority_minority_label_ratio > 1:
        majority_label = 0
    else:
        majority_minority_label_ratio = 1/majority_minority_label_ratio
        majority_label = 1

    # Oversample the minority class in the dataset list
    dataset_list_copy = list(dataset_list)
    for data_tuple in dataset_list_copy:
        if data_tuple[-1] != majority_label:
            for i in range(int(majority_minority_label_ratio)):
                dataset_list.append(data_tuple)

    random.shuffle(dataset_list)
    return dataset_list

def undersample_majority_class(dataset_list):
    """
    Undersamples the majority class in the dataset list
    """
    majority_minority_label_ratio = calculate_0_1_label_ratio(dataset_list)
    print("Ratio of 0s to 1s in the training set: ", majority_minority_label_ratio)

    if majority_minority_label_ratio > 1:
        majority_label = 0
    else:
        majority_minority_label_ratio = 1/majority_minority_label_ratio
        majority_label = 1

    # Undersample the majority class in the dataset list
    dataset_list_copy = []
    for data_tuple in dataset_list:
        if data_tuple[-1] == majority_label:
            if random.randint(1, int(majority_minority_label_ratio)) == 1:
                dataset_list_copy.append(data_tuple)
        else:
            dataset_list_copy.append(data_tuple)
    dataset_list = dataset_list_copy
    random.shuffle(dataset_list)
    return dataset_list

def f1_score_manual(labels, predictions):
    """
    Calculates the F1 score
    """
    labels_tensor = torch.tensor(labels)
    predictions_tensor = torch.tensor(predictions)
    
    label_1_prediction_1 = torch.sum((labels_tensor == 1) & (predictions_tensor == 1)).item()
    label_1_prediction_0 = torch.sum((labels_tensor == 1) & (predictions_tensor == 0)).item()
    label_0_prediction_1 = torch.sum((labels_tensor == 0) & (predictions_tensor == 1)).item()

    
    if label_1_prediction_1 + label_0_prediction_1 == 0:
        precision = 0
    else:
        precision = label_1_prediction_1 / (label_1_prediction_1 + label_0_prediction_1)

    if label_1_prediction_1 + label_1_prediction_0 == 0:
        recall = 0
    else:
        recall = label_1_prediction_1 / (label_1_prediction_1 + label_1_prediction_0)

    if precision + recall == 0:
        f1 =  0
    else: 
        f1 = 2 * (precision * recall) / (precision + recall)

    sklearn_f1 = f1_score(labels, predictions)
    if round(f1, 4) != round(sklearn_f1, 4):
        print("Warning: Manual F1 score and Sklearn F1 score are not equal")
        print("Manual F1 score: ", round(f1, 4))
        print("Sklearn F1 score: ", round(sklearn_f1, 4))

    return round(sklearn_f1, 4)

def calculate_double_nn_scores(double_nn_output_dict, all_original_labels=None, thresholds=None):
    """
    Calculates the scores for the double NN model
    """
    if thresholds is not None:
        double_nn_scores_dict = {}
        prediction_dict = {}
        for threshold in thresholds:
            double_nn_scores_dict[threshold] = {}
            prediction_dict[threshold] = []
    else:
        double_nn_scores_dict = None
        prediction_dict = None

    average_output = [0] * len(double_nn_output_dict[0])
    for net_index in range(len(double_nn_output_dict)):
        if net_index % 2 == 0:
            average_output = [average_output[i] + double_nn_output_dict[net_index][i]/len(double_nn_output_dict) for i in range(len(average_output))]
        else:
            average_output = [average_output[i] + (1 - double_nn_output_dict[net_index][i])/len(double_nn_output_dict) for i in range(len(average_output))]

    if thresholds is not None:
        for threshold in thresholds:
            threshold_output = torch.where(torch.tensor(average_output) > threshold, torch.tensor(1.0), torch.tensor(0.0))
            threshold_output = threshold_output.squeeze()
            prediction_dict[threshold] = threshold_output.cpu().numpy()
            if all_original_labels is not None:
                label_1_prediction_1 = 0
                label_1_prediction_0 = 0
                label_0_prediction_1 = 0
                label_0_prediction_0 = 0
                for label, prediction in zip(all_original_labels, threshold_output):
                    if label == 1 and prediction == 1:
                        label_1_prediction_1 += 1
                    elif label == 1 and prediction == 0:
                        label_1_prediction_0 += 1
                    elif label == 0 and prediction == 1:
                        label_0_prediction_1 += 1
                    elif label == 0 and prediction == 0:
                        label_0_prediction_0 += 1

                accuracy = (label_1_prediction_1 + label_0_prediction_0) / (
                    label_1_prediction_1 + label_1_prediction_0 + label_0_prediction_1 + label_0_prediction_0
                )
                accuracy_0 = label_0_prediction_0 / (label_0_prediction_0 + label_0_prediction_1) if (label_0_prediction_0 + label_0_prediction_1) > 0 else 0
                accuracy_1 = label_1_prediction_1 / (label_1_prediction_1 + label_1_prediction_0) if (label_1_prediction_1 + label_1_prediction_0) > 0 else 0
                balanced_accuracy = (accuracy_0 + accuracy_1) / 2
                f1 = f1_score_manual(all_original_labels, threshold_output.cpu().numpy())

                double_nn_scores_dict[threshold]['accuracy'] = accuracy
                double_nn_scores_dict[threshold]['accuracy_0'] = accuracy_0
                double_nn_scores_dict[threshold]['accuracy_1'] = accuracy_1
                double_nn_scores_dict[threshold]['balanced_accuracy'] = balanced_accuracy
                double_nn_scores_dict[threshold]['f1'] = f1

    return average_output, prediction_dict, double_nn_scores_dict

def plot_losses(net_list, net_index, losses_list, label_list, title, filename, weighted_loss, 
                images_percentage, attributes_percentage, oversample_training, 
                undersample_training, oversample_validation, undersample_validation, 
                use_processed_concepts, preprocessing, negative_concepts, split_attributes, 
                attribute_list, double_input):
    """
    Plots the losses over epochs
    """
    if preprocessing:
        type_string = "preprocessing"
    else:
        type_string = "original"

    double_nn_string = ""

    if negative_concepts:
        type_string += "_negative"

        if len(net_list) > 1:
            type_string += "_double_nn"

            if net_index % 2 == 0:
                double_nn_string = "_positive"
            else:
                double_nn_string = "_negative"

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

    weight_sampling_string = ""
    if weighted_loss:
        weight_sampling_string += "weighted_loss_"
    if oversample_training:
        weight_sampling_string += "over_train_"
    if undersample_training:
        weight_sampling_string += "under_train_"
    if oversample_validation:
        weight_sampling_string += "over_val_"
    if undersample_validation:
        weight_sampling_string += "under_val_"

    if weight_sampling_string == "":
        weight_sampling_string = "nothing_"

    if attribute_list is None:
        percentage_string = f"{images_percentage}_{attributes_percentage}{double_nn_string}"
    else:
        percentage_string = f"custom_attributes{double_nn_string}"


    dir_path = f"plots_{type_string}/{split_attributes_string}/{processed_concepts_string}concepts/{weight_sampling_string}{percentage_string}"
    os.makedirs(dir_path, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    for i, losses in enumerate(losses_list):
        plt.plot(losses, label=label_list[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(dir_path, filename))
    plt.close()

def plot_metric(net_list, net_index, thresholds, metric_dict, metric_name, label, title, filename, 
                weighted_loss, images_percentage, attributes_percentage, oversample_training, 
                undersample_training, oversample_validation, undersample_validation,
                use_processed_concepts, preprocessing, negative_concepts, split_attributes, 
                attribute_list, double_input):
    """
    Plots the metric over epochs
    """
    if preprocessing:
        type_string = "preprocessing"
    else:
        type_string = "original"

    double_nn_string = ""

    if negative_concepts:
        type_string += "_negative"

        if len(net_list) > 1:
            type_string += "_double_nn"

            if net_index % 2 == 0:
                double_nn_string = "_positive"
            else:
                double_nn_string = "_negative"

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

    weight_sampling_string = ""
    if weighted_loss:
        weight_sampling_string += "weighted_loss_"
    if oversample_training:
        weight_sampling_string += "over_train_"
    if undersample_training:
        weight_sampling_string += "under_train_"
    if oversample_validation:
        weight_sampling_string += "over_val_"
    if undersample_validation:
        weight_sampling_string += "under_val_"

    if weight_sampling_string == "":
        weight_sampling_string = "nothing_"

    if attribute_list is None:
        percentage_string = f"{images_percentage}_{attributes_percentage}{double_nn_string}"
    else:
        percentage_string = f"custom_attributes{double_nn_string}"

    dir_path = f"plots_{type_string}/{split_attributes_string}/{processed_concepts_string}concepts/{weight_sampling_string}{percentage_string}"
    os.makedirs(dir_path, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    for threshold in thresholds:
        plt.plot(metric_dict[threshold], label=f'{label} (Threshold {threshold})')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(dir_path, filename))
    plt.close()

def plot_metrics(net_list, net_index, thresholds, metric_dicts, metric_names, labels, titles, filenames,
                weighted_loss, images_percentage, attributes_percentage, oversample_training, 
                undersample_training, oversample_validation, undersample_validation,
                use_processed_concepts, preprocessing, negative_concepts, split_attributes, 
                attribute_list, double_input):
    """
    Plots the metrics over epochs
    """
    for i in range(len(metric_dicts)):
        plot_metric(net_list, net_index, thresholds, metric_dicts[i], metric_names[i], labels[i], titles[i], filenames[i],
                    weighted_loss, images_percentage, attributes_percentage, oversample_training, 
                    undersample_training, oversample_validation, undersample_validation,
                    use_processed_concepts, preprocessing, negative_concepts, split_attributes, 
                    attribute_list, double_input)
        
def plot_double_nn_metrics(double_nn_scores_evolution_dict, weighted_loss, images_percentage, attributes_percentage, 
                           oversample_training, undersample_training, oversample_validation, undersample_validation, 
                           use_processed_concepts, preprocessing, split_attributes, thresholds, attribute_list):
    """
    Plots the metrics for the double NN model
    """

    double_nn_metrics_dict = {type_key: {metric_key: {threshold_key: [] for threshold_key in thresholds} for metric_key in double_nn_scores_evolution_dict[type_key][0][thresholds[0]].keys()} for type_key in double_nn_scores_evolution_dict}

    for type_key in double_nn_scores_evolution_dict:  # keys are training, validation, test
        for epoch_key in double_nn_scores_evolution_dict[type_key]:  # keys are epochs
            for threshold_key in double_nn_scores_evolution_dict[type_key][epoch_key]:  # keys are thresholds
                for metric_key in double_nn_scores_evolution_dict[type_key][epoch_key][threshold_key]:  # keys are metrics
                    metric = double_nn_scores_evolution_dict[type_key][epoch_key][threshold_key][metric_key]
                    double_nn_metrics_dict[type_key][metric_key][threshold_key].append(metric)

                    if preprocessing:
                        type_string = "preprocessing"
                    else:
                        type_string = "original"

                    type_string += "_negative_double_nn"

                    if split_attributes:
                        split_attributes_string = "split_on_attributes"
                    else:
                        split_attributes_string = "no_split_on_attributes"

                    if use_processed_concepts:
                        processed_concepts_string = "processed_"
                    else:
                        processed_concepts_string = "original_"

                    weight_sampling_string = ""
                    if weighted_loss:
                        weight_sampling_string += "weighted_loss_"
                    if oversample_training:
                        weight_sampling_string += "over_train_"
                    if undersample_training:
                        weight_sampling_string += "under_train_"
                    if oversample_validation:
                        weight_sampling_string += "over_val_"
                    if undersample_validation:
                        weight_sampling_string += "under_val_"

                    if weight_sampling_string == "":
                        weight_sampling_string = "nothing_"

                    if attribute_list is None:
                        percentage_string = f"{images_percentage}_{attributes_percentage}"
                    else:
                        percentage_string = "custom_attributes"

                    dir_path = f"plots_{type_string}/{split_attributes_string}/{processed_concepts_string}concepts/{weight_sampling_string}{percentage_string}"
                    os.makedirs(dir_path, exist_ok=True)

    for type_key in double_nn_metrics_dict:
        for metric_key in double_nn_metrics_dict[type_key]:
            plt.figure(figsize=(10, 5))
            for threshold_key in thresholds:
                plt.plot(double_nn_metrics_dict[type_key][metric_key][threshold_key], label=f'Threshold {threshold_key}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_key.capitalize())
            plt.legend()
            plt.title(f'{type_key.capitalize()} {metric_key.capitalize()} Over Epochs')
            plt.savefig(os.path.join(dir_path, f'{type_key}_{metric_key}.png'))
            plt.close()

def test_model(net_list, net_index, test_loader, weighted_loss=False, majority_label=0, set_maj_min_label_ratio=1, 
               thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], preprocessing=False, double_nn_output_dict=None, double_input=False,
               regularization=False):
    """
    Tests the model on the test data (or validation data)
    """
    all_original_labels = [] if len(net_list) > 1 else None
    net_list[net_index].eval()
    total_loss = 0
    total_unweighted_loss = 0
    total_items = 0
    all_labels = []
    all_predictions = {}
    total_differences = {}
    total_label_1_prediction_1 = {}
    total_label_1_prediction_0 = {}
    total_label_0_prediction_1 = {}
    total_label_0_prediction_0 = {}
    accuracy_0 = {}
    accuracy_1 = {}
    accuracy = {}
    f1 = {}
    for threshold in thresholds:
        all_predictions[threshold] = []
        total_differences[threshold] = 0
        total_label_1_prediction_1[threshold] = 0
        total_label_1_prediction_0[threshold] = 0
        total_label_0_prediction_1[threshold] = 0
        total_label_0_prediction_0[threshold] = 0

    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(test_loader):
            input_list = list(data_tuple[:-1])
            if len(net_list) > 1:
                if not preprocessing:
                    input_list = [input_list[0]] + [input_list[i] for i in range(1, len(input_list)) if (i + net_index) % 2 != 0]
                else:
                    input_tensor = input_list[0]
                    input_tensor = input_tensor[:, [(i + net_index) % 2 == 0 for i in range(input_tensor.shape[1])]]
                    input_list = [input_tensor]
            input_list = [tensor.to(device) for tensor in input_list]
            label = data_tuple[-1]

            if len(net_list) > 1:
                all_original_labels.extend(label.cpu().numpy())
            if net_index % 2 != 0:
                label = 1 - label
            input_list = [tensor.to(device) for tensor in input_list]
            label = label.to(device)
            label = label.float().squeeze()
            output = net_list[net_index](input_list, regularization)
            output = output.squeeze()
            output = output.to(device)

            if double_input:
                if isinstance(output, torch.Tensor):
                    output_joint = (output[::2] + (1 - output[1::2])) / 2
                    output_joint = output_joint.squeeze()
                    output_joint = output_joint.to(device)
                else:
                    raise ValueError("output is not a tensor")

            if len(net_list) > 1:
                double_nn_output_dict[net_index].extend(output.detach().cpu().numpy())
            threshold_output = {}
            for threshold in thresholds:
                if double_input:
                    threshold_output[threshold] = torch.where(output_joint > threshold, torch.tensor(1.0, device=output.device), torch.tensor(0.0, device=output.device))
                    threshold_output[threshold] = threshold_output[threshold].squeeze()
                else:
                    threshold_output[threshold] = torch.where(output > threshold, torch.tensor(1.0, device=output.device), torch.tensor(0.0, device=output.device))
                    threshold_output[threshold] = threshold_output[threshold].squeeze()
            
            if weighted_loss:
                weight = torch.where(label == majority_label, torch.tensor(1.0, device=device), torch.tensor(set_maj_min_label_ratio, device=device))
                loss = nn.BCELoss(weight=weight)(output, label)
            else:
                loss = nn.BCELoss()(output, label)
            unweighted_loss = nn.BCELoss()(output, label)
            total_loss += loss.item()
            total_unweighted_loss += unweighted_loss.item()

            if double_input:
                if isinstance(label, torch.Tensor):
                    label = label[::2]
                    label = label.float().squeeze()
                    label = label.to(device)
                else:
                    raise ValueError("label is not a tensor")
            
            # Calculate the number of differences
            differences = {}
            for threshold in thresholds:
                differences[threshold] = torch.sum(threshold_output[threshold] != label)
                total_differences[threshold] += differences[threshold].item()

            # Calculate TN, FP, FN, TP for each threshold:
            label_1_prediction_1 = {}
            label_1_prediction_0 = {}
            label_0_prediction_1 = {}
            label_0_prediction_0 = {}
            for threshold in thresholds:
                label_1_prediction_1[threshold] = torch.sum((label == 1) & (threshold_output[threshold] == 1))
                label_1_prediction_0[threshold] = torch.sum((label == 1) & (threshold_output[threshold] == 0))
                label_0_prediction_1[threshold] = torch.sum((label == 0) & (threshold_output[threshold] == 1))
                label_0_prediction_0[threshold] = torch.sum((label == 0) & (threshold_output[threshold] == 0))
                total_label_1_prediction_1[threshold] += label_1_prediction_1[threshold].item()
                total_label_1_prediction_0[threshold] += label_1_prediction_0[threshold].item()
                total_label_0_prediction_1[threshold] += label_0_prediction_1[threshold].item()
                total_label_0_prediction_0[threshold] += label_0_prediction_0[threshold].item()

            total_items += label.numel()
            
            # Collect all labels and predictions for F1 score calculation
            all_labels.extend(label.cpu().numpy())

            for threshold in thresholds:
                all_predictions[threshold].extend(threshold_output[threshold].cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    avg_unweighted_loss = total_unweighted_loss / len(test_loader)

    for threshold in thresholds:
        accuracy[threshold] = 1 - (total_differences[threshold] / total_items)
        accuracy_0[threshold] = total_label_0_prediction_0[threshold] / (total_label_0_prediction_0[threshold] + total_label_0_prediction_1[threshold]) \
            if (total_label_0_prediction_0[threshold] + total_label_0_prediction_1[threshold]) > 0 else 0
        accuracy_1[threshold] = total_label_1_prediction_1[threshold] / (total_label_1_prediction_1[threshold] + total_label_1_prediction_0[threshold]) \
            if (total_label_1_prediction_1[threshold] + total_label_1_prediction_0[threshold]) > 0 else 0
        f1[threshold] = f1_score_manual(all_labels, all_predictions[threshold])
    
    return avg_loss, avg_unweighted_loss, accuracy, accuracy_0, accuracy_1, f1, double_nn_output_dict, all_original_labels

def train_model(net_list, optimizer_list, preprocessing, negative_concepts, train_loader, validation_loader, 
                test_loader, epochs=30, weighted_loss=False, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], training_majority_label=0, training_set_maj_min_label_ratio=1,
                validation_majority_label=0, validation_set_maj_min_label_ratio=1, test_majority_label=0, test_set_maj_min_label_ratio=1, 
                images_percentage=100, attributes_percentage=100, oversample_training=False, undersample_training=False, 
                oversample_validation=False, undersample_validation=False, use_processed_concepts=True, split_attributes=False, attribute_list=None,
                double_input=False, regularization=False):
    """
    Trains the model on the training data and validates it on the validation data
    """
    original_training_majority_label = training_majority_label
    original_validation_majority_label = validation_majority_label
    original_test_majority_label = test_majority_label

    double_nn_training_output_dict_epoch_list = [None] * epochs
    double_nn_validation_output_dict_epoch_list = [None] * epochs
    double_nn_test_output_dict_epoch_list = [None] * epochs
    if len(net_list) > 1:
        double_nn_training_output_dict_epoch_list = [] # Honestly, even I don't know what the fuck I'm doing here
        double_nn_validation_output_dict_epoch_list = []
        double_nn_test_output_dict_epoch_list = []
        for epoch in range(epochs):
            double_nn_training_output_dict_epoch_list.append({})
            double_nn_validation_output_dict_epoch_list.append({})
            double_nn_test_output_dict_epoch_list.append({})
            for net_index in range(len(net_list)):
                double_nn_training_output_dict_epoch_list[epoch][net_index] = []
                double_nn_validation_output_dict_epoch_list[epoch][net_index] = []
                double_nn_test_output_dict_epoch_list[epoch][net_index] = []
        double_nn_scores_evolution_dict = {}
        double_nn_scores_evolution_dict["training"] = {}
        double_nn_scores_evolution_dict["validation"] = {}
        double_nn_scores_evolution_dict["test"] = {}
        best_nets = []
        all_original_training_labels = []
        original_labels_complete = False

    for net_index in range(len(net_list)):
        print(f"Training network {net_index + 1} of {len(net_list)}")
        print("--------------------")
        net_list[net_index].train()

        best_validation_loss = float('inf')
        best_model_state = None

        # Lists to store losses and metrics
        training_losses = [] 
        training_unweighted_losses = [] 
        validation_losses = [] 
        validation_unweighted_losses = [] 
        test_losses = [] 
        test_unweighted_losses = [] 
        training_accuracies = {}
        validation_accuracies = {}
        test_accuracies = {}
        training_accuracies_0 = {}
        validation_accuracies_0 = {}
        test_accuracies_0 = {}
        training_accuracies_1 = {}
        validation_accuracies_1 = {}
        test_accuracies_1 = {}
        training_balanced_accuracies = {}
        validation_balanced_accuracies = {}
        test_balanced_accuracies = {}
        training_f1_scores = {}
        validation_f1_scores = {}
        test_f1_scores = {}
        for threshold in thresholds:
            training_accuracies[threshold] = [] 
            validation_accuracies[threshold] = [] 
            test_accuracies[threshold] = [] 
            training_accuracies_0[threshold] = [] 
            validation_accuracies_0[threshold] = [] 
            test_accuracies_0[threshold] = [] 
            training_accuracies_1[threshold] = [] 
            validation_accuracies_1[threshold] = [] 
            test_accuracies_1[threshold] = [] 
            training_balanced_accuracies[threshold] = [] 
            validation_balanced_accuracies[threshold] = [] 
            test_balanced_accuracies[threshold] = [] 
            training_f1_scores[threshold] = [] 
            validation_f1_scores[threshold] = [] 
            test_f1_scores[threshold] = [] 

        for epoch in range(epochs):
            total_loss = 0 
            total_unweighted_loss = 0 
            total_items = 0
            all_labels = []
            total_differences = {}
            all_predictions = {}
            total_label_1_prediction_1 = {}
            total_label_1_prediction_0 = {}
            total_label_0_prediction_1 = {}
            total_label_0_prediction_0 = {}
            for threshold in thresholds:
                all_predictions[threshold] = [] 
                total_differences[threshold] = 0 
                total_label_1_prediction_1[threshold] = 0 
                total_label_1_prediction_0[threshold] = 0 
                total_label_0_prediction_1[threshold] = 0 
                total_label_0_prediction_0[threshold] = 0 

            print(f"Starting epoch {epoch+1}/{epochs}")
            for batch_idx, data_tuple in enumerate(train_loader):
                input_list = list(data_tuple[:-1])
                if len(net_list) > 1:
                    if not preprocessing:
                        input_list = [input_list[0]] + [input_list[i] for i in range(1, len(input_list)) if (i + net_index) % 2 != 0]
                    else:
                        input_tensor = input_list[0]
                        input_tensor = input_tensor[:, [(i + net_index) % 2 == 0 for i in range(input_tensor.shape[1])]]
                        input_list = [input_tensor]
                input_list = [tensor.to(device) for tensor in input_list]
                label = data_tuple[-1]
                if net_index % 2 != 0:
                    label = 1 - label
                    training_majority_label = 1 - original_training_majority_label
                    validation_majority_label = 1 - original_validation_majority_label
                    test_majority_label = 1 - original_test_majority_label
                label = label.to(device)
                label = label.float().squeeze()
                optimizer_list[net_index].zero_grad()
                output = net_list[net_index](input_list, regularization)
                output = output.squeeze()
                output = output.to(device)
                
                if len(net_list) > 1:
                    double_nn_training_output_dict_epoch_list[epoch][net_index].extend(output.detach().cpu().numpy())
                    if net_index == 0 and not original_labels_complete:
                        all_original_training_labels.extend(label.cpu().numpy())

                if weighted_loss:
                    weight = torch.where(label == training_majority_label, torch.tensor(1.0, device=device), torch.tensor(training_set_maj_min_label_ratio, device=device))
                    loss = nn.BCELoss(weight=weight)(output, label)
                else:
                    loss = nn.BCELoss()(output, label)
                unweighted_loss = nn.BCELoss()(output, label)
                loss.backward()
                optimizer_list[net_index].step()
                total_loss += loss.item()
                total_unweighted_loss += unweighted_loss.item()

                if double_input:
                    if isinstance(output, torch.Tensor):
                        output_joint = (output[::2] + (1 - output[1::2])) / 2
                        output_joint = output_joint.squeeze()
                        output_joint = output_joint.to(device)
                    else:
                        raise ValueError("output is not a tensor")
                    
                    if isinstance(label, torch.Tensor):
                        label_joint = label[::2]
                        label_joint = label_joint.float().squeeze()
                        label_joint = label_joint.to(device)
                    else:
                        raise ValueError("label is not a tensor")

                # Calculate the number of differences
                threshold_output = {}
                differences = {}
                for threshold in thresholds:
                    if double_input:
                        threshold_output[threshold] = torch.where(output_joint > threshold, torch.tensor(1.0, device=output.device), torch.tensor(0.0, device=output.device))
                        threshold_output[threshold] = threshold_output[threshold].squeeze()
                        differences[threshold] = torch.sum(threshold_output[threshold] != label_joint)
                    else:
                        threshold_output[threshold] = torch.where(output > threshold, torch.tensor(1.0, device=output.device), torch.tensor(0.0, device=output.device))
                        threshold_output[threshold] = threshold_output[threshold].squeeze()
                        differences[threshold] = torch.sum(threshold_output[threshold] != label)
                    total_differences[threshold] += differences[threshold].item()

                # Calculate TN, FP, FN, TP for each threshold:
                label_1_prediction_1 = {}
                label_1_prediction_0 = {}
                label_0_prediction_1 = {}
                label_0_prediction_0 = {}
                for threshold in thresholds:
                    if double_input:
                        label_1_prediction_1[threshold] = torch.sum((label_joint == 1) & (threshold_output[threshold] == 1))
                        label_1_prediction_0[threshold] = torch.sum((label_joint == 1) & (threshold_output[threshold] == 0))
                        label_0_prediction_1[threshold] = torch.sum((label_joint == 0) & (threshold_output[threshold] == 1))
                        label_0_prediction_0[threshold] = torch.sum((label_joint == 0) & (threshold_output[threshold] == 0))
                    else:
                        label_1_prediction_1[threshold] = torch.sum((label == 1) & (threshold_output[threshold] == 1))
                        label_1_prediction_0[threshold] = torch.sum((label == 1) & (threshold_output[threshold] == 0))
                        label_0_prediction_1[threshold] = torch.sum((label == 0) & (threshold_output[threshold] == 1))
                        label_0_prediction_0[threshold] = torch.sum((label == 0) & (threshold_output[threshold] == 0))
                    total_label_1_prediction_1[threshold] += label_1_prediction_1[threshold].item()
                    total_label_1_prediction_0[threshold] += label_1_prediction_0[threshold].item()
                    total_label_0_prediction_1[threshold] += label_0_prediction_1[threshold].item()
                    total_label_0_prediction_0[threshold] += label_0_prediction_0[threshold].item()

                if double_input:
                    total_items += label_joint.numel()
                else:
                    total_items += label.numel()
                
                # Collect all labels and predictions for F1 score calculation
                if double_input:
                    all_labels.extend(label_joint.cpu().numpy())
                else:
                    all_labels.extend(label.cpu().numpy())

                for threshold in thresholds:
                    all_predictions[threshold].extend(threshold_output[threshold].cpu().numpy())
                
                # if batch_idx % 10 == 0:  # Print every 10 batches
                #     print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Batch Loss: {loss.item()}")
            
            if len(net_list) > 1 and all_original_training_labels != []:
                original_labels_complete = True

            avg_loss = total_loss / len(train_loader)
            avg_unweighted_loss = total_unweighted_loss / len(train_loader)
            training_losses.append(avg_loss)
            training_unweighted_losses.append(avg_unweighted_loss)
            # print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")
            # print(f"Epoch {epoch+1} completed, Average Unweighted Loss: {avg_unweighted_loss:.4f}")
            
            # Calculate training accuracy and F1 score
            training_accuracy = {}
            training_accuracy_0 = {}
            training_accuracy_1 = {}
            training_f1_score = {}
            for threshold in thresholds:
                training_accuracy[threshold] = 1 - (total_differences[threshold] / total_items)
                training_accuracy_0[threshold] = total_label_0_prediction_0[threshold] / (total_label_0_prediction_0[threshold] + total_label_0_prediction_1[threshold]) \
                    if (total_label_0_prediction_0[threshold] + total_label_0_prediction_1[threshold]) > 0 else 0
                training_accuracy_1[threshold] = total_label_1_prediction_1[threshold] / (total_label_1_prediction_1[threshold] + total_label_1_prediction_0[threshold]) \
                    if (total_label_1_prediction_1[threshold] + total_label_1_prediction_0[threshold]) > 0 else 0
                training_f1_score[threshold] = f1_score_manual(all_labels, all_predictions[threshold])
                training_accuracies[threshold].append(training_accuracy[threshold])
                training_accuracies_0[threshold].append(training_accuracy_0[threshold])
                training_accuracies_1[threshold].append(training_accuracy_1[threshold])
                training_balanced_accuracies[threshold].append((training_accuracy_0[threshold] + training_accuracy_1[threshold]) / 2)
                training_f1_scores[threshold].append(training_f1_score[threshold])

            # Validate the model on the validation data
            validation_loss, validation_unweighted_loss, validation_accuracy, validation_accuracy_0, validation_accuracy_1, validation_f1, \
            double_nn_validation_output_dict, all_original_validation_labels = test_model(net_list, net_index, validation_loader, 
                weighted_loss, validation_majority_label, validation_set_maj_min_label_ratio, thresholds, preprocessing, 
                double_nn_validation_output_dict_epoch_list[epoch], double_input, regularization)
            
            if len(net_list) > 1:
                double_nn_validation_output_dict_epoch_list[epoch] = double_nn_validation_output_dict

            validation_losses.append(validation_loss)
            validation_unweighted_losses.append(validation_unweighted_loss)
            for threshold in thresholds:
                validation_accuracies[threshold].append(validation_accuracy[threshold])
                validation_accuracies_0[threshold].append(validation_accuracy_0[threshold])
                validation_accuracies_1[threshold].append(validation_accuracy_1[threshold])
                validation_balanced_accuracies[threshold].append((validation_accuracy_0[threshold] + validation_accuracy_1[threshold]) / 2)
                validation_f1_scores[threshold].append(validation_f1[threshold])

            # Test the model on the test data
            test_loss, test_unweighted_loss, test_accuracy, test_accuracy_0, test_accuracy_1, test_f1, \
            double_nn_test_output_dict, all_original_test_labels = test_model(net_list, net_index, 
                test_loader, weighted_loss, test_majority_label, test_set_maj_min_label_ratio, thresholds, preprocessing, 
                double_nn_test_output_dict_epoch_list[epoch], double_input, regularization)
            
            if len(net_list) > 1:
                double_nn_test_output_dict_epoch_list[epoch] = double_nn_test_output_dict

            test_losses.append(test_loss)
            test_unweighted_losses.append(test_unweighted_loss)
            for threshold in thresholds:
                test_accuracies[threshold].append(test_accuracy[threshold])
                test_accuracies_0[threshold].append(test_accuracy_0[threshold])
                test_accuracies_1[threshold].append(test_accuracy_1[threshold])
                test_balanced_accuracies[threshold].append((test_accuracy_0[threshold] + test_accuracy_1[threshold]) / 2)
                test_f1_scores[threshold].append(test_f1[threshold])

            if len(net_list) > 1 and net_index == len(net_list) - 1:
                average_training_output, training_prediction_dict, double_nn_scores_evolution_dict["training"][epoch] = \
                calculate_double_nn_scores(double_nn_training_output_dict_epoch_list[epoch], all_original_training_labels, thresholds)
                average_validation_output, validation_prediction_dict, double_nn_scores_evolution_dict["validation"][epoch] = \
                calculate_double_nn_scores(double_nn_validation_output_dict_epoch_list[epoch], all_original_validation_labels, thresholds)
                average_test_output, test_prediction_dict, double_nn_scores_evolution_dict["test"][epoch] = \
                calculate_double_nn_scores(double_nn_test_output_dict_epoch_list[epoch], all_original_test_labels, thresholds)

            # Save the model if it has the best validation loss so far
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model_state = net_list[net_index].state_dict()

            # print("Training Losses: ", training_losses)
            # print("Validation Losses: ", validation_losses)
            # print("Test Losses: ", test_losses)
            # print("Training Unweighted Losses: ", training_unweighted_losses)
            # print("Validation Unweighted Losses: ", validation_unweighted_losses)
            # print("Test Unweighted Losses: ", test_unweighted_losses)
            # for threshold in thresholds:
            #     print(f"Training Accuracies with threshold {threshold}: ", training_accuracies[threshold])
            #     print(f"Validation Accuracies with threshold {threshold}: ", validation_accuracies[threshold])
            #     print(f"Test Accuracies with threshold {threshold}: ", test_accuracies[threshold])
            #     print(f"Training Accuracies (0) with threshold {threshold}: ", training_accuracies_0[threshold])
            #     print(f"Validation Accuracies (0) with threshold {threshold}: ", validation_accuracies_0[threshold])
            #     print(f"Test Accuracies (0) with threshold {threshold}: ", test_accuracies_0[threshold])
            #     print(f"Training Accuracies (1) with threshold {threshold}: ", training_accuracies_1[threshold])
            #     print(f"Validation Accuracies (1) with threshold {threshold}: ", validation_accuracies_1[threshold])
            #     print(f"Test Accuracies (1) with threshold {threshold}: ", test_accuracies_1[threshold])
            #     print(f"Training Balanced Accuracies with threshold {threshold}: ", training_balanced_accuracies[threshold])
            #     print(f"Validation Balanced Accuracies with threshold {threshold}: ", validation_balanced_accuracies[threshold])
            #     print(f"Test Balanced Accuracies with threshold {threshold}: ", test_balanced_accuracies[threshold])
            #     print(f"Training F1 Scores with threshold {threshold}: ", training_f1_scores[threshold])
            #     print(f"Validation F1 Scores with threshold {threshold}: ", validation_f1_scores[threshold])
            #     print(f"Test F1 Scores with threshold {threshold}: ", test_f1_scores[threshold])

        # Save the losses plot
        losses_list = [training_losses, validation_losses, test_losses, training_unweighted_losses, validation_unweighted_losses, test_unweighted_losses]
        label_list = ['Training Loss', 'Validation Loss', 'Test Loss', 'Training Unweighted Loss', 'Validation Unweighted Loss', 'Test Unweighted Loss']
        title = 'Training, Validation, and Test Loss Over Epochs'
        filename = 'losses.png'
        plot_losses(net_list, net_index, losses_list, label_list, title, filename, weighted_loss, 
                    images_percentage, attributes_percentage, oversample_training, 
                    undersample_training, oversample_validation, undersample_validation, 
                    use_processed_concepts, preprocessing, negative_concepts, split_attributes, attribute_list,
                    double_input)

        # Save the accuracies and F1 scores plots
        metric_dicts = [training_accuracies, validation_accuracies, test_accuracies,
                        training_accuracies_0, validation_accuracies_0, test_accuracies_0,
                        training_accuracies_1, validation_accuracies_1, test_accuracies_1,
                        training_balanced_accuracies, validation_balanced_accuracies, test_balanced_accuracies,
                        training_f1_scores, validation_f1_scores, test_f1_scores]
        metric_names = ['Accuracy', 'Accuracy', 'Accuracy', 'Accuracy (0)', 'Accuracy (0)', 'Accuracy (0)', 
                        'Accuracy (1)', 'Accuracy (1)', 'Accuracy (1)', 'Balanced Accuracy', 'Balanced Accuracy', 'Balanced Accuracy', 'F1 Score', 'F1 Score', 'F1 Score']
        labels = ['Training', 'Validation', 'Test', 'Training', 'Validation', 'Test', 'Training', 'Validation', 'Test', 'Training', 'Validation', 'Test', 'Training', 'Validation', 'Test']
        titles = ['Training Accuracy Over Epochs', 'Validation Accuracy Over Epochs', 'Test Accuracy Over Epochs',
                'Training Accuracy (0) Over Epochs', 'Validation Accuracy (0) Over Epochs', 'Test Accuracy (0) Over Epochs',
                'Training Accuracy (1) Over Epochs', 'Validation Accuracy (1) Over Epochs', 'Test Accuracy (1) Over Epochs',
                'Training Balanced Accuracy Over Epochs', 'Validation Balanced Accuracy Over Epochs', 'Test Balanced Accuracy Over Epochs',
                'Training F1 Score Over Epochs', 'Validation F1 Score Over Epochs', 'Test F1 Score Over Epochs']
        filenames = ['training_accuracies.png', 'validation_accuracies.png', 'test_accuracies.png',
                    'training_accuracies_0.png', 'validation_accuracies_0.png', 'test_accuracies_0.png',
                    'training_accuracies_1.png', 'validation_accuracies_1.png', 'test_accuracies_1.png',
                    'training_balanced_accuracies.png', 'validation_balanced_accuracies.png', 'test_balanced_accuracies.png',
                    'training_f1_scores.png', 'validation_f1_scores.png', 'test_f1_scores.png']
        plot_metrics(net_list, net_index, thresholds, metric_dicts, metric_names, labels, titles, filenames, 
                    weighted_loss, images_percentage, attributes_percentage, oversample_training, 
                    undersample_training, oversample_validation, undersample_validation, 
                    use_processed_concepts, preprocessing, negative_concepts, split_attributes, attribute_list,
                    double_input)

        # Load the best model state
        if best_model_state is not None:
            net_list[net_index].load_state_dict(best_model_state)
            print("Loaded the best model with validation loss: {:.4f}".format(best_validation_loss))
            if len(net_list) > 1:
                best_nets.append(best_model_state)

        if len(net_list) == 1:
            # Test the model on the test data
            test_loss, test_unweighted_loss, test_accuracy, test_accuracy_0, test_accuracy_1, test_f1, \
            double_nn_test_output_dict, all_original_test_labels = test_model(net_list, net_index, 
                test_loader, weighted_loss, test_majority_label, test_set_maj_min_label_ratio, thresholds, preprocessing, 
                None, double_input, regularization)

            print(f"Test Loss on best model: {test_loss:.4f}")
            print(f"Test Unweighted Loss on best model: {test_unweighted_loss:.4f}")
            for threshold in thresholds:
                print(f"Test Accuracy with threshold {threshold} on best model: {test_accuracy[threshold]:.4f}")
                print(f"Test Accuracy (0) with threshold {threshold} on best model: {test_accuracy_0[threshold]:.4f}")
                print(f"Test Accuracy (1) with threshold {threshold} on best model: {test_accuracy_1[threshold]:.4f}")
                print(f"Test Balanced Accuracy with threshold {threshold} on best model: {(test_accuracy_0[threshold] + test_accuracy_1[threshold]) / 2:.4f}")
                print(f"Test F1 score with threshold {threshold} on best model: {test_f1[threshold]:.4f}")

    if len(net_list) > 1:
        print("Plotting metrics for the double NN")
        plot_double_nn_metrics(double_nn_scores_evolution_dict, weighted_loss, images_percentage, attributes_percentage, 
                               oversample_training, undersample_training, oversample_validation, undersample_validation, 
                               use_processed_concepts, preprocessing, split_attributes, thresholds, attribute_list)
        print("Plotting metrics for the double NN completed")
    
        for net_index in range(len(net_list)):
            double_nn_test_output_dict[net_index] = []
        
        for net_index in range(len(net_list)):
            if net_index % 2 != 0:
                test_majority_label = 1 - original_test_majority_label
            net_list[net_index].load_state_dict(best_nets[net_index])
            test_loss, test_unweighted_loss, test_accuracy, test_accuracy_0, test_accuracy_1, test_f1, \
            double_nn_test_output_dict, all_original_test_labels = test_model(net_list, net_index,  
                test_loader, weighted_loss, test_majority_label, test_set_maj_min_label_ratio, thresholds, preprocessing, 
                double_nn_test_output_dict, double_input, regularization)

        average_test_output, test_prediction_dict, double_nn_scores_dict = \
        calculate_double_nn_scores(double_nn_test_output_dict, all_original_test_labels, thresholds)

        for threshold_key in double_nn_scores_dict:
            print(f"Test Accuracy with threshold {threshold_key} on best model: {double_nn_scores_dict[threshold_key]['accuracy']:.4f}")
            print(f"Test Accuracy (0) with threshold {threshold_key} on best model: {double_nn_scores_dict[threshold_key]['accuracy_0']:.4f}")
            print(f"Test Accuracy (1) with threshold {threshold_key} on best model: {double_nn_scores_dict[threshold_key]['accuracy_1']:.4f}")
            print(f"Test Balanced Accuracy with threshold {threshold_key} on best model: {double_nn_scores_dict[threshold_key]['balanced_accuracy']:.4f}")
            print(f"Test F1 score with threshold {threshold_key} on best model: {double_nn_scores_dict[threshold_key]['f1']:.4f}")


        
        
    


# Define all parameters 

# Should the model use metrics (preprocessing = True) or raw embeddings (preprocessing = False)
preprocessing = False

# Should the model inlude the use of negative concepts
negative_concepts = False

# Are positive and negative concepts inputted in 1 neural network or in 2 separate ones
double_nn = False

# Give the positive concepts and negative concepts as separate datapoints to the neural network
double_input = False

# The root directory of the CUB200 dataset
root = "./data/CUB_200_2011"

# Variables that determine the percentage of images and attributes to use
# Both should be a divisor of 100
images_percentage = 100
attributes_percentage = 100

# Instead of using a percentage of the attributes, you can also specify which attributes 
# should be picked by providing a list with the numbers of the attributes (1 to 312)
attribute_list = None # e.g. [55, 56, 134]

# Variable that states if the processed concepts should be used
use_processed_concepts = True

# Variable that states in which ratio the images should be split into training, validation, and test sets
image_split_list = [0.8, 0.1, 0.1]

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

# Learning rate for neural network
learning_rate = 0.002

# Using regularization or not in the neural network and optimizer
regularization = False

# If regularization is used, the weight decay for the optimizer
weight_decay = 0.0001

# If regularization is used, the dropout rate for the neural network
dropout_rate = 0.5

# Batch size for neural network
batch_size = 2048

# Amount of epochs used
epochs = 10

# Whether to use weighted loss or not (as compensation for class imbalance)
weighted_loss = True

# Thresholds to apply to the output of the neural network to determine the prediction
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]





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





# Step 1: Choose GPU device if available

device = torch_directml.device()  # Uses DirectML for AMD GPUs
print(f"Using device: {device}")

# Step 2: Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
    
# Step 3: Load CUB200 dataset

dataset = CUB200Dataset(root, device, model, preprocess, preprocessing, negative_concepts, images_percentage, attributes_percentage, 
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
train_model(net_list=net_list, optimizer_list=optimizer_list, preprocessing=preprocessing, negative_concepts=negative_concepts, train_loader=train_loader, 
            validation_loader=validation_loader, test_loader=test_loader, epochs=epochs, weighted_loss=weighted_loss, 
            thresholds=thresholds, training_majority_label=training_majority_label, training_set_maj_min_label_ratio=training_set_maj_min_label_ratio, 
            validation_majority_label=validation_majority_label, validation_set_maj_min_label_ratio=validation_set_maj_min_label_ratio, 
            test_majority_label=test_majority_label, test_set_maj_min_label_ratio=test_set_maj_min_label_ratio, 
            images_percentage=images_percentage, attributes_percentage=attributes_percentage, oversample_training=oversample_training, 
            undersample_training=undersample_training, oversample_validation=oversample_validation,
            undersample_validation=undersample_validation, use_processed_concepts=use_processed_concepts, 
            split_attributes=split_attributes, attribute_list=attribute_list, double_input=double_input, regularization=regularization)
