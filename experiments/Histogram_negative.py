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
import time
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
    def __init__(self, root, device, model, preprocess, images_percentage, attributes_percentage):
        self.root = root
        self.device = device
        self.image_dir = os.path.join(root, "images")
        self.attribute_file = os.path.join(root, "attributes", "image_attribute_labels.txt")
        self.image_file = os.path.join(root, "images.txt")
        self.attribute_names_file = os.path.join(root, "attributes", "attributes.txt")
        self.processed_attribute_names_file = os.path.join(root, "attributes", "processed_attributes.txt")
        self.negative_processed_attribute_names_file = os.path.join(root, "attributes", "negative_processed_attributes.txt")
        self.model = model
        self.preprocess = preprocess

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
            if i % attributes_divider == 0:
                parts = line.strip().split()
                self.attribute_numbers.append(int(parts[0]))
                self.attribute_names.append(parts[1])
            i += 1

        with open(self.processed_attribute_names_file, "r") as f:
            lines = f.readlines()
        i = 0
        for line in lines:
            if i % attributes_divider == 0:
                self.processed_attribute_names.append(re.sub(r'^\d+\.\s*', '', line.strip()))
            i += 1

        with open(self.negative_processed_attribute_names_file, "r") as f:
            lines = f.readlines()
        i = 0
        for line in lines:
            if i % attributes_divider == 0:
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
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
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
        self.attribute_encodings = []
        self.negative_attribute_encodings = []
        for attr in self.processed_attribute_names:
            self.attribute_encodings.append(get_clip_text_embedding(attr, self.model, self.device).float())
        for attr in self.negative_processed_attribute_names:
            self.negative_attribute_encodings.append(get_clip_text_embedding(attr, self.model, self.device).float())
        preprocessed_data = set()
        print("Preprocessing dataset...")
        print("Encoding all images...")
        for idx in range(self.size()):
            if idx % 100 == 0:
                print("Currently at index ", idx)
            image, labels = self.get_image_with_labels(idx)
            image_encoding = get_clip_image_embedding(image, self.model).float()
            for attr_idx in range(len(self.processed_attribute_names)):
                concept_encoding = self.attribute_encodings[attr_idx]
                negative_concept_encoding = self.negative_attribute_encodings[attr_idx]
                label = labels[attr_idx]
                norm_diff = torch.norm(image_encoding - concept_encoding)
                negative_norm_diff = torch.norm(image_encoding - negative_concept_encoding)
                cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, concept_encoding, dim=1)
                negative_cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, negative_concept_encoding, dim=1)
                angle = torch.acos(cosine_similarity)
                negative_angle = torch.acos(negative_cosine_similarity)
                dot_product = torch.dot(image_encoding.squeeze(), concept_encoding.squeeze())
                negative_dot_product = torch.dot(image_encoding.squeeze(), negative_concept_encoding.squeeze())
                inputs = torch.tensor([norm_diff, negative_norm_diff, cosine_similarity, negative_cosine_similarity, angle, negative_angle, dot_product, negative_dot_product])
                preprocessed_data.add((inputs, label))
        return preprocessed_data
    

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
    
def get_metrics(labels, predictions):
    """
    Returns the accuracy, balanced accuracy, precision, recall, and f1 score of the model
    """
    accuracy = sum([1 for i in range(len(labels)) if labels[i] == predictions[i]]) / len(labels)
    tp = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 1])
    tn = sum([1 for i in range(len(labels)) if labels[i] == 0 and predictions[i] == 0])
    fp = sum([1 for i in range(len(labels)) if labels[i] == 0 and predictions[i] == 1])
    fn = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 0])
    accuracy_on_zeros = tn / (tn + fp) if tn + fp != 0 else 0
    accuracy_on_ones = tp / (tp + fn) if tp + fn != 0 else 0
    balanced_accuracy = (accuracy_on_zeros + accuracy_on_ones) / 2
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    return accuracy, balanced_accuracy, precision, recall, f1
    
    
def find_optimal_threshold(sorted_list_of_tuples):
    sorted_list_of_labels = [x[1] for x in sorted_list_of_tuples]

    metrics_list = ["Threshold index", 
                    "Threshold value", 
                    "Accuracy if below threshold predicts 0 and above predicts 1", 
                    "Accuracy if below threshold predicts 1 and above predicts 0",
                    "Balanced accuracy if below threshold predicts 0 and above predicts 1",
                    "Balanced accuracy if below threshold predicts 1 and above predicts 0",
                    "Precision if below threshold predicts 0 and above predicts 1",
                    "Precision if below threshold predicts 1 and above predicts 0",
                    "Recall if below threshold predicts 0 and above predicts 1",
                    "Recall if below threshold predicts 1 and above predicts 0",
                    "F1 score if below threshold predicts 0 and above predicts 1",
                    "F1 score if below threshold predicts 1 and above predicts 0"]

    optimal_threshold_dictionary = {}
    optimal_threshold_dictionary["Best according to accuracy"] = {}
    optimal_threshold_dictionary["Best according to balanced accuracy"] = {}
    optimal_threshold_dictionary["Best according to f1 score"] = {}
    for metric in metrics_list:
        optimal_threshold_dictionary["Best according to accuracy"][metric] = 0
        optimal_threshold_dictionary["Best according to balanced accuracy"][metric] = 0
        optimal_threshold_dictionary["Best according to f1 score"][metric] = 0

    print("Starting to iterate through the sorted list of tuples with length ", len(sorted_list_of_tuples))
    for i in range(0, len(sorted_list_of_tuples)+1):
        if i % 100 == 0:
            print("Currently at index ", i)
        if i == 0:
            threshold = float('-inf')
        elif i == len(sorted_list_of_tuples):
            threshold = float('inf')
        else:
            threshold = (sorted_list_of_tuples[i-1][0] + sorted_list_of_tuples[i][0]) / 2
        below_threshold_list_of_labels = [x[1] for x in sorted_list_of_tuples if x[0] < threshold]
        above_threshold_list_of_labels = [x[1] for x in sorted_list_of_tuples if x[0] >= threshold]
        below_threshold_list_of_zeros = [0] * len(below_threshold_list_of_labels)
        above_threshold_list_of_ones = [1] * len(above_threshold_list_of_labels)
        below_threshold_list_of_ones = [1] * len(below_threshold_list_of_labels)
        above_threshold_list_of_zeros = [0] * len(above_threshold_list_of_labels)
        zeros_ones_list = below_threshold_list_of_zeros + above_threshold_list_of_ones
        ones_zeros_list = below_threshold_list_of_ones + above_threshold_list_of_zeros

        accuracy_0_1, balanced_accuracy_0_1, precision_0_1, recall_0_1, f1_0_1 = get_metrics(sorted_list_of_labels, zeros_ones_list)
        accuracy_1_0, balanced_accuracy_1_0, precision_1_0, recall_1_0, f1_1_0 = get_metrics(sorted_list_of_labels, ones_zeros_list)

        accuracy = max(accuracy_0_1, accuracy_1_0)

        balanced_accuracy = max(balanced_accuracy_0_1, balanced_accuracy_1_0)

        f1 = max(f1_0_1, f1_1_0)

        if accuracy > optimal_threshold_dictionary["Best according to accuracy"]["Accuracy if below threshold predicts 0 and above predicts 1"] and \
            accuracy > optimal_threshold_dictionary["Best according to accuracy"]["Accuracy if below threshold predicts 1 and above predicts 0"]:
            optimal_threshold_dictionary["Best according to accuracy"]["Threshold index"] = f"{i} of the {len(sorted_list_of_tuples)}"
            optimal_threshold_dictionary["Best according to accuracy"]["Threshold value"] = threshold
            optimal_threshold_dictionary["Best according to accuracy"]["Accuracy if below threshold predicts 0 and above predicts 1"] = accuracy_0_1
            optimal_threshold_dictionary["Best according to accuracy"]["Accuracy if below threshold predicts 1 and above predicts 0"] = accuracy_1_0
            optimal_threshold_dictionary["Best according to accuracy"]["Balanced accuracy if below threshold predicts 0 and above predicts 1"] = balanced_accuracy_0_1
            optimal_threshold_dictionary["Best according to accuracy"]["Balanced accuracy if below threshold predicts 1 and above predicts 0"] = balanced_accuracy_1_0
            optimal_threshold_dictionary["Best according to accuracy"]["Precision if below threshold predicts 0 and above predicts 1"] = precision_0_1
            optimal_threshold_dictionary["Best according to accuracy"]["Precision if below threshold predicts 1 and above predicts 0"] = precision_1_0
            optimal_threshold_dictionary["Best according to accuracy"]["Recall if below threshold predicts 0 and above predicts 1"] = recall_0_1
            optimal_threshold_dictionary["Best according to accuracy"]["Recall if below threshold predicts 1 and above predicts 0"] = recall_1_0
            optimal_threshold_dictionary["Best according to accuracy"]["F1 score if below threshold predicts 0 and above predicts 1"] = f1_0_1
            optimal_threshold_dictionary["Best according to accuracy"]["F1 score if below threshold predicts 1 and above predicts 0"] = f1_1_0

        if balanced_accuracy > optimal_threshold_dictionary["Best according to balanced accuracy"]["Balanced accuracy if below threshold predicts 0 and above predicts 1"] and \
            balanced_accuracy > optimal_threshold_dictionary["Best according to balanced accuracy"]["Balanced accuracy if below threshold predicts 1 and above predicts 0"]:
            optimal_threshold_dictionary["Best according to balanced accuracy"]["Threshold index"] = f"{i} of the {len(sorted_list_of_tuples)}"
            optimal_threshold_dictionary["Best according to balanced accuracy"]["Threshold value"] = threshold
            optimal_threshold_dictionary["Best according to balanced accuracy"]["Accuracy if below threshold predicts 0 and above predicts 1"] = accuracy_0_1
            optimal_threshold_dictionary["Best according to balanced accuracy"]["Accuracy if below threshold predicts 1 and above predicts 0"] = accuracy_1_0
            optimal_threshold_dictionary["Best according to balanced accuracy"]["Balanced accuracy if below threshold predicts 0 and above predicts 1"] = balanced_accuracy_0_1
            optimal_threshold_dictionary["Best according to balanced accuracy"]["Balanced accuracy if below threshold predicts 1 and above predicts 0"] = balanced_accuracy_1_0
            optimal_threshold_dictionary["Best according to balanced accuracy"]["Precision if below threshold predicts 0 and above predicts 1"] = precision_0_1
            optimal_threshold_dictionary["Best according to balanced accuracy"]["Precision if below threshold predicts 1 and above predicts 0"] = precision_1_0
            optimal_threshold_dictionary["Best according to balanced accuracy"]["Recall if below threshold predicts 0 and above predicts 1"] = recall_0_1
            optimal_threshold_dictionary["Best according to balanced accuracy"]["Recall if below threshold predicts 1 and above predicts 0"] = recall_1_0
            optimal_threshold_dictionary["Best according to balanced accuracy"]["F1 score if below threshold predicts 0 and above predicts 1"] = f1_0_1
            optimal_threshold_dictionary["Best according to balanced accuracy"]["F1 score if below threshold predicts 1 and above predicts 0"] = f1_1_0

        if f1 > optimal_threshold_dictionary["Best according to f1 score"]["F1 score if below threshold predicts 0 and above predicts 1"] and \
            f1 > optimal_threshold_dictionary["Best according to f1 score"]["F1 score if below threshold predicts 1 and above predicts 0"]:
            optimal_threshold_dictionary["Best according to f1 score"]["Threshold index"] = f"{i} of the {len(sorted_list_of_tuples)}"
            optimal_threshold_dictionary["Best according to f1 score"]["Threshold value"] = threshold
            optimal_threshold_dictionary["Best according to f1 score"]["Accuracy if below threshold predicts 0 and above predicts 1"] = accuracy_0_1
            optimal_threshold_dictionary["Best according to f1 score"]["Accuracy if below threshold predicts 1 and above predicts 0"] = accuracy_1_0
            optimal_threshold_dictionary["Best according to f1 score"]["Balanced accuracy if below threshold predicts 0 and above predicts 1"] = balanced_accuracy_0_1
            optimal_threshold_dictionary["Best according to f1 score"]["Balanced accuracy if below threshold predicts 1 and above predicts 0"] = balanced_accuracy_1_0
            optimal_threshold_dictionary["Best according to f1 score"]["Precision if below threshold predicts 0 and above predicts 1"] = precision_0_1
            optimal_threshold_dictionary["Best according to f1 score"]["Precision if below threshold predicts 1 and above predicts 0"] = precision_1_0
            optimal_threshold_dictionary["Best according to f1 score"]["Recall if below threshold predicts 0 and above predicts 1"] = recall_0_1
            optimal_threshold_dictionary["Best according to f1 score"]["Recall if below threshold predicts 1 and above predicts 0"] = recall_1_0
            optimal_threshold_dictionary["Best according to f1 score"]["F1 score if below threshold predicts 0 and above predicts 1"] = f1_0_1
            optimal_threshold_dictionary["Best according to f1 score"]["F1 score if below threshold predicts 1 and above predicts 0"] = f1_1_0

    return optimal_threshold_dictionary
    
def plot_histogram(data, title, xlabel, ylabel, frequency=False):
    data_0 = [x[0] for x in data if x[1] == 0]
    data_1 = [x[0] for x in data if x[1] == 1]
    
    if frequency:
        plt.hist(data_0, bins=50, alpha=0.5, label='Label 0', color='blue', density=True)
        plt.hist(data_1, bins=50, alpha=0.5, label='Label 1', color='red', density=True)
    else:
        plt.hist(data_0, bins=50, alpha=0.5, label='Label 0', color='blue')
        plt.hist(data_1, bins=50, alpha=0.5, label='Label 1', color='red')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    plt.show()


# Step 1: Choose GPU device if available

device = torch_directml.device()  # Uses DirectML for AMD GPUs
print(f"Using device: {device}")

# Step 2: Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
    
# Step 3: Load CUB200 dataset

# Variables that determine the percentage of images and attributes to use
# Both should be a divisor of 100
images_percentage = 100
attributes_percentage = 100

dataset = CUB200Dataset(root="./data/CUB_200_2011", device=device, model=model, preprocess=preprocess,
                        images_percentage=images_percentage, attributes_percentage=attributes_percentage)

print("Size of the CUB200 dataset before preprocessing: ", dataset.size())

start_preprocessing_time = time.time()

preprocessed_set = dataset.preprocess_dataset()

end_preprocessing_time = time.time()

print("Time taken to preprocess the dataset: ", end_preprocessing_time - start_preprocessing_time)

print("Size of the CUB200 dataset after preprocessing", len(preprocessed_set))

preprocessed_list = list(preprocessed_set)

print("Creating metric lists...")

difference_norm_pos_minus_neg_list = [(inputs[0].item() - inputs[1].item(), label) for inputs, label in preprocessed_list]
cosine_similarity_pos_minus_neg_list = [(inputs[2].item() - inputs[3].item(), label) for inputs, label in preprocessed_list]
angle_pos_minus_neg_list = [(inputs[4].item() - inputs[5].item(), label) for inputs, label in preprocessed_list]
dot_product_pos_minus_neg_list = [(inputs[6].item() - inputs[7].item(), label) for inputs, label in preprocessed_list]

print("Sorting metric lists...")

start_sorting_time = time.time()

difference_norm_pos_minus_neg_list.sort(key=lambda x: x[0])
cosine_similarity_pos_minus_neg_list.sort(key=lambda x: x[0])
angle_pos_minus_neg_list.sort(key=lambda x: x[0])
dot_product_pos_minus_neg_list.sort(key=lambda x: x[0])

end_sorting_time = time.time()

print("Time taken to sort metric lists: ", end_sorting_time - start_sorting_time)

print("Checking results for threshold 0")

difference_norm_labels = [x[1] for x in difference_norm_pos_minus_neg_list]
cosine_similarity_labels = [x[1] for x in cosine_similarity_pos_minus_neg_list]
angle_labels = [x[1] for x in angle_pos_minus_neg_list]
dot_product_labels = [x[1] for x in dot_product_pos_minus_neg_list]

difference_norm_predictions = [1 if x[0] < 0 else 0 for x in difference_norm_pos_minus_neg_list]
cosine_similarity_predictions = [1 if x[0] > 0 else 0 for x in cosine_similarity_pos_minus_neg_list]
angle_predictions = [1 if x[0] < 0 else 0 for x in angle_pos_minus_neg_list]
dot_product_predictions = [1 if x[0] > 0 else 0 for x in dot_product_pos_minus_neg_list]

difference_norm_accuracy, difference_norm_balanced_accuracy, difference_norm_precision, difference_norm_recall, difference_norm_f1 = get_metrics(difference_norm_labels, difference_norm_predictions)
cosine_similarity_accuracy, cosine_similarity_balanced_accuracy, cosine_similarity_precision, cosine_similarity_recall, cosine_similarity_f1 = get_metrics(cosine_similarity_labels, cosine_similarity_predictions)
angle_accuracy, angle_balanced_accuracy, angle_precision, angle_recall, angle_f1 = get_metrics(angle_labels, angle_predictions)
dot_product_accuracy, dot_product_balanced_accuracy, dot_product_precision, dot_product_recall, dot_product_f1 = get_metrics(dot_product_labels, dot_product_predictions)

print("Results for the norm of the difference between the image and attribute embeddings (pos - neg), using threshold 0:")
print("Accuracy: ", difference_norm_accuracy)
print("Balanced accuracy: ", difference_norm_balanced_accuracy)
print("Precision: ", difference_norm_precision)
print("Recall: ", difference_norm_recall)
print("F1 score: ", difference_norm_f1)

print("--------------------")

print("Results for the cosine similarity between the image and attribute embeddings (pos - neg), using threshold 0:")
print("Accuracy: ", cosine_similarity_accuracy)
print("Balanced accuracy: ", cosine_similarity_balanced_accuracy)
print("Precision: ", cosine_similarity_precision)
print("Recall: ", cosine_similarity_recall)
print("F1 score: ", cosine_similarity_f1)

print("--------------------")

print("Results for the angle between the image and attribute embeddings (pos - neg), using threshold 0:")
print("Accuracy: ", angle_accuracy)
print("Balanced accuracy: ", angle_balanced_accuracy)
print("Precision: ", angle_precision)
print("Recall: ", angle_recall)
print("F1 score: ", angle_f1)

print("--------------------")

print("Results for the dot product between the image and attribute embeddings (pos - neg), using threshold 0:")
print("Accuracy: ", dot_product_accuracy)
print("Balanced accuracy: ", dot_product_balanced_accuracy)
print("Precision: ", dot_product_precision)
print("Recall: ", dot_product_recall)
print("F1 score: ", dot_product_f1)

print("--------------------")

print("Finding optimal threshold for each metric...", "\n")

# Find optimal threshold for each metric

start_optimal_threshold_time = time.time()

optimal_threshold_dictionary_difference_norm_ = find_optimal_threshold(difference_norm_pos_minus_neg_list)
optimal_threshold_dictionary_cosine_similarity = find_optimal_threshold(cosine_similarity_pos_minus_neg_list)
optimal_threshold_dictionary_angle = find_optimal_threshold(angle_pos_minus_neg_list)
optimal_threshold_dictionary_dot_product = find_optimal_threshold(dot_product_pos_minus_neg_list)

end_optimal_threshold_time = time.time()

print("Time taken to find optimal threshold for all metrics: ", end_optimal_threshold_time - start_optimal_threshold_time)

print("Optimal threshold information for the norm of the difference (pos - neg) between the image and attribute embeddings:", "\n")
for key in optimal_threshold_dictionary_difference_norm_:
    print("\n", key, "\n")
    for metric_key in optimal_threshold_dictionary_difference_norm_[key]:
        print(metric_key, optimal_threshold_dictionary_difference_norm_[key][metric_key])
print("\n", "--------------------", "\n")

print("Optimal threshold information for the cosine similarity (pos - neg) between the image and attribute embeddings:", "\n")
for key in optimal_threshold_dictionary_cosine_similarity:
    print("\n", key, "\n")
    for metric_key in optimal_threshold_dictionary_cosine_similarity[key]:
        print(metric_key, optimal_threshold_dictionary_cosine_similarity[key][metric_key])
print("\n", "--------------------", "\n")

print("Optimal threshold information for the angle (pos - neg) between the image and attribute embeddings:", "\n")
for key in optimal_threshold_dictionary_angle:
    print("\n", key, "\n")
    for metric_key in optimal_threshold_dictionary_angle[key]:
        print(metric_key, optimal_threshold_dictionary_angle[key][metric_key])
print("\n", "--------------------", "\n")

print("Optimal threshold information for the dot product (pos - neg) between the image and attribute embeddings:", "\n")
for key in optimal_threshold_dictionary_dot_product:
    print("\n", key, "\n")
    for metric_key in optimal_threshold_dictionary_dot_product[key]:
        print(metric_key, optimal_threshold_dictionary_dot_product[key][metric_key])

# Plot histograms for frequencies
plot_histogram(difference_norm_pos_minus_neg_list, 'Histogram of Norm Differences Pos minus Neg (Frequency)', 'Norm Difference Pos minus Neg', 'Frequency', frequency=True)
plot_histogram(cosine_similarity_pos_minus_neg_list, 'Histogram of Cosine Similarities Pos minus Neg (Frequency)', 'Cosine Similarity Pos minus Neg', 'Frequency', frequency=True)
plot_histogram(angle_pos_minus_neg_list, 'Histogram of Angles Pos minus Neg (Frequency)', 'Angle Pos minus Neg (radians)', 'Frequency', frequency=True)
plot_histogram(dot_product_pos_minus_neg_list, 'Histogram of Dot Products Pos minus Neg (Frequency)', 'Dot Product Pos minus Neg', 'Frequency', frequency=True)

# Plot histograms for counts
plot_histogram(difference_norm_pos_minus_neg_list, 'Histogram of Norm Differences Pos minus Neg ', 'Norm Difference Pos minus Neg', 'Count')
plot_histogram(cosine_similarity_pos_minus_neg_list, 'Histogram of Cosine Similarities Pos minus Neg', 'Cosine Similarity Pos minus Neg', 'Count')
plot_histogram(angle_pos_minus_neg_list, 'Histogram of Angles Pos minus Neg', 'Angle Pos minus Neg (radians)', 'Count')
plot_histogram(dot_product_pos_minus_neg_list, 'Histogram of Dot Products Pos minus Neg', 'Dot Product Pos minus Neg', 'Count')