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
    def __init__(self, root, device, model, preprocess, images_percentage, attributes_percentage):
        self.root = root
        self.device = device
        self.image_dir = os.path.join(root, "images")
        self.attribute_file = os.path.join(root, "attributes", "image_attribute_labels.txt")
        self.image_file = os.path.join(root, "images.txt")
        self.attribute_names_file = os.path.join(root, "attributes", "attributes.txt")
        self.processed_attribute_names_file = os.path.join(root, "attributes", "processed_attributes.txt")
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
        Preprocesses the dataset to create a set of tuples containing image encodings and labels
        """
        preprocessed_data = []
        print("Preprocessing dataset...")
        print("Encoding all images...")
        for idx in range(self.size()):
            if idx % 100 == 0:
                print("Currently at index ", idx)
            image, labels = self.get_image_with_labels(idx)
            image_encoding = get_clip_image_embedding(image, self.model).float()
            labels_tensor = torch.tensor(labels).float().to(self.device)
            preprocessed_data.append((image_encoding, labels_tensor))
        return preprocessed_data

class AttributeClassifier(nn.Module):
    def __init__(self, input_size=512, output_size=312):
        super(AttributeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 312)
        self.fc3 = nn.Linear(312, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, img_vec):
        x = img_vec
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x

def get_clip_image_embedding(image, model):
    """
    Returns the image embedding from the CLIP model
    """
    with torch.no_grad():
        image_features = model.encode_image(image)
        return image_features

def calculate_accuracy(outputs, labels, threshold=0.5):
    predictions = (outputs > threshold).float()
    correct = (predictions == labels).float().sum()
    accuracy = correct / labels.numel()
    return accuracy.item()

def calculate_accuracy_per_label(outputs, labels, threshold=0.5):
    predictions = (outputs > threshold).float()
    tp = ((predictions == 1) & (labels == 1)).float().sum()
    tn = ((predictions == 0) & (labels == 0)).float().sum()
    fp = ((predictions == 1) & (labels == 0)).float().sum()
    fn = ((predictions == 0) & (labels == 1)).float().sum()
    accuracy_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    return accuracy_0.item(), accuracy_1.item()

def calculate_f1_score(outputs, labels, threshold=0.5):
    predictions = (outputs > threshold).float()
    return f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='micro')

def calculate_0_1_label_ratio(dataset_list):
    """
    Calculates the ratio of 0s to 1s in the dataset
    """
    count0 = 0
    count1 = 0
    for data in dataset_list:
        labels = data[1]
        count0 += (labels == 0).sum().item()
        count1 += (labels == 1).sum().item()
    return count0 / count1 if count1 > 0 else float('inf')

def train_model(train_loader, validation_loader, test_loader, model, optimizer, epochs=30, weighted=False):
    best_validation_loss = float('inf')
    best_model_state = None

    # Calculate the weight for the minority class (1)
    train_ratio = calculate_0_1_label_ratio(train_loader.dataset)
    val_ratio = calculate_0_1_label_ratio(validation_loader.dataset)
    test_ratio = calculate_0_1_label_ratio(test_loader.dataset)
    weight = torch.tensor(train_ratio, device=device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        total_accuracy_0 = 0
        total_accuracy_1 = 0
        total_f1_score = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)  # Remove the extra dimension
            weights = torch.where(labels == 1, weight, torch.tensor(1.0, device=device))
            if weighted:
                loss = nn.BCELoss(weight=weights)(outputs, labels)
            else:
                loss = nn.BCELoss()(outputs, labels)            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)
            accuracy_0, accuracy_1 = calculate_accuracy_per_label(outputs, labels)
            total_accuracy_0 += accuracy_0
            total_accuracy_1 += accuracy_1
            total_f1_score += calculate_f1_score(outputs, labels)
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        avg_accuracy_0 = total_accuracy_0 / len(train_loader)
        avg_accuracy_1 = total_accuracy_1 / len(train_loader)
        avg_balanced_accuracy = (avg_accuracy_0 + avg_accuracy_1) / 2
        avg_f1_score = total_f1_score / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {avg_accuracy:.4f}, Training Accuracy (0): {avg_accuracy_0:.4f}, Training Accuracy (1): {avg_accuracy_1:.4f}, Training Balanced Accuracy: {avg_balanced_accuracy:.4f}, Training F1 Score: {avg_f1_score:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        total_val_accuracy_0 = 0
        total_val_accuracy_1 = 0
        total_val_f1_score = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze(1)  # Remove the extra dimension
                weights = torch.where(labels == 1, weight, torch.tensor(1.0, device=device))
                if weighted:
                    loss = nn.BCELoss(weight=weights, reduction='mean')(outputs, labels)
                else:
                    loss = nn.BCELoss(reduction='mean')(outputs, labels)
                total_val_loss += loss.item()
                total_val_accuracy += calculate_accuracy(outputs, labels)
                accuracy_0, accuracy_1 = calculate_accuracy_per_label(outputs, labels)
                total_val_accuracy_0 += accuracy_0
                total_val_accuracy_1 += accuracy_1
                total_val_f1_score += calculate_f1_score(outputs, labels)
        avg_val_loss = total_val_loss / len(validation_loader)
        avg_val_accuracy = total_val_accuracy / len(validation_loader)
        avg_val_accuracy_0 = total_val_accuracy_0 / len(validation_loader)
        avg_val_accuracy_1 = total_val_accuracy_1 / len(validation_loader)
        avg_val_balanced_accuracy = (avg_val_accuracy_0 + avg_val_accuracy_1) / 2
        avg_val_f1_score = total_val_f1_score / len(validation_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}, Validation Accuracy (0): {avg_val_accuracy_0:.4f}, Validation Accuracy (1): {avg_val_accuracy_1:.4f}, Validation Balanced Accuracy: {avg_val_balanced_accuracy:.4f}, Validation F1 Score: {avg_val_f1_score:.4f}")

        # Save the best model
        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            best_model_state = model.state_dict()

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded the best model with validation loss: {:.4f}".format(best_validation_loss))

    # Test the model on the test data
    model.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    total_test_accuracy_0 = 0
    total_test_accuracy_1 = 0
    total_test_f1_score = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)  # Remove the extra dimension
            weights = torch.where(labels == 1, weight, torch.tensor(1.0, device=device))
            if weighted:
                loss = nn.BCELoss(weight=weights, reduction='mean')(outputs, labels)
            else:
                loss = nn.BCELoss(reduction='mean')(outputs, labels)
            total_test_loss += loss.item()
            total_test_accuracy += calculate_accuracy(outputs, labels)
            accuracy_0, accuracy_1 = calculate_accuracy_per_label(outputs, labels)
            total_test_accuracy_0 += accuracy_0
            total_test_accuracy_1 += accuracy_1
            total_test_f1_score += calculate_f1_score(outputs, labels)
    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_accuracy = total_test_accuracy / len(test_loader)
    avg_test_accuracy_0 = total_test_accuracy_0 / len(test_loader)
    avg_test_accuracy_1 = total_test_accuracy_1 / len(test_loader)
    avg_test_balanced_accuracy = (avg_test_accuracy_0 + avg_test_accuracy_1) / 2
    avg_test_f1_score = total_test_f1_score / len(test_loader)
    print(f"Test Loss on best model: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}, Test Accuracy (0): {avg_test_accuracy_0:.4f}, Test Accuracy (1): {avg_test_accuracy_1:.4f}, Test Balanced Accuracy: {avg_test_balanced_accuracy:.4f}, Test F1 Score: {avg_test_f1_score:.4f}")

# Example usage
device = torch_directml.device()  # Uses DirectML for AMD GPUs
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

images_percentage = 100
attributes_percentage = 100

dataset = CUB200Dataset(root="./data/CUB_200_2011", device=device, model=model, preprocess=preprocess,
                        images_percentage=images_percentage, attributes_percentage=attributes_percentage)

preprocessed_set = dataset.preprocess_dataset()

# Split dataset into 80% train, 10% validation, 10% test
indices = list(range(len(preprocessed_set)))
train_indices, no_train_indices = train_test_split(indices, test_size=0.2, random_state=69)
validation_indices, test_indices = train_test_split(no_train_indices, test_size=0.5, random_state=69)
preprocessed_list = list(preprocessed_set)
training_list = [preprocessed_list[i] for i in train_indices]
validation_list = [preprocessed_list[i] for i in validation_indices]
test_list = [preprocessed_list[i] for i in test_indices]

training_set = PreprocessedDataset(training_list)
validation_set = PreprocessedDataset(validation_list)
test_set = PreprocessedDataset(test_list)

num_labels = len(dataset.attribute_names)

attribute_classifier = AttributeClassifier(input_size=512, output_size=num_labels).to(device)
optimizer = optim.Adam(attribute_classifier.parameters(), lr=0.001)

train_loader = DataLoader(training_set, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

train_model(train_loader, validation_loader, test_loader, attribute_classifier, optimizer, epochs=50, weighted=True)

