import csv
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import re
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import clip
from sklearn.model_selection import train_test_split


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



class CUB200Dataset(Dataset):
    def __init__(self, root, device, model, preprocess, preprocessing, negative_concepts, 
                 images_percentage, attributes_percentage, attribute_list=None, 
                 use_processed_concepts=False, split_images=False, 
                 image_split_list=[0.8, 0.1, 0.1], split_attributes=False, 
                 attribute_split_list=[0.8, 0.1, 0.1], double_input=False):

        self.root = root
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.preprocessing = preprocessing
        self.negative_concepts = negative_concepts
        self.split_images = split_images
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
        image_indices = list(range(self.size()))
        if self.split_images:
            image_test_size_1 = self.image_split_list[1] + self.image_split_list[2]
            image_test_size_2 = round(self.image_split_list[2] / image_test_size_1,1)
            image_train_indices, image_no_train_indices = train_test_split(image_indices, test_size=image_test_size_1, random_state=69)
            image_validation_indices, image_test_indices = train_test_split(image_no_train_indices, test_size=image_test_size_2, random_state=69)
        else: 
            image_train_indices = image_indices
            image_validation_indices = image_indices
            image_test_indices = image_indices
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
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((image_encoding, concept_encoding, label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
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
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((inputs, label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((inputs, label))
                elif not self.preprocessing and self.negative_concepts and not self.double_input:
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
                elif not self.preprocessing and self.negative_concepts and self.double_input:
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((image_encoding, concept_encoding, label))
                        training_list.append((image_encoding, negative_concept_encoding, 1 - label))
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((image_encoding, concept_encoding, label))
                        validation_list.append((image_encoding, negative_concept_encoding, 1 - label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
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
                        training_list.append((inputs, label,image_encoding))
                        training_list.append((inputs, label,image_encoding))
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((inputs, label,image_encoding))
                        validation_list.append((inputs, label,image_encoding))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((inputs, label,image_encoding))
                        test_list.append((inputs, label,image_encoding))
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
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((inputs_pos, label))
                        validation_list.append((inputs_neg, 1 - label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((inputs_pos, label))
                        test_list.append((inputs_neg, 1 - label))
        return training_list, validation_list, test_list

class VOC2012Dataset(Dataset):
    def __init__(self, root, device, model, preprocess, preprocessing, negative_concepts, 
                 images_percentage, attributes_percentage, attribute_list=None, # attributes_list expects attribute names
                 use_processed_concepts=False, split_images=False, image_split_list=[0.8, 0.1, 0.1], 
                 split_attributes=False, attribute_split_list=[0.8, 0.1, 0.1],
                 double_input=False):

        self.root = root
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.preprocessing = preprocessing
        self.negative_concepts = negative_concepts
        self.split_images = split_images
        self.image_split_list = image_split_list
        self.split_attributes = split_attributes
        self.attribute_split_list = attribute_split_list
        self.double_input = double_input

        self.image_dir = os.path.join(root, "JPEGImages")
        self.attribute_dir = os.path.join(root, "ImageSets","Main")
        self.image_file_part1 = os.path.join(root, "ImageSets/Main/train.txt")
        self.image_file_part2 = os.path.join(root, "ImageSets/Main/val.txt")
        self.image_file_part3 = os.path.join(root, "ImageSets/Main/trainval.txt")
        self.attribute_names_file = os.path.join(root, "attributes", "attributes.txt")
        self.processed_attribute_names_file = os.path.join(root, "attributes", "processed_attributes.txt")
        self.negative_processed_attribute_names_file = os.path.join(root, "attributes", "negative_processed_attributes.txt")

        # Load image paths
        images_divider = 100 // images_percentage
        lines = set()
        with open(self.image_file_part1, "r") as f:
            lines.update(f.read().splitlines())
        with open(self.image_file_part2, "r") as f:
            lines.update(f.read().splitlines())
        with open(self.image_file_part3, "r") as f:
            lines.update(f.read().splitlines())
        lines = sorted(list(lines))

        self.image_ids = []
        self.image_paths = []
        i = 0
        for line in lines:
            if i % images_divider == 0:
                self.image_ids.append(line.strip().split()[0])
                self.image_paths.append(os.path.join(self.image_dir, line + ".jpg"))
            i += 1
        
     

        # Load attribute names and attribute values
        attributes_divider = 100 // attributes_percentage
        self.all_attribute_names = []
        self.attribute_names = []
        self.attribute_numbers = []
        self.processed_attribute_names = []
        self.negative_processed_attribute_names = []


        self.all_attribute_names = sorted(list(set([element.split("_")[0] for element 
                          in os.listdir(self.attribute_dir)
                          if element not in {"train.txt", "val.txt", "trainval.txt"}])))

        
        
        if (attribute_list is None):
            for i in range(len(self.all_attribute_names)):
                if i % attributes_divider == 0:
                    self.attribute_numbers.append(i)
                    self.attribute_names.append(self.all_attribute_names[i])
        else:
            for i in range(len(self.all_attribute_names)):
                if i in attribute_list:
                    self.attribute_numbers.append(i)
                    self.attribute_names.append(self.all_attribute_names[i])
        
        # Write here code if want to use attribute sentences 

        #with open(self.processed_attribute_names_file, "r") as f:
        #    lines = f.readlines()
        #i = 0
        #for line in lines:
        #    if (attribute_list is None and i % attributes_divider == 0) or (attribute_list is not None and i in attribute_list):
        #        self.processed_attribute_names.append(re.sub(r'^\d+\.\s*', '', line.strip()))
        #    i += 1

        # choose if attributes names are used or attributes sentences
        if use_processed_concepts:
            self.used_attribute_names = self.processed_attribute_names
        else:
            self.used_attribute_names = self.attribute_names
        #self.used_attribute_names = self.processed_attribute_names if use_processed_concepts else self.attribute_names

        # negative attributes
        #with open(self.negative_processed_attribute_names_file, "r") as f:
        #    lines = f.readlines()
        #i = 0
        #for line in lines:
        #    if (attribute_list is None and i % attributes_divider == 0) or (attribute_list is not None and i in attribute_list):
        #        self.negative_processed_attribute_names.append(re.sub(r'^\d+\.\s*', '', line.strip()))
        #    i += 1

        self.image_attributes = dict()
        # Initialize empty attribute vectors for selected image IDs
        for image_id in self.image_ids:
            self.image_attributes[image_id] = [-1] * len(self.attribute_numbers)

        # Loop over attributes
        for i, name in enumerate(self.attribute_names):
            attribute_index = self.attribute_numbers[i]
            
            for split in ["train", "val", "trainval"]:
                file_path = os.path.join(self.attribute_dir, f"{name}_{split}.txt")
                if not os.path.exists(file_path):
                    continue

                with open(file_path, "r") as file:
                    for line in file:
                        parts = line.strip().split()
                        image_id = parts[0]
                        is_present = 0 if int(parts[1]) == -1 else 1

                        if image_id in self.image_ids:
                            self.image_attributes[image_id][i] = is_present
        for key in self.image_attributes:
            self.image_attributes[key] = [
                val if val != -1 else 0  # or just skip
                for val in self.image_attributes[key]
            ]

        #print("image_ids: ", self.image_ids)
        print("Attribute numbers: ", self.attribute_numbers)
        print("Attribute names: ", self.attribute_names)

        for atr in self.used_attribute_names:
            self.negative_processed_attribute_names.append("Not a " + atr)

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
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        attributes = self.image_attributes[image_id]
        return image, attributes
    def get_image_name_with_labels(self, list_idx):
        """
        Returns the image name and its attributes at the given index
        """
        if type(list_idx) == list:
            for idx in list_idx:
                image_id = self.image_ids[idx]
                print(os.path.join(self.image_dir, image_id + ".jpg"),self.image_attributes[image_id])
                
        else:
            image_id = self.image_ids[list_idx]
            image_path = os.path.join(self.image_dir, image_id + ".jpg")
            attributes = self.image_attributes[image_id]
            return image_path, attributes, image_id
    
    def calculate_label_ratio(self):
        """
        Calculates the ratio of 0s to 1s in the dataset
        Assumes imagesdivider is 1
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
            self.attribute_encodings.append(get_clip_text_embedding("" + attr, self.model, self.device).float())
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
        image_indices = list(range(self.size()))
        if self.split_images:
            image_test_size_1 = self.image_split_list[1] + self.image_split_list[2]
            image_test_size_2 = round(self.image_split_list[2] / image_test_size_1,1)
            image_train_indices, image_no_train_indices = train_test_split(image_indices, test_size=image_test_size_1, random_state=69)
            image_validation_indices, image_test_indices = train_test_split(image_no_train_indices, test_size=image_test_size_2, random_state=69)
        else: 
            image_train_indices = image_indices
            image_validation_indices = image_indices
            image_test_indices = image_indices
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
            
        print("self.attribute_numbers",self.attribute_numbers)
        print("self.used_attribute_names",self.used_attribute_names)
        #print("self.attribute_encodings",self.attribute_encodings)
        for idx in range(self.size()):
            if idx % 100 == 0:
                print("Currently at index ",idx)
            image, labels = self.get_image_with_labels(idx)
            image_encoding = get_clip_image_embedding(image, self.model).float()
            for attr_idx in range(len(self.attribute_numbers)):
                concept_encoding = self.attribute_encodings[attr_idx]
                if self.negative_concepts:
                    negative_concept_encoding = self.negative_attribute_encodings[attr_idx]
                label = labels[attr_idx]
                if not self.preprocessing and not self.negative_concepts:
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((image_encoding, concept_encoding, label))
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((image_encoding, concept_encoding, label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
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
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((inputs, label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((inputs, label))
                elif not self.preprocessing and self.negative_concepts and not self.double_input:
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
                elif not self.preprocessing and self.negative_concepts and self.double_input:
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((image_encoding, concept_encoding, label))
                        training_list.append((image_encoding, negative_concept_encoding, 1 - label))
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((image_encoding, concept_encoding, label))
                        validation_list.append((image_encoding, negative_concept_encoding, 1 - label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
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
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((inputs, label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
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
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((inputs_pos, label))
                        validation_list.append((inputs_neg, 1 - label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((inputs_pos, label))
                        test_list.append((inputs_neg, 1 - label))
        return training_list, validation_list, test_list

class celebADataset(Dataset):
    def __init__(self, root, device, model, preprocess, preprocessing, negative_concepts, 
                 images_percentage, attributes_percentage, attribute_list=None, # attributes_list expects attribute names
                 use_processed_concepts=False, split_images=False, image_split_list=[0.8, 0.1, 0.1], 
                 split_attributes=False, attribute_split_list=[0.8, 0.1, 0.1],
                 double_input=False):

        self.root = root
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.preprocessing = preprocessing
        self.negative_concepts = negative_concepts
        self.split_images = split_images
        self.image_split_list = image_split_list
        self.split_attributes = split_attributes
        self.attribute_split_list = attribute_split_list
        self.double_input = double_input

        self.image_dir = os.path.join(root, "Celebrity_Faces_Dataset")
        self.attribute_dir = os.path.join(root, "list_attr_celeba.csv")
        
        #self.processed_attribute_names_file = os.path.join(root, "attributes", "processed_attributes.txt")
        #self.negative_processed_attribute_names_file = os.path.join(root, "attributes", "negative_processed_attributes.txt")

        # Load image paths
        images_divider = 100 // images_percentage
        attributes_divider = 100 // attributes_percentage
        
        self.image_ids = [] # from 1 to 50,000 so when using as index: idx = image_id -1
        self.image_paths = []
      
        # Load attribute names and attribute values
        self.all_attribute_names = []
        self.attribute_names = []
        self.attribute_numbers = []
        self.processed_attribute_names = []
        self.negative_processed_attribute_names = []

        self.image_attributes = dict()
        with open(self.attribute_dir, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            first_row = next(reader)  # Read the header row
            self.all_attribute_names = [text.replace("_"," ") for text in first_row[1:]]
            image_id = 0
            image_index = 0
            for row in reader:
                if image_index >= 50000:
                    break
                if image_index % images_divider == 0:
                    self.image_ids.append(image_id)
                    self.image_paths.append(row[0])  # Assuming the first column contains the image paths
                    self.image_attributes[image_id] = [1 if int(k) == 1 else 0 for k in row[1:]]
                    image_id += 1
                image_index += 1

        if (attribute_list is None):
            for i in range(len(self.all_attribute_names)):
                if i % attributes_divider == 0:
                    self.attribute_numbers.append(i)
                    self.attribute_names.append(self.all_attribute_names[i])
        else:
            for i in range(len(self.all_attribute_names)):
                if i in attribute_list:
                    print("i: ",i)
                    self.attribute_numbers.append(i)
                    self.attribute_names.append(self.all_attribute_names[i])
        
        # Write here code if want to use attribute sentences 

        #with open(self.processed_attribute_names_file, "r") as f:
        #    lines = f.readlines()
        #i = 0
        #for line in lines:
        #    if (attribute_list is None and i % attributes_divider == 0) or (attribute_list is not None and i in attribute_list):
        #        self.processed_attribute_names.append(re.sub(r'^\d+\.\s*', '', line.strip()))
        #    i += 1

        # choose if attributes names are used or attributes sentences
        if use_processed_concepts:
            self.used_attribute_names = self.processed_attribute_names
        else:
            self.used_attribute_names = self.attribute_names
        #self.used_attribute_names = self.processed_attribute_names if use_processed_concepts else self.attribute_names

        # negative attributes
        #with open(self.negative_processed_attribute_names_file, "r") as f:
        #    lines = f.readlines()
        #i = 0
        #for line in lines:
        #    if (attribute_list is None and i % attributes_divider == 0) or (attribute_list is not None and i in attribute_list):
        #        self.negative_processed_attribute_names.append(re.sub(r'^\d+\.\s*', '', line.strip()))
        #    i += 1

        # select only chosen attributes
        for key in self.image_attributes.keys():
            self.image_attributes[key] = [self.image_attributes[key][attribute_index] 
                                          for attribute_index in self.attribute_numbers]

        #print("image_ids: ", self.image_ids)
        print("Attribute names: ", self.attribute_names)
        # print("--------------------")
        # print("Processed Attributes: ", self.processed_attribute_names)
        for atr in self.used_attribute_names:
            self.negative_processed_attribute_names.append("Not a " + atr)

      

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
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        attributes = self.image_attributes[image_id]
        return image, attributes
    
    def get_image_name_with_labels(self, list_idx):
        """
        Returns the image name and its attributes at the given index
        """
        if type(list_idx) == list:
            for idx in list_idx:
                image_id = self.image_ids[idx]
                print(os.path.join(self.image_dir, self.image_paths[idx]),self.image_attributes[image_id])
            return None
                
        else:
            image_id = self.image_ids[list_idx]
            image_path = os.path.join(self.image_dir, self.image_paths[list_idx])
            attributes = self.image_attributes[image_id] # used as key not idx
            return image_path, attributes, image_id
    
    def calculate_label_ratio(self):
        """
        Calculates the ratio of 0s to 1s in the dataset
        Assumes imagesdivider is 1
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
            self.attribute_encodings.append(get_clip_text_embedding("" + attr, self.model, self.device).float())
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
        image_indices = list(range(self.size()))
        if self.split_images:
            image_test_size_1 = self.image_split_list[1] + self.image_split_list[2]
            image_test_size_2 = round(self.image_split_list[2] / image_test_size_1,1)
            image_train_indices, image_no_train_indices = train_test_split(image_indices, test_size=image_test_size_1, random_state=69)
            image_validation_indices, image_test_indices = train_test_split(image_no_train_indices, test_size=image_test_size_2, random_state=69)
        else: 
            image_train_indices = image_indices
            image_validation_indices = image_indices
            image_test_indices = image_indices
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
            
        print("self.attribute_numbers",self.attribute_numbers)
        print("self.used_attribute_names",self.used_attribute_names)
        #print("self.attribute_encodings",self.attribute_encodings)
        for idx in range(self.size()):
            if idx % 100 == 0:
                print("Currently at index ",idx)
            image, labels = self.get_image_with_labels(idx)
            image_encoding = get_clip_image_embedding(image, self.model).float()
            for attr_idx in range(len(self.attribute_numbers)):
                concept_encoding = self.attribute_encodings[attr_idx]
                if self.negative_concepts:
                    negative_concept_encoding = self.negative_attribute_encodings[attr_idx]
                label = labels[attr_idx]
                if not self.preprocessing and not self.negative_concepts:
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((image_encoding, concept_encoding, label))
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((image_encoding, concept_encoding, label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
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
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((inputs, label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((inputs, label))
                elif not self.preprocessing and self.negative_concepts and not self.double_input:
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
                elif not self.preprocessing and self.negative_concepts and self.double_input:
                    if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
                        training_list.append((image_encoding, concept_encoding, label))
                        training_list.append((image_encoding, negative_concept_encoding, 1 - label))
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((image_encoding, concept_encoding, label))
                        validation_list.append((image_encoding, negative_concept_encoding, 1 - label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
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
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((inputs, label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
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
                    if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
                        validation_list.append((inputs_pos, label))
                        validation_list.append((inputs_neg, 1 - label))
                    if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
                        test_list.append((inputs_pos, label))
                        test_list.append((inputs_neg, 1 - label))
        return training_list, validation_list, test_list


class PreprocessedDataset(Dataset):
    def __init__(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]



    # def size(self):
    #     """
    #     Returns the size of the dataset
    #     """
    #     return len(self.image_paths)

    # def get_image_with_labels(self, idx):
    #     """
    #     Returns the image and its attributes at the given index
    #     """
    #     image_id = self.image_ids[idx]
    #     image_path = self.image_paths[idx]
    #     image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
    #     attributes = self.image_attributes[image_id]
    #     return image, attributes
    
    # def calculate_label_ratio(self):
    #     """
    #     Calculates the ratio of 0s to 1s in the dataset
    #     """
    #     count0 = 0
    #     count1 = 0
    #     for idx in range(self.size()):
    #         _, labels = self.get_image_with_labels(idx)
    #         for label in labels:
    #             if label == 0:
    #                 count0 += 1
    #             else:
    #                 count1 += 1
    #     return count0/count1
    
    # def preprocess_dataset(self):
    #     """
    #     Preprocesses the dataset to create a set of tuples containing image encodings, attribute encodings, and labels
    #     """
    #     training_list = []
    #     validation_list = []
    #     test_list = []
    #     self.attribute_encodings = []
    #     self.negative_attribute_encodings = []
    #     for attr in self.used_attribute_names:
    #         self.attribute_encodings.append(get_clip_text_embedding(attr, self.model, self.device).float())
    #     for attr in self.negative_processed_attribute_names:
    #         self.negative_attribute_encodings.append(get_clip_text_embedding(attr, self.model, self.device).float())
    #     if self.split_attributes:
    #         attribute_test_size_1 = self.attribute_split_list[1] + self.attribute_split_list[2]
    #         attribute_test_size_2 = round(self.attribute_split_list[2] / attribute_test_size_1,1)
    #         attribute_indices = list(range(len(self.attribute_numbers)))
    #         attribute_train_indices, attribute_no_train_indices = train_test_split(attribute_indices, test_size=attribute_test_size_1, random_state=69)
    #         attribute_validation_indices, attribute_test_indices = train_test_split(attribute_no_train_indices, test_size=attribute_test_size_2, random_state=69)
    #     else:
    #         attribute_train_indices = []
    #         attribute_validation_indices = []
    #         attribute_test_indices = []

    #     print("Preprocessing dataset...")
    #     print("Encoding all images...")
    #     image_indices = list(range(self.size()))
    #     if self.split_images:
    #         image_test_size_1 = self.image_split_list[1] + self.image_split_list[2]
    #         image_test_size_2 = round(self.image_split_list[2] / image_test_size_1,1)
    #         image_train_indices, image_no_train_indices = train_test_split(image_indices, test_size=image_test_size_1, random_state=69)
    #         image_validation_indices, image_test_indices = train_test_split(image_no_train_indices, test_size=image_test_size_2, random_state=69)
    #     else: 
    #         image_train_indices = image_indices
    #         image_validation_indices = image_indices
    #         image_test_indices = image_indices
    #     print("Amount of training images: ", len(image_train_indices))
    #     print("Amount of validation images: ", len(image_validation_indices))
    #     print("Amount of test images: ", len(image_test_indices))
    #     if attribute_train_indices != [] and attribute_validation_indices != [] and attribute_test_indices != []:
    #         print("Amount of training attributes: ", len(attribute_train_indices))
    #         print("Amount of validation attributes: ", len(attribute_validation_indices))
    #         print("Amount of test attributes: ", len(attribute_test_indices))
    #     else:
    #         print("Amount of training attributes: ", len(self.attribute_numbers))
    #         print("Amount of validation attributes: ", len(self.attribute_numbers))
    #         print("Amount of test attributes: ", len(self.attribute_numbers))


    #     for idx in range(self.size()):
    #         if idx % 100 == 0:
    #             print("Currently at index ",idx)
    #         image, labels = self.get_image_with_labels(idx)
    #         image_encoding = get_clip_image_embedding(image, self.model).float()
    #         for attr_idx in range(len(self.attribute_numbers)):
    #             concept_encoding = self.attribute_encodings[attr_idx]
    #             negative_concept_encoding = self.negative_attribute_encodings[attr_idx]
    #             label = labels[attr_idx]
    #             if not self.preprocessing and not self.negative_concepts:
    #                 if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
    #                     training_list.append((image_encoding, concept_encoding, label))
    #                 if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
    #                     validation_list.append((image_encoding, concept_encoding, label))
    #                 if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
    #                     test_list.append((image_encoding, concept_encoding, label))
    #             elif self.preprocessing and not self.negative_concepts:
    #                 norm_text = torch.norm(concept_encoding)
    #                 norm_image = torch.norm(image_encoding)
    #                 norm_diff = torch.norm(image_encoding - concept_encoding)
    #                 cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, concept_encoding, dim=1)
    #                 angle = torch.acos(cosine_similarity)
    #                 dot_product = torch.dot(image_encoding.squeeze(), concept_encoding.squeeze())
    #                 inputs = torch.tensor([norm_text, norm_image, norm_diff, angle, dot_product])
    #                 if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
    #                     training_list.append((inputs, label))
    #                 if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
    #                     validation_list.append((inputs, label))
    #                 if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
    #                     test_list.append((inputs, label))
    #             elif not self.preprocessing and self.negative_concepts and not self.double_input:
    #                 if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
    #                     training_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
    #                 if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
    #                     validation_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
    #                 if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
    #                     test_list.append((image_encoding, concept_encoding, negative_concept_encoding, label))
    #             elif not self.preprocessing and self.negative_concepts and self.double_input:
    #                 if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
    #                     training_list.append((image_encoding, concept_encoding, label))
    #                     training_list.append((image_encoding, negative_concept_encoding, 1 - label))
    #                 if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
    #                     validation_list.append((image_encoding, concept_encoding, label))
    #                     validation_list.append((image_encoding, negative_concept_encoding, 1 - label))
    #                 if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
    #                     test_list.append((image_encoding, concept_encoding, label))
    #                     test_list.append((image_encoding, negative_concept_encoding, 1 - label))
    #             elif self.preprocessing and self.negative_concepts and not self.double_input:
    #                 norm_diff = torch.norm(image_encoding - concept_encoding)
    #                 negative_norm_diff = torch.norm(image_encoding - negative_concept_encoding)
    #                 cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, concept_encoding, dim=1)
    #                 negative_cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, negative_concept_encoding, dim=1)
    #                 angle = torch.acos(cosine_similarity)
    #                 negative_angle = torch.acos(negative_cosine_similarity)
    #                 dot_product = torch.dot(image_encoding.squeeze(), concept_encoding.squeeze())
    #                 negative_dot_product = torch.dot(image_encoding.squeeze(), negative_concept_encoding.squeeze())
    #                 inputs = torch.tensor([norm_diff, negative_norm_diff, cosine_similarity, negative_cosine_similarity, angle, negative_angle, dot_product, negative_dot_product])
    #                 if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
    #                     training_list.append((inputs, label))
    #                 if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
    #                     validation_list.append((inputs, label))
    #                 if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
    #                     test_list.append((inputs, label))
    #             elif self.preprocessing and self.negative_concepts and self.double_input:
    #                 norm_diff = torch.norm(image_encoding - concept_encoding)
    #                 negative_norm_diff = torch.norm(image_encoding - negative_concept_encoding)
    #                 cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, concept_encoding, dim=1)
    #                 negative_cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, negative_concept_encoding, dim=1)
    #                 angle = torch.acos(cosine_similarity)
    #                 negative_angle = torch.acos(negative_cosine_similarity) 
    #                 dot_product = torch.dot(image_encoding.squeeze(), concept_encoding.squeeze())
    #                 negative_dot_product = torch.dot(image_encoding.squeeze(), negative_concept_encoding.squeeze())
    #                 inputs_pos = torch.tensor([norm_diff, cosine_similarity, angle, dot_product])
    #                 inputs_neg = torch.tensor([negative_norm_diff, negative_cosine_similarity, negative_angle, negative_dot_product])
    #                 if idx in image_train_indices and (attr_idx in attribute_train_indices or not self.split_attributes):
    #                     training_list.append((inputs_pos, label))
    #                     training_list.append((inputs_neg, 1 - label))
    #                 if idx in image_validation_indices and (attr_idx in attribute_validation_indices or not self.split_attributes):
    #                     validation_list.append((inputs_pos, label))
    #                     validation_list.append((inputs_neg, 1 - label))
    #                 if idx in image_test_indices and (attr_idx in attribute_test_indices or not self.split_attributes):
    #                     test_list.append((inputs_pos, label))
    #                     test_list.append((inputs_neg, 1 - label))
    #     return training_list, validation_list, test_list