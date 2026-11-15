import torch
from torch.utils.data import Dataset
import clip
import os
import torch_directml
from PIL import Image
import time
import matplotlib.pyplot as plt
import re
import os
import numpy as np
from scipy.optimize import curve_fit, root_scalar, minimize
import pickle
import csv
import math
from scipy.optimize import minimize_scalar
from scipy.special import erf

class PreprocessedDataset(Dataset):
    def __init__(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]

class CUB200Dataset(Dataset):
    def __init__(self, root, device, model, preprocess, negative_concepts, 
                 images_percentage, attributes_percentage, attribute_list=None, 
                 use_processed_concepts=True, all_image_attributes=None,
                 image_encoding_labels_list = None):

        self.root = root
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.negative_concepts = negative_concepts
        self.all_image_attributes = all_image_attributes
        self.image_encoding_labels_list = image_encoding_labels_list

        self.image_dir = os.path.join(root, "images")
        self.attribute_file = os.path.join(root, "attributes", "image_attribute_labels.txt")
        self.image_file = os.path.join(root, "images.txt")
        self.attribute_names_file = os.path.join(root, "attributes", "attributes.txt")
        self.processed_attribute_names_file = os.path.join(root, "attributes", "processed_attributes.txt")
        self.negative_processed_attribute_names_file = os.path.join(root, "attributes", "negative_processed_attributes.txt")

        # Load image paths
        # Make lists of the image ids and the image paths that are selected by the images_percentage
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

        # Make a list with all attribute names (not only those selected by attributes_percentage)
        with open(self.attribute_names_file, "r") as f:
            lines = f.readlines()
        self.all_attribute_names = [line.strip().split()[1] for line in lines]

        # Make a list with the attribute names and attribute numbers that are selected by attributes_percentage
        with open(self.attribute_names_file, "r") as f:
            lines = f.readlines()
        i = 0
        for line in lines:
            if (attribute_list is None and i % attributes_divider == 0) or (attribute_list is not None and i in attribute_list):
                parts = line.strip().split()
                self.attribute_numbers.append(int(parts[0]))
                self.attribute_names.append(parts[1])
            i += 1

        # Make a list with the processed attribute names that are selected by attributes_percentage
        with open(self.processed_attribute_names_file, "r") as f:
            lines = f.readlines()
        i = 0
        for line in lines:
            if (attribute_list is None and i % attributes_divider == 0) or (attribute_list is not None and i in attribute_list):
                self.processed_attribute_names.append(re.sub(r'^\d+\.\s*', '', line.strip()))
            i += 1

        self.used_attribute_names_file = self.processed_attribute_names_file if use_processed_concepts else self.attribute_names_file
        self.used_attribute_names = self.processed_attribute_names if use_processed_concepts else self.attribute_names

        # Make a list with the negative processed attribute names that are selected by attributes_percentage
        with open(self.negative_processed_attribute_names_file, "r") as f:
            lines = f.readlines()
        i = 0
        for line in lines:
            if (attribute_list is None and i % attributes_divider == 0) or (attribute_list is not None and i in attribute_list):
                self.negative_processed_attribute_names.append(re.sub(r'^\d+\.\s*', '', line.strip()))
            i += 1

        # Create dictionary with all attribute ids as keys and as value a list of lists of all image ids selected by the images_percentage and the corresponding label
        if self.all_image_attributes is None:
            print("Going through the file with all attribute labels for all images (only happens once), this may take a while...")
            self.all_image_attributes = {}
            with open(self.attribute_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    image_id = int(parts[0])
                    attribute_id = int(parts[1])
                    is_present = int(parts[2])
                    if image_id in self.image_ids:
                        if attribute_id not in self.all_image_attributes:
                            self.all_image_attributes[attribute_id] = [[image_id, is_present]]
                        else:
                            self.all_image_attributes[attribute_id].append([image_id, is_present])
            print("Finished going through the file with all attribute labels for all images", "\n")

        # Create dictionary with as keys the image ids selected by images_percentage and as value a list with the length
        # of the total number of attributes and at index i the label of the attribute with id i+1
        self.image_attributes = {}
        for attribute_id in self.attribute_numbers:
            for image_id, is_present in self.all_image_attributes[attribute_id]:
                if image_id not in self.image_attributes:
                    self.image_attributes[image_id] = [-1] * len(self.all_attribute_names)
                self.image_attributes[image_id][attribute_id - 1] = is_present

        # For each list in the dictionary, remove the -1 values
        # This means that for a certain image_id, self.image_attributes[image_id] will be a list with at index i the label of the attribute with id self.attribute_numbers[i]
        for key in self.image_attributes:
            self.image_attributes[key] = [x for x in self.image_attributes[key] if x != -1]

    def size(self):
        """
        Returns the size of the dataset
        """
        return len(self.image_paths)
    
    def get_image(self, idx):
        """
        Returns the image at the given index
        """
        image_path = self.image_paths[idx]
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        return image

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
        self.attribute_encodings = []
        for attr in self.used_attribute_names:
            self.attribute_encodings.append(get_clip_text_embedding(attr, self.model, self.device).float())
        if self.negative_concepts:
            self.negative_attribute_encodings = []
            for attr in self.negative_processed_attribute_names:
                self.negative_attribute_encodings.append(get_clip_text_embedding(attr, self.model, self.device).float())
        preprocessed_data = set()
        
        if self.image_encoding_labels_list is None:
            print("Going through the dataset to get all image encodings and labels (only happens once), this may take a while...")
            self.image_encoding_labels_list = []
            for idx in range(self.size()):
                if idx % 100 == 0:
                    print("Currently at index ", idx, "out of ", self.size())
                image = self.get_image(idx)
                image_encoding = get_clip_image_embedding(image, self.model).float()
                self.image_encoding_labels_list.append(image_encoding)
            print("Finished going through the dataset to get all image encodings and labels", "\n")

        for idx in range(self.size()):
            image_encoding = self.image_encoding_labels_list[idx]
            image_id = self.image_ids[idx]
            for attr_idx in range(len(self.used_attribute_names)):
                concept_encoding = self.attribute_encodings[attr_idx]
                if self.negative_concepts:
                    negative_concept_encoding = self.negative_attribute_encodings[attr_idx]
                label = self.image_attributes[image_id][attr_idx]
                cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, concept_encoding, dim=1)
                if not self.negative_concepts:
                    inputs = torch.tensor([cosine_similarity])
                else:
                    negative_cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, negative_concept_encoding, dim=1)
                    inputs = torch.tensor([cosine_similarity - negative_cosine_similarity])
                preprocessed_data.add((inputs, label))

        return preprocessed_data
    


class VOC2012Dataset(Dataset):
    def __init__(self, root, device, model, preprocess, negative_concepts, 
                 images_percentage, attributes_percentage, attribute_list=None, 
                 use_processed_concepts=True, all_image_attributes=None,
                 image_encoding_labels_list = None):
        
        negative_concepts = False # Overwrite negative_concepts to False

        self.root = root
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.negative_concepts = negative_concepts
        self.all_image_attributes = all_image_attributes
        self.image_encoding_labels_list = image_encoding_labels_list

        self.image_dir = os.path.join(root, "JPEGImages")
        self.attribute_dir = os.path.join(root, "ImageSets","Main")
        self.image_file_part1 = os.path.join(root, "ImageSets/Main/train.txt")
        self.image_file_part2 = os.path.join(root, "ImageSets/Main/val.txt")
        self.image_file_part3 = os.path.join(root, "ImageSets/Main/trainval.txt")

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
                    self.attribute_numbers.append(i+1)
                    self.attribute_names.append(self.all_attribute_names[i])
        else:
            for i in range(len(self.all_attribute_names)):
                if i in attribute_list:
                    self.attribute_numbers.append(i+1)
                    self.attribute_names.append(self.all_attribute_names[i])

        self.processed_attribute_names = self.attribute_names
        
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

        # Create dictionary with all attribute ids as keys and as value a list of lists of all image ids selected by the images_percentage and the corresponding label
        if self.all_image_attributes is None:
            print("Going through the file with all attribute labels for all images (only happens once), this may take a while...")
            self.all_image_attributes = {}
            # Loop over attributes
            for i, name in enumerate(self.all_attribute_names):
                attribute_id = i+1
                
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

                                if attribute_id not in self.all_image_attributes:
                                    self.all_image_attributes[attribute_id] = [[image_id, is_present]]
                                else:
                                    if [image_id, is_present] not in self.all_image_attributes[attribute_id]:
                                        self.all_image_attributes[attribute_id].append([image_id, is_present])
                                    
            print("Finished going through the file with all attribute labels for all images", "\n")

        # Create dictionary with as keys the image ids selected by images_percentage and as value a list with the length
        # of the total number of attributes and at index i the label of the attribute with id i+1
        self.image_attributes = {}
        for attribute_id in self.attribute_numbers:
            for image_id, is_present in self.all_image_attributes[attribute_id]:
                if image_id not in self.image_attributes:
                    self.image_attributes[image_id] = [-1] * len(self.all_attribute_names)
                self.image_attributes[image_id][attribute_id - 1] = is_present

        # For each list in the dictionary, remove the -1 values
        # This means that for a certain image_id, self.image_attributes[image_id] will be a list with at index i the label of the attribute with id self.attribute_numbers[i]
        for key in self.image_attributes:
            self.image_attributes[key] = [x for x in self.image_attributes[key] if x != -1]

        # self.image_attributes = dict()
        # # Initialize empty attribute vectors for selected image IDs
        # for image_id in self.image_ids:
        #     self.image_attributes[image_id] = [-1] * len(self.attribute_numbers)

        # # Loop over attributes
        # for i, name in enumerate(self.all_attribute_names):
        #     attribute_index = i
            
        #     for split in ["train", "val", "trainval"]:
        #         file_path = os.path.join(self.attribute_dir, f"{name}_{split}.txt")
        #         if not os.path.exists(file_path):
        #             continue

        #         with open(file_path, "r") as file:
        #             for line in file:
        #                 parts = line.strip().split()
        #                 image_id = parts[0]
        #                 is_present = 0 if int(parts[1]) == -1 else 1

        #                 if image_id in self.image_ids:
        #                     self.image_attributes[image_id][i] = is_present
        # for key in self.image_attributes:
        #     self.image_attributes[key] = [
        #         val if val != -1 else 0  # or just skip
        #         for val in self.image_attributes[key]
        #     ]

        # #print("image_ids: ", self.image_ids)
        # print("Attribute numbers: ", self.attribute_numbers)
        # print("Attribute names: ", self.attribute_names)

        # for atr in self.used_attribute_names:
        #     self.negative_processed_attribute_names.append("Not a " + atr)

        # # print("--------------------")
        # # print("Processed Attributes: ", self.processed_attribute_names)

    def size(self):
        """
        Returns the size of the dataset
        """
        return len(self.image_paths)

    def get_image(self, idx):
        """
        Returns the image at the given index
        """
        image_path = self.image_paths[idx]
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        return image

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
        self.attribute_encodings = []
        for attr in self.used_attribute_names:
            self.attribute_encodings.append(get_clip_text_embedding(attr, self.model, self.device).float())
        if self.negative_concepts:
            self.negative_attribute_encodings = []
            for attr in self.negative_processed_attribute_names:
                self.negative_attribute_encodings.append(get_clip_text_embedding(attr, self.model, self.device).float())
        preprocessed_data = set()
        
        if self.image_encoding_labels_list is None:
            print("Going through the dataset to get all image encodings and labels (only happens once), this may take a while...")
            self.image_encoding_labels_list = []
            for idx in range(self.size()):
                if idx % 100 == 0:
                    print("Currently at index ", idx, "out of ", self.size())
                image = self.get_image(idx)
                image_encoding = get_clip_image_embedding(image, self.model).float()
                self.image_encoding_labels_list.append(image_encoding)
            print("Finished going through the dataset to get all image encodings and labels", "\n")

        for idx in range(self.size()):
            image_encoding = self.image_encoding_labels_list[idx]
            image_id = self.image_ids[idx]
            for attr_idx in range(len(self.used_attribute_names)):
                concept_encoding = self.attribute_encodings[attr_idx]
                if self.negative_concepts:
                    negative_concept_encoding = self.negative_attribute_encodings[attr_idx]
                label = self.image_attributes[image_id][attr_idx]
                cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, concept_encoding, dim=1)
                if not self.negative_concepts:
                    inputs = torch.tensor([cosine_similarity])
                else:
                    negative_cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, negative_concept_encoding, dim=1)
                    inputs = torch.tensor([cosine_similarity - negative_cosine_similarity])
                preprocessed_data.add((inputs, label))

        return preprocessed_data




class celebADataset(Dataset):
    def __init__(self, root, device, model, preprocess, negative_concepts, 
                 images_percentage, attributes_percentage, attribute_list=None, 
                 use_processed_concepts=True, all_image_attributes=None,
                 image_encoding_labels_list = None):

        negative_concepts = False # Overwrite negative_concepts to False

        self.root = root
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.negative_concepts = negative_concepts
        self.all_image_attributes = all_image_attributes
        self.image_encoding_labels_list = image_encoding_labels_list

        self.image_dir = os.path.join(root, "Celebrity_Faces_Dataset")
        self.attribute_dir = os.path.join(root, "list_attr_celeba.csv")
        
        #self.processed_attribute_names_file = os.path.join(root, "attributes", "processed_attributes.txt")
        #self.negative_processed_attribute_names_file = os.path.join(root, "attributes", "negative_processed_attributes.txt")

        # Load image paths
        images_divider = 100 // images_percentage
        attributes_divider = 100 // attributes_percentage
        
        self.image_ids = []
        self.image_paths = []
      
        # Load attribute names and attribute values
        self.all_attribute_names = []
        self.attribute_names = []
        self.attribute_numbers = []
        self.processed_attribute_names = []
        self.negative_processed_attribute_names = []

        self.all_image_attributes = {}
        with open(self.attribute_dir, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            first_row = next(reader)  # Read the header row
            self.all_attribute_names = [text.replace("_"," ") for text in first_row[1:]]
            image_id = 1
            image_index = 0
            for row in reader:
                if image_index >= 50000:
                    break
                if image_index % images_divider == 0:
                    self.image_ids.append(image_id)
                    self.image_paths.append(os.path.join(self.image_dir, row[0]))  # Assuming the first column contains the image paths
                    for attribute_id in range(1, len(self.all_attribute_names) + 1):
                        if attribute_id not in self.all_image_attributes:
                            self.all_image_attributes[attribute_id] = [[image_id, 1 if int(row[attribute_id]) == 1 else 0]]
                        else:
                            self.all_image_attributes[attribute_id].append([image_id, 1 if int(row[attribute_id]) == 1 else 0])
                image_id += 1
                image_index += 1

        if (attribute_list is None):
            for i in range(len(self.all_attribute_names)):
                if i % attributes_divider == 0:
                    self.attribute_numbers.append(i+1)
                    self.attribute_names.append(self.all_attribute_names[i])
        else:
            for i in range(len(self.all_attribute_names)):
                if i in attribute_list:
                    self.attribute_numbers.append(i+1)
                    self.attribute_names.append(self.all_attribute_names[i])

        # Create dictionary with as keys the image ids selected by images_percentage and as value a list with the length
        # of the total number of attributes and at index i the label of the attribute with id i+1
        self.image_attributes = {}
        for attribute_id in self.attribute_numbers:
            for image_id, is_present in self.all_image_attributes[attribute_id]:
                if image_id not in self.image_attributes:
                    self.image_attributes[image_id] = [-1] * len(self.all_attribute_names)
                self.image_attributes[image_id][attribute_id - 1] = is_present

        # For each list in the dictionary, remove the -1 values
        # This means that for a certain image_id, self.image_attributes[image_id] will be a list with at index i the label of the attribute with id self.attribute_numbers[i]
        for key in self.image_attributes:
            self.image_attributes[key] = [x for x in self.image_attributes[key] if x != -1]

        self.processed_attribute_names = self.attribute_names
        
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

        # # select only chosen attributes
        # for key in self.image_attributes.keys():
        #     self.image_attributes[key] = [self.image_attributes[key][attribute_index] 
        #                                   for attribute_index in self.attribute_numbers]

        # #print("image_ids: ", self.image_ids)
        # print("Attribute names: ", self.attribute_names)
        # # print("--------------------")
        # # print("Processed Attributes: ", self.processed_attribute_names)
        # for atr in self.used_attribute_names:
        #     self.negative_processed_attribute_names.append("Not a " + atr)

    def size(self):
        """
        Returns the size of the dataset
        """
        return len(self.image_paths)
    
    def get_image(self, idx):
        """
        Returns the image at the given index
        """
        image_path = self.image_paths[idx]
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        return image

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
        self.attribute_encodings = []
        for attr in self.used_attribute_names:
            self.attribute_encodings.append(get_clip_text_embedding(attr, self.model, self.device).float())
        if self.negative_concepts:
            self.negative_attribute_encodings = []
            for attr in self.negative_processed_attribute_names:
                self.negative_attribute_encodings.append(get_clip_text_embedding(attr, self.model, self.device).float())
        preprocessed_data = set()
        
        if self.image_encoding_labels_list is None:
            print("Going through the dataset to get all image encodings and labels (only happens once), this may take a while...")
            self.image_encoding_labels_list = []
            for idx in range(self.size()):
                if idx % 100 == 0:
                    print("Currently at index ", idx, "out of ", self.size())
                image = self.get_image(idx)
                image_encoding = get_clip_image_embedding(image, self.model).float()
                self.image_encoding_labels_list.append(image_encoding)
            print("Finished going through the dataset to get all image encodings and labels", "\n")

        for idx in range(self.size()):
            image_encoding = self.image_encoding_labels_list[idx]
            image_id = self.image_ids[idx]
            for attr_idx in range(len(self.used_attribute_names)):
                concept_encoding = self.attribute_encodings[attr_idx]
                if self.negative_concepts:
                    negative_concept_encoding = self.negative_attribute_encodings[attr_idx]
                label = self.image_attributes[image_id][attr_idx]
                cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, concept_encoding, dim=1)
                if not self.negative_concepts:
                    inputs = torch.tensor([cosine_similarity])
                else:
                    negative_cosine_similarity = torch.nn.functional.cosine_similarity(image_encoding, negative_concept_encoding, dim=1)
                    inputs = torch.tensor([cosine_similarity - negative_cosine_similarity])
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
    
def single_gaussian_approximation(labelled_data):
    """
    This function takes as input a list of data points (floats) and fits as good as possible to this data
    a Gaussian, from which mu and sigma are unknown in advance.
    """
    data = [x[0] for x in labelled_data]

    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    
    # Create a histogram from the data.
    counts, bin_edges = np.histogram(data, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Set initial guesses for the parameters.
    amp_guess = np.max(counts)
    mu_guess = np.mean(data)
    sigma_guess = np.std(data)

    # Rough initial estimates for the Gaussian.
    initial_guess = [amp_guess, mu_guess, sigma_guess]

    # Define bounds for the parameters.
    lower_bounds = [0, min(data), 0]  # Ensure amplitude and sigma is non-negative
    upper_bounds = [np.inf, max(data), np.inf]

    # Use the constrained function for curve fitting
    try:
        solution_parameters, _ = curve_fit(gaussian, bin_centers, counts, p0=initial_guess, bounds=(lower_bounds, upper_bounds))
        # Calculate the goodness of fit (R-squared)
        residuals = counts - gaussian(bin_centers, *solution_parameters)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((counts - np.mean(counts))**2)
        r_squared = 1 - (ss_res / ss_tot)
    except RuntimeError:
        print("Error - single curve_fit failed")
        solution_parameters = initial_guess
        r_squared = float("inf")  # Indicate that the fit was unsuccessful

    return solution_parameters, r_squared

def double_gaussian_approximation(labelled_data, create_ranking=False, plot_histograms=False):
    """
    This function takes as input a list of data points (floats) and fits as good as possible to this data
    a sum of 2 Gaussians, from which mu and sigma are unknown in advance.
    This double Gaussian is then plotted, along with the histogram of the data points.
    """
    data = [x[0] for x in labelled_data]
    max_retries = 5

    def constrained_double_gaussian(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
        # Compute the basic Gaussian components
        g1 = amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
        g2 = amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
        
        # Initialize penalty
        penalty = 0.0

        # Constraint 1: The sigmas shouldn't differ too much
        if sigma1 > 3 * sigma2:
            penalty += 1e6 * (sigma1 - 3 * sigma2)

        if sigma2 > 3 * sigma1:
            penalty += 1e6 * (sigma2 - 3 * sigma1)

        # Constraint 2: If one Gaussian is much larger, it should have the larger sigma
        if amp1 > 2 * amp2 and sigma1 < sigma2:
            amp1_2 = amp1 / amp2
            if sigma1 * (1 + amp1_2 / 16) < sigma2:
                penalty += 1e3 * (sigma2 - sigma1 * (1 + amp1_2 / 16))

        if amp2 > 2 * amp1 and sigma2 < sigma1:
            amp2_1 = amp2 / amp1
            if sigma2 * (1 + amp2_1 / 16) < sigma1:
                penalty += 1e3 * (sigma1 - sigma2 * (1 + amp2_1 / 16))

        # # Constraint 3: If one is much smaller, centers should be sufficiently apart
        # if amp1 > 2 * amp2 and abs(mu1 - mu2) < sigma1:
        #     amp1_2 = amp1 / amp2
        #     if abs(mu1 - mu2) < (1.5 + amp1_2/16) * sigma1:
        #         penalty += 1e3 * ((1.5 + amp1_2/16) * sigma1 - abs(mu1 - mu2))
        # if amp2 > 2 * amp1 and abs(mu1 - mu2) < sigma2:
        #     amp2_1 = amp2 / amp1
        #     if abs(mu1 - mu2) < (1.5 + amp2_1/16) * sigma2:
        #         penalty += 1e3 * ((1.5 + amp2_1/16) * sigma2 - abs(mu1 - mu2))

        return g1 + g2 + penalty


    # Create a histogram from the data.
    counts, bin_edges = np.histogram(data, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Set initial guesses for the parameters.
    amp_guess = np.max(counts)
    mu_guess = np.mean(data)
    sigma_guess = np.std(data)

    # Rough initial estimates for both Gaussians.
    base_guess = [amp_guess * 0.67, mu_guess - sigma_guess / 2, sigma_guess,
                  amp_guess * 0.33, mu_guess + sigma_guess / 2, sigma_guess]
    
    simple_guess = [amp_guess * 0.5, mu_guess, sigma_guess, 
                    amp_guess * 0.5, mu_guess, sigma_guess]
    
    extreme_guess_1 = [amp_guess * 0.25, mu_guess - 2.5 * sigma_guess, sigma_guess,
                       amp_guess * 0.75, mu_guess + 2.5 * sigma_guess, sigma_guess]
    
    extreme_guess_2 = [amp_guess * 0.75, mu_guess - 2.5 * sigma_guess, sigma_guess,
                       amp_guess * 0.25, mu_guess + 2.5 * sigma_guess, sigma_guess]
    
    extra_guess_list = [simple_guess, extreme_guess_1, extreme_guess_2]
    extra_guess_r_list = []
    extra_guess_parameters_list = []

    random_guess_r_list = []
    random_guess_parameters_list = []

    lower_bounds = [0, min(data), 0, 0, min(data), 0]
    upper_bounds = [np.inf, max(data), np.inf, np.inf, max(data), np.inf]

    # Retry fitting with random perturbations if it fails
    success = False
    for attempt in range(max_retries + len(extra_guess_list)):
        try:
            if attempt == 0:
                guess = base_guess
            elif attempt <= len(extra_guess_list):
                guess = extra_guess_list[attempt - 1]
            else:
                # Apply small random perturbations with a fixed seed for reproducibility
                np.random.seed(69)
                guess = [p * np.random.uniform(0.5, 1.5) if i % 3 != 1 else p + np.random.normal(0, sigma_guess * 0.5)
                         for i, p in enumerate(simple_guess)]

            solution_parameters, _ = curve_fit(
                constrained_double_gaussian,
                bin_centers, counts,
                p0=guess, bounds=(lower_bounds, upper_bounds),
                maxfev=2000
            )
            success = True
            # Calculate R²
            residuals = counts - constrained_double_gaussian(bin_centers, *solution_parameters)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((counts - np.mean(counts))**2)
            r_squared = 1 - (ss_res / ss_tot)

            if attempt == 0:
                # print(f"Attempt {attempt + 1}: succeeded with base guess.")
                # print("The base guess was used.")
                break

            if attempt > 0 and attempt <= len(extra_guess_list):
                # print(f"Attempt {attempt + 1}: succeeded with one of the extra_guesses.")
                extra_guess_r_list.append(r_squared)
                extra_guess_parameters_list.append(solution_parameters)
            elif attempt > len(extra_guess_list):
                # print(f"Attempt {attempt + 1}: succeeded with one of the random guesses.")
                random_guess_r_list.append(r_squared)
                random_guess_parameters_list.append(solution_parameters)

            if attempt == len(extra_guess_list):
                solution_parameters = extra_guess_parameters_list[extra_guess_r_list.index(max(extra_guess_r_list))]
                # print("The best extra guess was used.")
                break
            if attempt == max_retries + len(extra_guess_list) - 1:
                solution_parameters = random_guess_parameters_list[random_guess_r_list.index(max(random_guess_r_list))]
                # print("The best random guess was used.")
                break

        except RuntimeError:
            if attempt == 0:
                # print(f"Attempt {attempt + 1}: failed with base guess.")
                pass
            elif attempt > 0 and attempt <= len(extra_guess_list):
                # print(f"Attempt {attempt + 1}: failed with one of the extra_guesses.")
                extra_guess_r_list.append(-1)
                extra_guess_parameters_list.append(None)
            elif attempt > len(extra_guess_list):
                # print(f"Attempt {attempt + 1}: failed with one of the random guesses.")
                random_guess_r_list.append(-1)
                random_guess_parameters_list.append(None)

            if attempt == len(extra_guess_list) and max(extra_guess_r_list) != -1:
                solution_parameters = extra_guess_parameters_list[extra_guess_r_list.index(max(extra_guess_r_list))]
                # print("The best extra guess was used.")
                break
            if attempt == max_retries + len(extra_guess_list) - 1 and max(random_guess_r_list) != -1:
                solution_parameters = random_guess_parameters_list[random_guess_r_list.index(max(random_guess_r_list))]
                # print("The best random guess was used.")
                break

    if not success:
        # print("All fitting attempts failed. Using initial guess.")
        solution_parameters = base_guess

    # Calculate R²
    residuals = counts - constrained_double_gaussian(bin_centers, *solution_parameters)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((counts - np.mean(counts))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # # Print out the parameters for both Gaussians.
    # print(f"Gaussian 1: amp = {solution_parameters[0]}, mu = {solution_parameters[1]}, sigma = {solution_parameters[2]}")
    # print(f"Gaussian 2: amp = {solution_parameters[3]}, mu = {solution_parameters[4]}, sigma = {solution_parameters[5]}")

    # Prepare data for plotting the fitted curve.
    x_fit = np.linspace(min(data), max(data), 500)
    y_fit = constrained_double_gaussian(x_fit, *solution_parameters)

    # Compute individual Gaussian curves.
    gaussian1 = solution_parameters[0] * np.exp(-((x_fit - solution_parameters[1]) ** 2) / (2 * solution_parameters[2] ** 2))
    gaussian2 = solution_parameters[3] * np.exp(-((x_fit - solution_parameters[4]) ** 2) / (2 * solution_parameters[5] ** 2))

    # Separate data into positive and negative based on labels.
    positive_data = [x[0] for x in labelled_data if x[1] == 1]
    negative_data = [x[0] for x in labelled_data if x[1] == 0]
    all_data = [x[0] for x in labelled_data]

    if not create_ranking and plot_histograms:

        # Plot the histogram and the fitted curves side by side
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))

        # Plot for positive and negative data
        axs[0].hist(positive_data, bins=50, alpha=0.6, color='b', label='Positive Data')
        axs[0].hist(negative_data, bins=50, alpha=0.6, color='y', label='Negative Data')
        axs[0].plot(x_fit, gaussian1, 'b--', label='Gaussian 1')
        axs[0].plot(x_fit, gaussian2, 'm--', label='Gaussian 2')
        axs[0].set_xlabel('Data values')
        axs[0].set_ylabel('Counts')
        axs[0].set_title('Positive and Negative Data')
        axs[0].legend()

        # Plot for all data with fitted double Gaussian
        axs[1].hist(all_data, bins=50, alpha=0.6, color='g', label='All Data')
        axs[1].plot(x_fit, y_fit, 'r-', label='Fitted Double Gaussian')
        axs[1].set_xlabel('Data values')
        axs[1].set_ylabel('Counts')
        axs[1].set_title('All Data with Fitted Double Gaussian')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    return solution_parameters, r_squared

def double_gaussian_approximation_given_one_gaussian(labelled_data, single_gaussian_parameters, create_ranking=False, plot_histograms=False):
    """
    This function takes as input a list of data points (floats) and fits as good as possible to this data
    a sum of 2 Gaussians, from which one of the two Gausians is already known (mu and sigma)
    """
    data = [x[0] for x in labelled_data]
    amp2, mu2, sigma2 = single_gaussian_parameters

    def constrained_double_gaussian_with_one_fixed(x, amp1, mu1, sigma1):
        # Compute the basic Gaussian components
        g1 = amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
        g2 = amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))

        # Initialize penalty
        penalty = 0.0

        # Constraint 1: The sigmas shouldn't differ too much
        if sigma1 > 2 * sigma2:
            penalty += 1e6 * (sigma1 - 3 * sigma2)

        if sigma2 > 2 * sigma1:
            penalty += 1e6 * (sigma2 - 3 * sigma1)

        return g1 + g2 + penalty
    
    # Create a histogram from the data.
    counts, bin_edges = np.histogram(data, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Set initial guesses for the parameters.
    amp_guess = np.max(counts)
    mu_guess = np.mean(data)
    sigma_guess = np.std(data)

    # Rough initial estimates for the Gaussian.
    initial_guess = [amp_guess, mu_guess, sigma_guess]

    # Define bounds for the parameters.
    lower_bounds = [0, min(data), 0]  # Ensure amplitude and sigma is non-negative
    upper_bounds = [np.inf, max(data), np.inf]

    # Use the constrained function for curve fitting
    try:
        solution_parameters, _ = curve_fit(constrained_double_gaussian_with_one_fixed, bin_centers, counts, p0=initial_guess, bounds=(lower_bounds, upper_bounds))
        # Calculate the goodness of fit (R-squared)
        residuals = counts - constrained_double_gaussian_with_one_fixed(bin_centers, *solution_parameters)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((counts - np.mean(counts))**2)
        r_squared = 1 - (ss_res / ss_tot)
    except RuntimeError:
        # print("Error - double curve_fit failed")
        solution_parameters = initial_guess
        residuals = counts - constrained_double_gaussian_with_one_fixed(bin_centers, *solution_parameters)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((counts - np.mean(counts))**2)
        r_squared = 1 - (ss_res / ss_tot)

    # # Print out the parameters for both Gaussians.
    # print(f"Gaussian 1: amp = {solution_parameters[0]}, mu = {solution_parameters[1]}, sigma = {solution_parameters[2]}")
    # print(f"Gaussian 2: amp = {amp2}, mu = {mu2}, sigma = {sigma2}")

    # Prepare data for plotting the fitted curve.
    x_fit = np.linspace(min(data), max(data), 500)
    y_fit = constrained_double_gaussian_with_one_fixed(x_fit, *solution_parameters)

    # Compute individual Gaussian curves.
    gaussian1 = solution_parameters[0] * np.exp(-((x_fit - solution_parameters[1]) ** 2) / (2 * solution_parameters[2] ** 2))
    gaussian2 = amp2 * np.exp(-((x_fit - mu2) ** 2) / (2 * sigma2 ** 2))

    # Separate data into positive and negative based on labels.
    positive_data = [x[0] for x in labelled_data if x[1] == 1]
    negative_data = [x[0] for x in labelled_data if x[1] == 0]
    all_data = [x[0] for x in labelled_data]

    if not create_ranking and plot_histograms:

        # Plot the histogram and the fitted curves side by side
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))

        # Plot for positive and negative data
        axs[0].hist(positive_data, bins=50, alpha=0.6, color='b', label='Positive Data')
        axs[0].hist(negative_data, bins=50, alpha=0.6, color='y', label='Negative Data')
        axs[0].plot(x_fit, gaussian1, 'b--', label='Gaussian 1')
        axs[0].plot(x_fit, gaussian2, 'm--', label='Gaussian 2')
        axs[0].set_xlabel('Data values')
        axs[0].set_ylabel('Counts')
        axs[0].set_title('Positive and Negative Data, BOTH GAUSSIANS CORRECTED')
        axs[0].legend()

        # Plot for all data with fitted double Gaussian
        axs[1].hist(all_data, bins=50, alpha=0.6, color='g', label='All Data')
        axs[1].plot(x_fit, y_fit, 'r-', label='Fitted Double Gaussian')
        axs[1].set_xlabel('Data values')
        axs[1].set_ylabel('Counts')
        axs[1].set_title('All Data with Fitted Double Gaussian, BOTH GAUSSIANS CORRECTED')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    return solution_parameters, r_squared


    
def double_gaussian_correction(labelled_data, double_gaussian_parameters, r_squared, create_ranking=False, plot_histograms=False, per_attribute=True):
    """
    This function tries to correct the double Gaussian parameters if they are not good enough.
    """
    data = [x[0] for x in labelled_data]
    amp1, mu1, sigma1, amp2, mu2, sigma2 = double_gaussian_parameters
    if per_attribute:
        sigma_scale = 1.5
    else:
        sigma_scale = 1.25
    if (amp1 > 1.5*amp2 and sigma2 > sigma_scale*sigma1) or (amp2 > 1.5*amp1 and sigma1 > sigma_scale*sigma2):
        old_amp1 = amp1
        old_amp2 = amp2
        sigma_good = min(sigma1, sigma2)
        mu_good = mu1 if amp1 > amp2 else mu2
        mu_bad = mu2 if amp1 > amp2 else mu1
        if mu_bad > mu_good:
            safe_max = np.mean(sorted(data, reverse=True)[:10]) if len(data) > 10 else max(data)
            right_end_data = [x for x in data if x >= safe_max - 3*sigma_good]
            min_right_end_data = min(right_end_data)
            centered_right_end_data = [x - min_right_end_data for x in right_end_data]
            centered_mirrored_right_end_data = [-x for x in centered_right_end_data]
            mirrored_right_end_data = [x + min_right_end_data for x in centered_mirrored_right_end_data]
            double_end_data = mirrored_right_end_data[:-1] + right_end_data
        else:
            safe_min = np.mean(sorted(data)[:10]) if len(data) > 10 else min(data)
            left_end_data = [x for x in data if x <= safe_min + 3*sigma_good]
            max_left_end_data = max(left_end_data)
            centered_left_end_data = [x - max_left_end_data for x in left_end_data]
            centered_mirrored_left_end_data = [-x for x in centered_left_end_data]
            mirrored_left_end_data = [x + max_left_end_data for x in centered_mirrored_left_end_data]
            double_end_data = left_end_data + mirrored_left_end_data[1:]

        # Fit a Gaussian with fixed sigma to the double_right_end_data
        def fixed_sigma_gaussian(x, amp, mu, sigma):
            return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        # Create a histogram from the double_right_end_data
        counts, bin_edges = np.histogram(double_end_data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Set initial guesses for amp and mu
        amp_guess = np.max(counts)
        mu_guess = np.mean(double_end_data)
        sigma_guess = sigma_good

        # Fit the Gaussian with fixed sigma
        try:
            popt, _ = curve_fit(fixed_sigma_gaussian, bin_centers, counts, p0=[amp_guess, mu_guess, sigma_guess])
            amp_fit, mu_fit, sigma_fit = popt
            # print(f"Fitted Gaussian parameters: amp = {amp_fit}, mu = {mu_fit}")
        except RuntimeError:
            # print("Error - Gaussian fit with fixed sigma failed")
            amp_fit, mu_fit, sigma_fit = amp_guess, mu_guess, sigma_guess

        if amp1 == max(amp1, amp2):
            amp2 = amp_fit
            mu2 = mu_fit
            sigma2 = sigma_fit
        else:
            amp1 = amp_fit
            mu1 = mu_fit
            sigma1

        corrected_string = "ONE GAUSSIAN CORRECTED"

        x_fit = np.linspace(min(data), max(data), 500)

        # Compute individual Gaussian curves.
        gaussian1 = amp1 * np.exp(-((x_fit - mu1) ** 2) / (2 * sigma1 ** 2))
        gaussian2 = amp2 * np.exp(-((x_fit - mu2) ** 2) / (2 * sigma2 ** 2))
        y_fit = gaussian1 + gaussian2

        # Separate data into positive and negative based on labels.
        positive_data = [x[0] for x in labelled_data if x[1] == 1]
        negative_data = [x[0] for x in labelled_data if x[1] == 0]
        all_data = [x[0] for x in labelled_data]

        if not create_ranking and plot_histograms:

            # Plot the histogram and the fitted curves side by side
            fig, axs = plt.subplots(1, 2, figsize=(16, 5))

            # Plot for positive and negative data
            axs[0].hist(positive_data, bins=50, alpha=0.6, color='b', label='Positive Data')
            axs[0].hist(negative_data, bins=50, alpha=0.6, color='y', label='Negative Data')
            axs[0].plot(x_fit, gaussian1, 'b--', label='Gaussian 1')
            axs[0].plot(x_fit, gaussian2, 'm--', label='Gaussian 2')
            axs[0].set_xlabel('Data values')
            axs[0].set_ylabel('Counts')
            axs[0].set_title('Positive and Negative Data, ' + corrected_string)
            axs[0].legend()

            # Plot for all data with fitted double Gaussian
            axs[1].hist(all_data, bins=50, alpha=0.6, color='g', label='All Data')
            axs[1].plot(x_fit, y_fit, 'r-', label='Fitted Double Gaussian')
            axs[1].set_xlabel('Data values')
            axs[1].set_ylabel('Counts')
            axs[1].set_title('All Data with Fitted Double Gaussian, ' + corrected_string)
            axs[1].legend()

            plt.tight_layout()
            plt.show()

        if old_amp1 == max(old_amp1, old_amp2):
            [amp1, mu1, sigma1], r_squared = \
                double_gaussian_approximation_given_one_gaussian(labelled_data, [amp2, mu2, sigma2], create_ranking, plot_histograms)
        else:
            [amp2, mu2, sigma2], r_squared = \
                double_gaussian_approximation_given_one_gaussian(labelled_data, [amp1, mu1, sigma1], create_ranking, plot_histograms)

    return [amp1, mu1, sigma1, amp2, mu2, sigma2], r_squared


def double_gaussian_intersection(double_gaussian_parameters):
    """
    This function takes as input the parameters of the double Gaussian
    and returns the x-coordinate and y_coordinate of an intersection point of the two Gaussians.
    """
    amp1, mu1, sigma1, amp2, mu2, sigma2 = double_gaussian_parameters

    # The two Gaussian functions
    def gaussian1(x):
        return amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))

    def gaussian2(x):
        return amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))

    # Find the intersection point of the two Gaussians
    def difference(x):
        return gaussian1(x) - gaussian2(x)

    # # Plot the difference function
    # x_vals = np.linspace(min(mu1, mu2) - 3 * max(sigma1, sigma2), max(mu1, mu2) + 3 * max(sigma1, sigma2), 500)
    # y_vals = [difference(x) for x in x_vals]

    # plt.figure(figsize=(8, 6))
    # plt.plot(x_vals, y_vals, label="Difference Function")
    # plt.axhline(0, color='red', linestyle='--', label="y = 0")
    # plt.xlabel("x")
    # plt.ylabel("Difference (Gaussian1 - Gaussian2)")
    # plt.title("Difference Function of Two Gaussians")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Dynamically adjust the range to evaluate the difference function
    lower_bound = min(mu1, mu2) - 1 * max(sigma1, sigma2)
    upper_bound = max(mu1, mu2) + 1 * max(sigma1, sigma2)
    x_vals = np.linspace(lower_bound, upper_bound, 2000)
    y_vals = [difference(x) for x in x_vals]

    # Find all the points where the difference function crosses the x-axis
    crossings = []
    for i in range(len(x_vals) - 1):
        if y_vals[i] * y_vals[i + 1] < 0:  # Sign change indicates a crossing
            # Linear interpolation to find the zero crossing
            x_cross = x_vals[i] - y_vals[i] * (x_vals[i + 1] - x_vals[i]) / (y_vals[i + 1] - y_vals[i])
            crossings.append(x_cross)

    # If crossings are found, pick the one with the highest Gaussian1(x) value
    if crossings:
        intersection_x = max(crossings, key=lambda x: gaussian1(x))
        intersection_y = gaussian1(intersection_x)  # or gaussian2(intersection_x), as they are equal at the intersection
    else:
        # If no crossings are found, use the average of the means as the x value
        intersection_x = (mu1 + mu2) / 2 - 2 * max(sigma1, sigma2)
        max_gaussian1 = max(gaussian1(x) for x in x_vals)
        max_gaussian2 = max(gaussian2(x) for x in x_vals)
        intersection_y = (max_gaussian1 + max_gaussian2) / 2

    return intersection_x, intersection_y


def get_score(single_gaussian_parameters, double_gaussian_parameters, single_r_squared, double_r_squared):
    """
    This function takes as input the parameters of the single and double Gaussian 
    and returns a score that estimates how well a good threshold can be found for this attribute.
    """
    amp, mu, sigma = single_gaussian_parameters
    amp1, mu1, sigma1, amp2, mu2, sigma2 = double_gaussian_parameters

    intersection_x, intersection_y = double_gaussian_intersection(double_gaussian_parameters) 

    def calculate_gaussian_overlap(amp1, mu1, sigma1, amp2, mu2, sigma2):
        """
        Calculate the overlap percentage between two Gaussians.
        This is the sum of the percentage of the area of Gaussian 1 that is also in Gaussian 2,
        and the percentage of the area of Gaussian 2 that is also in Gaussian 1.
        """
        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        # Define the range for integration
        lower_bound = min(mu1 - 4 * sigma1, mu2 - 4 * sigma2)
        upper_bound = max(mu1 + 4 * sigma1, mu2 + 4 * sigma2)

        # Integrate the product of the two Gaussians
        x_vals = np.linspace(lower_bound, upper_bound, 1000)
        g1 = gaussian(x_vals, amp1, mu1, sigma1)
        g2 = gaussian(x_vals, amp2, mu2, sigma2)
        overlap_area = np.trapz(np.minimum(g1, g2), x_vals)

        # Integrate the total area of each Gaussian
        total_area_g1 = np.trapz(g1, x_vals)
        total_area_g2 = np.trapz(g2, x_vals)

        # Calculate the overlap percentages
        overlap_percentage_g1 = overlap_area / total_area_g1
        overlap_percentage_g2 = overlap_area / total_area_g2

        # Return the sum of the overlap percentages
        return (overlap_percentage_g1 + overlap_percentage_g2) * 100

    # We calculate a score based on some heuristics
    mu_difference = abs(mu2 - mu1)  # Big difference between the means -> distinguishable
    relative_intersection_height = intersection_y / max(amp1, amp2)  # Low intersection height -> distinguishable
    double_over_single_advantage = double_r_squared/single_r_squared - 1 # High value -> low single_r_squared -> distinguishable
    intersection_to_mean_of_peaks = abs(intersection_x - (mu1 + mu2)/2) # Low distance to mean of peaks -> distinguishable
    # overlap = calculate_gaussian_overlap(amp1, mu1, sigma1, amp2, mu2, sigma2) # High overlap -> indistinguishable

    score = 750 * mu_difference - 20 * relative_intersection_height + 500 * double_over_single_advantage - 150 * intersection_to_mean_of_peaks
    # print("Scaled first term of the score, mu difference: ", 750 * mu_difference)
    # print("Scaled second term of the score, relative intersection height: ", -20 * relative_intersection_height)
    # print("Scaled third term of the score, double over single advantage: ", 500 * double_over_single_advantage)
    # print("Scaled fourth term of the score, intersection to mean of peaks: ", -150 * intersection_to_mean_of_peaks)

    return score

def gaussian_cdf(x, mu, sigma):
    """
    Cumulative area from -inf to x for a Gaussian distribution (not normalized).
    """
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + erf(z))

def optimal_threshold_for_balanced_accuracy(mu1, sigma1, mu2, sigma2, intersection_x):
    # Step 1: Assign a left and right Gaussian based on the means
    if mu1 < mu2:
        mu_left, sigma_left = mu1, sigma1
        mu_right, sigma_right = mu2, sigma2
    else:
        mu_left, sigma_left = mu2, sigma2
        mu_right, sigma_right = mu1, sigma1

    # Objective function to maximize: percentage of the area under the left curve from -inf to threshold
    # + percentage of the area under the right curve from threshold to +inf
    def objective(threshold):
        perc_left = gaussian_cdf(threshold, mu_left, sigma_left)
        perc_right = 1 - gaussian_cdf(threshold, mu_right, sigma_right)
        perc_sum = perc_left + perc_right
        return -perc_sum  # we willen maximum

    # Optimize around the intersection point of the 2 gaussians
    result = minimize_scalar(objective, bounds=(intersection_x - 2*max(sigma1, sigma2), intersection_x + 2*max(sigma1, sigma2)), method='bounded')

    threshold = result.x
    perc_sum = -result.fun

    return threshold, perc_sum


def get_threshold(double_gaussian_parameters, balanced_accuracy_threshold):
    """
    This function takes as input the parameters of the double Gaussian 
    and returns a threshold that can be used to classify the data points.
    """
    amp1, mu1, sigma1, amp2, mu2, sigma2 = double_gaussian_parameters
    intersection_x, intersection_y = double_gaussian_intersection(double_gaussian_parameters)
    if not balanced_accuracy_threshold:
        return intersection_x
    threshold, perc_sum = optimal_threshold_for_balanced_accuracy(mu1, sigma1, mu2, sigma2, intersection_x)
    return threshold


def get_optimal_thresholds(chosen_dataset, images_percentage, attributes_to_test):
    """
    This function creates a dictionary with the best possible thresholds for the attributes to test
    for the given dataset and percentage of images.
    It extracts the thresholds from the text file with the ground truth rankings.
    """
    ground_truth_file = os.path.join(f"optimal_thresholds_{chosen_dataset}", f"results_ranking_im_percentage_{images_percentage}.txt")

    optimal_threshold_dict = {}
    with open(ground_truth_file, "r") as f:
        for line in f:
            for attribute in attributes_to_test:
                if f"Attribute {attribute}" in line and not re.search(rf"Attribute {attribute}\d", line):
                    match = re.search(r"Threshold = (-inf|[\d.-]+)", line)
                    if match:
                        threshold_value = match.group(1)
                        if threshold_value == "-inf":
                            optimal_threshold_dict[attribute] = float('-inf')
                        else:   
                            optimal_threshold_dict[attribute] = float(threshold_value)
                    break
    return optimal_threshold_dict

def get_optimal_scores(chosen_dataset, images_percentage, attributes_to_test):
    """
    This function creates a dictionary with the best possible scores for the attributes to test
    for the given dataset and percentage of images.
    It extracts the balanced accuracies from the text file with the ground truth rankings and uses these as scores.
    """
    ground_truth_file = os.path.join(f"optimal_thresholds_{chosen_dataset}", f"results_ranking_im_percentage_{images_percentage}.txt")

    optimal_scores_dict = {}
    with open(ground_truth_file, "r") as f:
        for line in f:
            for attribute in attributes_to_test:
                if f"Attribute {attribute}" in line and not re.search(rf"Attribute {attribute}\d", line):
                    match = re.search(r"Balanced Accuracy = ([\d.-]+)", line)
                    if match:
                        balanced_accuracy = match.group(1)
                        optimal_scores_dict[attribute] = [attribute, float(balanced_accuracy)]
                    break
    return optimal_scores_dict

def get_balanced_accuracy(cosine_similarity_list_dict, threshold_dict, score_dict, cutoff):
    """
    This function calculates the balanced accuracy for the given dataset, 
    using the cosine similarities and thresholds for the best <cutoff> attributes, 
    following the ranking from the Gaussian method.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Make a list of the score_dict, which is sorted from highest to lowest score
    sorted_scores = [[attribute, score] for attribute, score in score_dict.values()]
    sorted_scores.sort(key=lambda x: x[1], reverse=True)

    if cutoff <= len(sorted_scores):
        sorted_scores = sorted_scores[:cutoff]

    for attribute, _ in sorted_scores:
        tp_per_attribute = 0
        tn_per_attribute = 0
        fp_per_attribute = 0
        fn_per_attribute = 0
        if attribute in threshold_dict:
            threshold = threshold_dict[attribute]
            for cosine_similarity, label in cosine_similarity_list_dict[attribute]:
                if cosine_similarity > threshold:
                    if label == 1:
                        tp += 1
                        tp_per_attribute += 1
                    else:
                        fp += 1
                        fp_per_attribute += 1
                else:
                    if label == 0:
                        tn += 1
                        tn_per_attribute += 1
                    else:
                        fn += 1
                        fn_per_attribute += 1
            balanced_accuracy_per_attribute = (tp_per_attribute / (tp_per_attribute + fn_per_attribute) + tn_per_attribute / (tn_per_attribute + fp_per_attribute)) / 2 \
                if (tp_per_attribute + fn_per_attribute) > 0 and (tn_per_attribute + fp_per_attribute) > 0 else 0
            # print(f"Balanced accuracy for attribute {attribute}: {balanced_accuracy_per_attribute}")
        else:
            raise ValueError(f"Attribute {attribute} not found in threshold_dict")
        
    balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0
    return balanced_accuracy, tp, tn, fp, fn
    

def compare_rankings(chosen_dataset, images_percentage, cutoff):
    """
    This function compares the rankings created by the Gaussian method with the ground truth rankings.
    It takes as input the dataset name, the percentage of images used, and the cutoff value.
    If the cutoff value is x, this means that the top x attributes are selected from the ranking.
    """
    ground_truth_file = os.path.join(f"optimal_thresholds_{chosen_dataset}", f"results_ranking_im_percentage_{images_percentage}.txt")
    gaussian_file = os.path.join("gaussian_rankings", f"{chosen_dataset}_scores_{images_percentage}percent.txt")

    # Ensure the directory for rank comparison exists
    os.makedirs(f"{chosen_dataset}_rank_comparison", exist_ok=True)

    rank_difference_gaussian_to_ground_truth = 0

    # Define the output file path
    output_file_path = os.path.join(f"{chosen_dataset}_rank_comparison", f"gaussian_to_ground_truth_{images_percentage}_im_percentage.txt")

    with open(gaussian_file, "r") as gaussian_f, open(ground_truth_file, "r") as ground_truth_f:
        ground_truth_lines = ground_truth_f.readlines()
        with open(output_file_path, "w") as output_f:
            for line in gaussian_f:
                # Check if the line starts with an integer followed by a point
                if line.strip() and re.match(r"^\d+\.", line.split()[0]):
                    gaussian_rank = int(line.split()[0][:-1])
                    if gaussian_rank > cutoff:
                        break

                    # Extract the attribute number string
                    match = re.search(r"Attribute (\d+)", line)
                    if match:
                        attribute_number_string = f"Attribute {match.group(1)}"

                    # Search for the attribute number string in the ground truth file
                    ground_truth_rank = None
                    for ground_truth_line in ground_truth_lines:
                        if attribute_number_string in ground_truth_line and \
                            not ground_truth_line[ground_truth_line.find(attribute_number_string) + len(attribute_number_string):][0].isdigit():
                            ground_truth_rank = int(ground_truth_line.split()[0][:-1])
                            break

                    # Calculate rank difference and write to the output file
                    if ground_truth_rank is not None:
                        rank_difference_gaussian_to_ground_truth += abs(gaussian_rank - ground_truth_rank)
                        output_f.write(f"{gaussian_rank} -> {ground_truth_rank}, {attribute_number_string}\n")


    rank_difference_ground_truth_to_gaussian = 0

    # Define the output file path
    output_file_path = os.path.join(f"{chosen_dataset}_rank_comparison", f"ground_truth_to_gaussian_{images_percentage}_im_percentage.txt")

    with open(ground_truth_file, "r") as ground_truth_f, open(gaussian_file, "r") as gaussian_f:
        gaussian_lines = gaussian_f.readlines()
        with open(output_file_path, "w") as output_f:
            for line in ground_truth_f:
                # Check if the line starts with an integer followed by a point
                if line.strip() and re.match(r"^\d+\.", line.split()[0]):
                    ground_truth_rank = int(line.split()[0][:-1])
                    if ground_truth_rank > cutoff:
                        break

                    # Extract the attribute number string
                    match = re.search(r"Attribute (\d+)", line)
                    if match:
                        attribute_number_string = f"Attribute {match.group(1)}"

                    # Search for the attribute number string in the gaussian file
                    gaussian_rank = None
                    for gaussian_line in gaussian_lines:
                        if attribute_number_string in gaussian_line and \
                            not gaussian_line[gaussian_line.find(attribute_number_string) + len(attribute_number_string):][0].isdigit():
                            gaussian_rank = int(gaussian_line.split()[0][:-1])
                            break

                    # Calculate rank difference and write to the output file
                    if gaussian_rank is not None:
                        rank_difference_ground_truth_to_gaussian += abs(ground_truth_rank - gaussian_rank)
                        output_f.write(f"{ground_truth_rank} -> {gaussian_rank}, {attribute_number_string}\n")

    return rank_difference_gaussian_to_ground_truth, rank_difference_ground_truth_to_gaussian



# Step 0: Choose the dataset

CUB = "CUB"
VOC = "VOC"
CELEBA = "CELEBA"

# chosen_dataset = CUB  # Change this to VOC or CELEBA if needed

# Step 1: Choose GPU device if available

device = torch_directml.device()
# device = "cuda"  # Uses DirectML for AMD GPUs
print(f"Using device: {device}")

# Step 2: Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Are we using also negative concepts?
negative_concepts = False

# Are we using the raw attributes or the processed ones?
use_processed_concepts = True

# Do you want to create a file with the ranking or not?
# If this parameter is set to True, the program will create a file with the ranking of the attributes based on the mu difference, and the histogram of the data points will not be plotted.
# If this parameter is set to False, the program will not create a file with the ranking
create_ranking = False

# Do you want to plot histograms
plot_histograms = False

# Do you want to correct the Gaussians after the curve fit
use_double_gaussian_correction = True

# Do you want to calculate the threshold in a way that maximize the theoretical balanced accuracy, 
# under the condition that the two Gaussians perfectly fit the positive and negative data
balanced_accuracy_threshold = True

# Do you want to find the optimal threshold per attribute or for all attributes in attributes_to_test together 
per_attribute = True

# Step 3: Load CUB200 dataset

# Variables that determine the percentage of images and attributes to use
# Both should be a divisor of 100
images_percentage = 100

# Changing this won't do anything, just keep this to 100
attributes_percentage = 100 

datasets_with_attribute_range = [(CUB, 313), (VOC, 21), (CELEBA, 41)]

for chosen_dataset, attribute_range in datasets_with_attribute_range:

    # Provide the indices of the attributes you want to use
    attributes_to_test = list(range(1, attribute_range))

    if chosen_dataset == CELEBA:
        attributes_to_test.remove(25)  # remove no beard for CELEBA

    balanced_accuracies = []

    # attributes_to_test = list(range(1,313)) # if you want to test all attributes for CUB
    # attributes_to_test = list(range(1,21)) # if you want to test all attributes for VOC
    # attributes_to_test = list(range(1,41)) # if you want all attributes for CELEBA

    # If cutoff is x, this means that the balanced accuracy is calculated for the top x attributes
    cutoff = 20
    # cutoff = 30 # if you want a possibly good value for CUB
    # cutoff = 15 # if you want a possibly good value for VOC
    # cutoff = 20 # if you want a possibly good value for CELEBA 

    difference_string = "difference " if negative_concepts else ""

    # if negative_concepts:
    #     print("Using negative concepts", "\n")
    # else:
    #     print("Not using negative concepts", "\n")

    if not per_attribute:
        
        print("Calculating the threshold for all attributes together", "\n")
        
        cos_sim_pickle_dir = os.path.join(f"{chosen_dataset}_cos_sim_pickles")
        all_pickles_exist = True
        for attribute in attributes_to_test:
            # Define the pickle file path
            cos_sim_pickle_file_path = os.path.join(cos_sim_pickle_dir, f"attribute_{attribute}_im_percentage_{images_percentage}.pkl")

            # Check if the pickle file exists
            if not os.path.exists(cos_sim_pickle_file_path):
                pickles_exist = False
                break
        
        if all_pickles_exist:
            full_cosine_similarity_list = []
            for attribute in attributes_to_test:
                # Load the cosine similarity list from the pickle file
                cos_sim_pickle_file_path = os.path.join(cos_sim_pickle_dir, f"attribute_{attribute}_im_percentage_{images_percentage}.pkl")
                with open(cos_sim_pickle_file_path, "rb") as f:
                    cosine_similarity_list = pickle.load(f)
                    full_cosine_similarity_list += cosine_similarity_list
            full_cosine_similarity_list.sort(key=lambda x: x[0])

        else:
            attribute_list = [attribute - 1 for attribute in attributes_to_test] 

            all_image_attributes_path = os.path.join("gaussian_pickles", f"{chosen_dataset}_all_image_attributes_{images_percentage}percent.pkl")
            image_encoding_labels_list_path = os.path.join("gaussian_pickles", f"{chosen_dataset}_image_encoding_labels_{images_percentage}percent.pkl")
            
            if os.path.exists(all_image_attributes_path):
                with open(all_image_attributes_path, "rb") as f:
                    all_image_attributes = pickle.load(f)
            else:
                all_image_attributes = None

            if os.path.exists(image_encoding_labels_list_path):
                with open(image_encoding_labels_list_path, "rb") as f:
                    image_encoding_labels_list = pickle.load(f)
                    for idx, image_encoding in enumerate(image_encoding_labels_list):
                        image_encoding_labels_list[idx] = image_encoding_labels_list[idx].to(device)
            else:
                image_encoding_labels_list = None

            if chosen_dataset == CUB:
            
                dataset = CUB200Dataset(root="./data/CUB_200_2011", device=device, model=model, preprocess=preprocess, 
                                            negative_concepts=negative_concepts, images_percentage=images_percentage, 
                                            attributes_percentage=attributes_percentage, attribute_list=attribute_list, 
                                            use_processed_concepts=use_processed_concepts, all_image_attributes=all_image_attributes,
                                            image_encoding_labels_list=image_encoding_labels_list)
            elif chosen_dataset == VOC:

                dataset = VOC2012Dataset(root="./data/VOC2012", device=device, model=model, preprocess=preprocess, 
                                            negative_concepts=negative_concepts, images_percentage=images_percentage, 
                                            attributes_percentage=attributes_percentage, attribute_list=attribute_list, 
                                            use_processed_concepts=use_processed_concepts, all_image_attributes=all_image_attributes,
                                            image_encoding_labels_list=image_encoding_labels_list)
                
            elif chosen_dataset == CELEBA:

                dataset = celebADataset(root="./data/celebA", device=device, model=model, preprocess=preprocess, 
                                            negative_concepts=negative_concepts, images_percentage=images_percentage, 
                                            attributes_percentage=attributes_percentage, attribute_list=attribute_list, 
                                            use_processed_concepts=use_processed_concepts, all_image_attributes=all_image_attributes,
                                            image_encoding_labels_list=image_encoding_labels_list)

            if all_image_attributes is None:
                all_image_attributes = dataset.all_image_attributes
                # Save all_image_attributes as a pickle file in the subdirectory
                with open(os.path.join("gaussian_pickles", f"{chosen_dataset}_all_image_attributes_{images_percentage}percent.pkl"), "wb") as f:
                    pickle.dump(all_image_attributes, f)

            start_preprocessing_time = time.time()

            preprocessed_set = dataset.preprocess_dataset()

            if image_encoding_labels_list is None:
                image_encoding_labels_list = dataset.image_encoding_labels_list
                for idx, image_encoding in enumerate(image_encoding_labels_list):
                    image_encoding_labels_list[idx] = image_encoding_labels_list[idx].to('cpu')
                # Save image_encoding_labels_list as a pickle file in the subdirectory
                with open(os.path.join("gaussian_pickles", f"{chosen_dataset}_image_encoding_labels_{images_percentage}percent.pkl"), "wb") as f:
                    pickle.dump(image_encoding_labels_list, f)

            preprocessed_list = list(preprocessed_set)

            full_cosine_similarity_list = [(inputs[0].item(), label) for inputs, label in preprocessed_list]

            full_cosine_similarity_list.sort(key=lambda x: x[0])
        
        single_gaussian_parameters, single_r_squared = single_gaussian_approximation(full_cosine_similarity_list)

        double_gaussian_parameters, double_r_squared = double_gaussian_approximation(full_cosine_similarity_list, False, plot_histograms)

        if use_double_gaussian_correction:

            double_gaussian_parameters, double_r_squared = double_gaussian_correction(full_cosine_similarity_list, double_gaussian_parameters, double_r_squared, False, plot_histograms, per_attribute)

        score = get_score(single_gaussian_parameters, double_gaussian_parameters, single_r_squared, double_r_squared)

        print(f"Score for the group of attributes: ", score, "\n")

        threshold = get_threshold(double_gaussian_parameters, balanced_accuracy_threshold)

        print(f"Threshold for the group of attributes: ", threshold, "\n")

        balanced_accuracy_group, tp, tn, fp, fn = get_balanced_accuracy({0: full_cosine_similarity_list}, {0: threshold}, {0: [0, score]}, float("inf"))
        print(f"Balanced accuracy for the group of concepts: ", balanced_accuracy_group, "\n")
        print(f"True positives for the group of concepts, using Gaussian threshold selection: ", tp, "\n")
        print(f"True negatives for the group of concepts, using Gaussian threshold selection: ", tn, "\n")
        print(f"False positives for the group of concepts, using Gaussian threshold selection: ", fp, "\n")
        print(f"False negatives for the group of concepts, using Gaussian threshold selection: ", fn, "\n")


    else:
        

        cosine_similarity_list_dict = {}
        score_dict = {}
        threshold_dict = {}

        for attribute in attributes_to_test:

            start_time = time.time()

            # print(f"Processing attribute {attribute}", "\n")

            # Define the pickle file path
            cos_sim_pickle_dir = os.path.join(f"{chosen_dataset}_cos_sim_pickles")
            cos_sim_pickle_file_path = os.path.join(cos_sim_pickle_dir, f"attribute_{attribute}_im_percentage_{images_percentage}.pkl")

            # Check if the pickle file exists
            if not os.path.exists(cos_sim_pickle_file_path):
                # Ensure the directory exists
                os.makedirs(cos_sim_pickle_dir, exist_ok=True)

                attribute_list = [attribute] 

                attribute_list = [attribute - 1 for attribute in attribute_list]

                if attribute == attributes_to_test[0]:
                    all_image_attributes_path = os.path.join("gaussian_pickles", f"{chosen_dataset}_all_image_attributes_{images_percentage}percent.pkl")
                    image_encoding_labels_list_path = os.path.join("gaussian_pickles", f"{chosen_dataset}_image_encoding_labels_{images_percentage}percent.pkl")
                    
                    if os.path.exists(all_image_attributes_path):
                        with open(all_image_attributes_path, "rb") as f:
                            all_image_attributes = pickle.load(f)
                    else:
                        all_image_attributes = None

                    if os.path.exists(image_encoding_labels_list_path):
                        with open(image_encoding_labels_list_path, "rb") as f:
                            image_encoding_labels_list = pickle.load(f)
                            for idx, image_encoding in enumerate(image_encoding_labels_list):
                                image_encoding_labels_list[idx] = image_encoding_labels_list[idx].to(device)
                    else:
                        image_encoding_labels_list = None

                if chosen_dataset == CUB:
                
                    dataset = CUB200Dataset(root="./data/CUB_200_2011", device=device, model=model, preprocess=preprocess, 
                                                negative_concepts=negative_concepts, images_percentage=images_percentage, 
                                                attributes_percentage=attributes_percentage, attribute_list=attribute_list, 
                                                use_processed_concepts=use_processed_concepts, all_image_attributes=all_image_attributes,
                                                image_encoding_labels_list=image_encoding_labels_list)
                elif chosen_dataset == VOC:

                    dataset = VOC2012Dataset(root="./data/VOC2012", device=device, model=model, preprocess=preprocess, 
                                                negative_concepts=negative_concepts, images_percentage=images_percentage, 
                                                attributes_percentage=attributes_percentage, attribute_list=attribute_list, 
                                                use_processed_concepts=use_processed_concepts, all_image_attributes=all_image_attributes,
                                                image_encoding_labels_list=image_encoding_labels_list)
                    
                elif chosen_dataset == CELEBA:

                    dataset = celebADataset(root="./data/celebA", device=device, model=model, preprocess=preprocess, 
                                                negative_concepts=negative_concepts, images_percentage=images_percentage, 
                                                attributes_percentage=attributes_percentage, attribute_list=attribute_list, 
                                                use_processed_concepts=use_processed_concepts, all_image_attributes=all_image_attributes,
                                                image_encoding_labels_list=image_encoding_labels_list)
                
                # Ensure the subdirectory 'gaussian_pickles' exists
                os.makedirs("gaussian_pickles", exist_ok=True)

                if all_image_attributes is None:
                    all_image_attributes = dataset.all_image_attributes
                    # Save all_image_attributes as a pickle file in the subdirectory
                    with open(os.path.join("gaussian_pickles", f"{chosen_dataset}_all_image_attributes_{images_percentage}percent.pkl"), "wb") as f:
                        pickle.dump(all_image_attributes, f)

                start_preprocessing_time = time.time()

                preprocessed_set = dataset.preprocess_dataset()

                if image_encoding_labels_list is None:
                    image_encoding_labels_list = dataset.image_encoding_labels_list
                    for idx, image_encoding in enumerate(image_encoding_labels_list):
                        image_encoding_labels_list[idx] = image_encoding_labels_list[idx].to('cpu')
                    # Save image_encoding_labels_list as a pickle file in the subdirectory
                    with open(os.path.join("gaussian_pickles", f"{chosen_dataset}_image_encoding_labels_{images_percentage}percent.pkl"), "wb") as f:
                        pickle.dump(image_encoding_labels_list, f)

                preprocessed_list = list(preprocessed_set)

                cosine_similarity_list = [(inputs[0].item(), label) for inputs, label in preprocessed_list]

                cosine_similarity_list.sort(key=lambda x: x[0])

                # Save the cosine similarity list as a pickle file
                with open(cos_sim_pickle_file_path, "wb") as f:
                    pickle.dump(cosine_similarity_list, f)
            else:
                # Load the cosine similarity list from the pickle file
                with open(cos_sim_pickle_file_path, "rb") as f:
                    cosine_similarity_list = pickle.load(f)

            cosine_similarity_list_dict[attribute] = list(cosine_similarity_list)

            single_gaussian_parameters, single_r_squared = single_gaussian_approximation(cosine_similarity_list)

            double_gaussian_parameters, double_r_squared = double_gaussian_approximation(cosine_similarity_list, create_ranking, plot_histograms)

            if use_double_gaussian_correction:

                double_gaussian_parameters, double_r_squared = double_gaussian_correction(cosine_similarity_list, double_gaussian_parameters, double_r_squared, create_ranking, plot_histograms)

            score = get_score(single_gaussian_parameters, double_gaussian_parameters, single_r_squared, double_r_squared)
            score_dict[attribute] = [attribute, score]

            # print(f"Score for attribute {attribute}: ", score, "\n")

            threshold = get_threshold(double_gaussian_parameters, balanced_accuracy_threshold)
            threshold_dict[attribute] = threshold

            # print(f"Threshold for attribute {attribute}: ", threshold, "\n")
            
            end_time = time.time()


        optimal_threshold_dict = get_optimal_thresholds(chosen_dataset, images_percentage, attributes_to_test)
        optimal_score_dict = get_optimal_scores(chosen_dataset, images_percentage, attributes_to_test)

        # print("Optimal thresholds for the attributes: ", optimal_threshold_dict, "\n")

        for cutoff_iter in range(1, len(attributes_to_test) + 1):

            cutoff = cutoff_iter

            print(f"Chosen dataset: {chosen_dataset}, cutoff: {cutoff}", "\n")

            balanced_accuracy_full = get_balanced_accuracy(cosine_similarity_list_dict, threshold_dict, score_dict, float("inf"))[0]
            # print("Balanced accuracy for all concepts, using Gaussian threshold selection: ", balanced_accuracy_full, "\n")

            optimal_balanced_accuracy_full = get_balanced_accuracy(cosine_similarity_list_dict, optimal_threshold_dict, score_dict, float("inf"))[0]
            # print("Balanced accuracy for all concepts, using optimal threshold selection (upper bound): ", optimal_balanced_accuracy_full, "\n")

            balanced_accuracy_cutoff = get_balanced_accuracy(cosine_similarity_list_dict, threshold_dict, score_dict, cutoff)[0]
            # print(f"Balanced accuracy for top {cutoff} concepts, selected by Gaussian concept selection, using Gaussian threshold selection: ", balanced_accuracy_cutoff, "\n")

            optimal_balanced_accuracy_cutoff = get_balanced_accuracy(cosine_similarity_list_dict, optimal_threshold_dict, score_dict, cutoff)[0]
            # print(f"Balanced accuracy for top {cutoff} concepts, selected by Gaussian concept selection, using optimal threshold selection (upper bound): ", optimal_balanced_accuracy_cutoff, "\n")

            double_optimal_balanced_accuracy_cutoff = get_balanced_accuracy(cosine_similarity_list_dict, optimal_threshold_dict, optimal_score_dict, cutoff)[0]
            # print(f"Balanced accuracy for top {cutoff} concepts, selected by optimal concept selection, using optimal threshold selection (even better upper bound): ", double_optimal_balanced_accuracy_cutoff, "\n")

            tp = get_balanced_accuracy(cosine_similarity_list_dict, threshold_dict, score_dict, cutoff)[1]
            tn = get_balanced_accuracy(cosine_similarity_list_dict, threshold_dict, score_dict, cutoff)[2]
            fp = get_balanced_accuracy(cosine_similarity_list_dict, threshold_dict, score_dict, cutoff)[3]
            fn = get_balanced_accuracy(cosine_similarity_list_dict, threshold_dict, score_dict, cutoff)[4]
            # print(f"True positives for top {cutoff} concepts, selected by Gaussian concept selection, using Gaussian threshold selection: ", tp, "\n")
            # print(f"True negatives for top {cutoff} concepts, selected by Gaussian concept selection, using Gaussian threshold selection: ", tn, "\n")
            # print(f"False positives for top {cutoff} concepts, selected by Gaussian concept selection, using Gaussian threshold selection: ", fp, "\n")
            # print(f"False negatives for top {cutoff} concepts, selected by Gaussian concept selection, using Gaussian threshold selection: ", fn, "\n")

            balanced_accuracies.append(balanced_accuracy_cutoff)

        if create_ranking:

            # Ensure the subdirectory 'gaussian_rankings' exists
            os.makedirs("gaussian_rankings", exist_ok=True)

            # Define the output file path
            output_file_path = os.path.join("gaussian_rankings", f"{chosen_dataset}_scores_{images_percentage}percent.txt")

            # Make a list of the score_dict, which is sorted from highest to lowest score
            sorted_scores = [[attribute, score] for attribute, score in score_dict.values()]
            sorted_scores.sort(key=lambda x: x[1], reverse=True)

            # Write the attribute and score to the file
            ranking_position = 1
            with open(output_file_path, "w") as f:
                for attribute, score in sorted_scores:
                    f.write(f"{ranking_position}. Attribute {attribute}, Score: {score}\n\n")
                    ranking_position += 1 

            rank_difference_gaussian_to_ground_truth_full, rank_difference_ground_truth_to_gaussian_full = \
                compare_rankings(chosen_dataset, images_percentage, float("inf"))
            
            rank_difference_gaussian_to_ground_truth_cutoff, rank_difference_ground_truth_to_gaussian_cutoff = \
                compare_rankings(chosen_dataset, images_percentage, cutoff)
            
            # print("Rank difference Gaussian to ground truth for all concepts (lower score = better): ", rank_difference_gaussian_to_ground_truth_full, "\n")
            # print("Rank difference ground truth to Gaussian for all concepts (lower score = better): ", rank_difference_ground_truth_to_gaussian_full, "\n")
            # print(f"Rank difference Gaussian to ground truth for top {cutoff} concepts (lower score = better): ", rank_difference_gaussian_to_ground_truth_cutoff, "\n")
            # print(f"Rank difference ground truth to Gaussian for top {cutoff} concepts (lower score = better): ", rank_difference_ground_truth_to_gaussian_cutoff, "\n")

    print(balanced_accuracies, "\n")
    # Save the raw list of balanced accuracies to a file
    with open(f"balanced_accuracies_{chosen_dataset}_{images_percentage}percent.txt", "w") as f:
        f.write(str(balanced_accuracies))       
