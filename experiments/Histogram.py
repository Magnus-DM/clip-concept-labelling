import torch
from torch.utils.data import Dataset
import clip
import os
import torch_directml
from sklearn.model_selection import train_test_split
from PIL import Image
import time
import matplotlib.pyplot as plt
import re
from sklearn.metrics import f1_score
import os
import threading
import csv

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
    
def find_optimal_threshold(sorted_list_of_tuples, start_index=0, end_index=None):

    # print("Starting to iterate through the sorted list of tuples with length ", len(sorted_list_of_tuples))

    best_i = 0
    best_threshold = float('-inf')
    best_balanced_accuracy = 0
    
    if end_index is None:
        end_index = len(sorted_list_of_tuples) + 1

    for i in range(start_index, end_index):
        # if i % 100 == 0:
        #     print("Currently at index ", i)
        if i == 0:
            threshold = float('-inf')
            label_switch = True
        elif i == len(sorted_list_of_tuples):
            threshold = float('inf')
            label_switch = True
        else:
            label_switch = (sorted_list_of_tuples[i-1][0] != sorted_list_of_tuples[i][0])
            if label_switch:
                threshold = (sorted_list_of_tuples[i-1][0] + sorted_list_of_tuples[i][0]) / 2
        if label_switch:

            tn = 0; tp = 0
            fn = 0; fp = 0

            for j in range(len(sorted_list_of_tuples)):
                if sorted_list_of_tuples[j][0] < threshold: # below threshold
                    if sorted_list_of_tuples[j][1] == 0:
                        tn += 1
                    else:
                        fn += 1
                
                else: # above threshold
                    if sorted_list_of_tuples[j][1] == 0:
                        fp += 1
                    else:
                        tp += 1
            
            # > Threshold: predict 1, < Threshold: predict 0        
            balanced_accuracy = 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if (tp + fn) > 0 and (tn + fp) > 0 else 0

            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy
                best_i = i
                best_threshold = threshold

    return best_i, best_threshold, best_balanced_accuracy

def plot_histogram(chosen_dataset, attribute, data, title, xlabel, ylabel, frequency=False, 
                   use_processed_concepts=True, negative_concepts=False, 
                   images_percentage=100, attributes_percentage=100):

    # Create the full directory path for saving histograms
    param_string = "processed" if use_processed_concepts else "raw"
    param_string += "_negative" if negative_concepts else ""
    param_string += f"_{images_percentage}"
    if attribute is None:
        param_string += f"_{attributes_percentage}_concept_percentage"
    else:
        param_string += f"_concept_{attribute}"

    histogram_dir = os.path.join(f"histograms_{chosen_dataset}", param_string)
    if not os.path.exists(histogram_dir):
        os.makedirs(histogram_dir)

    data_0 = [x[0] for x in data if x[1] == 0]
    data_1 = [x[0] for x in data if x[1] == 1]
    
    if frequency:
        plt.hist(data_0, bins=50, alpha=0.5, label='Label 0', color='blue', density=True)
        plt.hist(data_1, bins=50, alpha=0.5, label='Label 1', color='red', density=True)
    else:
        plt.hist(data_0, bins=50, alpha=0.5, label='Label 0', color='blue')
        plt.hist(data_1, bins=50, alpha=0.5, label='Label 1', color='red')
    
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    # Adjust layout to add whitespace and prevent text cutoff
    plt.tight_layout()

    # Save the figure
    filename = title.lower().replace(" ", "_")
    plt.savefig(os.path.join(histogram_dir, f"{filename}.png"), bbox_inches='tight')
    plt.savefig(os.path.join(histogram_dir, f"{filename}.pdf"), bbox_inches='tight')
    plt.legend(loc='upper right', fontsize=16)

    # Clear the plot to avoid overlapping
    plt.clf()

def save_results(chosen_dataset, attribute_threshold_percentage_list, negative_concepts, difference_string, dataset, images_percentage=100):
    # Create the directory for saving optimal thresholds if it doesn't exist
    optimal_thresholds_dir = "optimal_thresholds" + "_" + chosen_dataset
    if not os.path.exists(optimal_thresholds_dir):
        os.makedirs(optimal_thresholds_dir)
    # Determine the filename based on the parameters
    results_filename = f"results_{'negative_' if negative_concepts else ''}im_percentage_{images_percentage}.txt"
    results_filepath = os.path.join(optimal_thresholds_dir, results_filename)

    # Open the file for writing
    with open(results_filepath, "w") as results_file:
        # Sort the list by the best balanced accuracy
        attribute_threshold_percentage_list.sort(key=lambda x: x[2], reverse=True)
        results_file.write("--------------------------------------------------------------\n")
        results_file.write(f"Ranking of concepts best suited for thresholding based on cosine similarity {difference_string}:\n")
        results_file.write("--------------------------------------------------------------\n")
        for attribute, threshold, balanced_accuracy in attribute_threshold_percentage_list:
            # with open(dataset.used_attribute_names_file, "r") as f:
            #     attribute_name = None
            #     for line in f:
            #         if line.startswith(f"{attribute}"):
            #             attribute_name = line.strip()
            #             break
            results_file.write(f"Attribute {attribute}: Threshold = {threshold}, Balanced Accuracy = {balanced_accuracy}\n")
            results_file.write("\n")
            print(f"Attribute {attribute}: Threshold = {threshold}, Balanced Accuracy = {balanced_accuracy}")

    print("--------------------------------------------------------------", "\n")
    print(f"Ranking of concepts best suited for thresholding based on cosine similarity {difference_string}: ", "\n")
    print("--------------------------------------------------------------", "\n")
    for attribute, threshold, balanced_accuracy in attribute_threshold_percentage_list:
        # with open(dataset.used_attribute_names_file, "r") as f:
        #     attribute_name = None
        #     for line in f:
        #         if line.startswith(f"{attribute}"):
        #             attribute_name = line.strip()
        #             break
        print(f"Attribute {attribute}: Threshold = {threshold}, Balanced Accuracy = {balanced_accuracy}")




# Step 0: Choose the dataset

CUB = "CUB"
VOC = "VOC"
CELEBA = "CELEBA"

chosen_dataset = CUB # Choose the dataset you want to use

# Step 1: Choose GPU device if available

device = torch_directml.device()  # Uses DirectML for AMD GPUs
print(f"Using device: {device}")

# Step 2: Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Ammount of threads to use for the search of the optimal threshold, used to speed up the process
threads = 16

# Are we using also negative concepts?
negative_concepts_list = [False] # Keep this on False, as negative concepts dont work for VOC and CelebA

# Are we using the raw attributes or the processed ones?
use_processed_concepts = True

# If True, the code will search for the optimal threshold for each concept separately
# If False, the code will search for the optimal threshold for all concepts together (given by the attributes_percentage variable)
per_concept = True

for negative_concepts in negative_concepts_list:
    if negative_concepts:
        print("Using negative concepts", "\n")
    else:
        print("Not using negative concepts", "\n")
    
    # Step 3: Load CUB200 dataset

    # Variables that determine the percentage of images and attributes to use
    # Both should be a divisor of 100
    images_percentage = 100

    attributes_percentage = 100 # only matters if per_concept is False

    if per_concept:
        attributes_to_test = list(range(1,313)) # if you want to test all attributes for CUB
        # attributes_to_test = list(range(1,21)) # if you want to test all attributes for VOC
        # attributes_to_test = list(range(1,41)) # if you want all attributes for CELEBA
    else:
        attributes_to_test = [None] # Provide [None] if you want to use all attributes

    attribute_threshold_percentage_list = []

    difference_string = "difference " if negative_concepts else ""

    for attribute in attributes_to_test:

        start_time = time.time()

        attribute_list = [attribute] # Provide the indices of the attributes you want to use, as specified in the attributes.txt file
        if attribute_list == [None]:
            attribute_list = None
            print(f"Using {attributes_percentage}% of all attributes to find the optimal threshold for cosine similarity {difference_string}")
        else:
            print(f"Searching the optimal threshold for cosine similarity {difference_string}for attribute {attribute}")

        # Each element of the attribute_list is reduced by 1, to correspond to the correct index in the dataset
        if attribute_list is not None:
            attribute_list = [attribute - 1 for attribute in attribute_list]

        if attribute_list is None or attribute == attributes_to_test[0]:
            all_image_attributes = None
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

        start_preprocessing_time = time.time()

        preprocessed_set = dataset.preprocess_dataset()

        if image_encoding_labels_list is None:
            image_encoding_labels_list = dataset.image_encoding_labels_list

        preprocessed_list = list(preprocessed_set)

        cosine_similarity_list = [(inputs[0].item(), label) for inputs, label in preprocessed_list]

        cosine_similarity_list.sort(key=lambda x: x[0])

        # Find optimal threshold for the cosine similarity (difference)

        part_length = len(cosine_similarity_list) // threads

        results = [None] * threads

        def thread_function(thread_id, cosine_similarity_list, start_index, end_index):
            results[thread_id] = find_optimal_threshold(cosine_similarity_list, start_index, end_index)

        threads_list = []
        for i in range(threads):
            start_index = i * part_length
            end_index = (i + 1) * part_length if i != threads - 1 else len(cosine_similarity_list) + 1
            thread = threading.Thread(target=thread_function, args=(i, cosine_similarity_list, start_index, end_index))
            threads_list.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads_list:
            thread.join()

        # Find the best result across all threads
        best_result = max(results, key=lambda x: x[2])  # Compare by best_balanced_accuracy
        best_i, best_threshold, best_balanced_accuracy = best_result

        if per_concept:
            attribute_threshold_percentage_list.append((attribute, best_threshold, best_balanced_accuracy))

        # Plot histograms for cosine similarity
        plot_histogram(chosen_dataset, attribute, cosine_similarity_list, 'Histogram of Cosine Similarities (Frequency)', 'Cosine Similarity', 'Frequency', frequency=True,
                    use_processed_concepts=use_processed_concepts, negative_concepts=negative_concepts, 
                    images_percentage=images_percentage, attributes_percentage=attributes_percentage)
        plot_histogram(chosen_dataset, attribute, cosine_similarity_list, 'Histogram of Cosine Similarities', 'Cosine Similarity', 'Count', frequency=False, 
                    use_processed_concepts=use_processed_concepts, negative_concepts=negative_concepts,
                    images_percentage=images_percentage, attributes_percentage=attributes_percentage)
        
        end_time = time.time()

        print(f"Time taken to find the optimal threshold for cosine similarity {difference_string}for attribute {attribute}: ", end_time - start_time, "seconds", "\n")
        if attribute == attributes_to_test[0]:
            print("Because this is the first attribute, this time is longer than the others, as the dataset is being preprocessed for the first time")
        print("Optimal threshold for this attribute: ", best_threshold, "\n")
        print("Balanced accuracy for this threshold: ", best_balanced_accuracy, "\n")
        print("--------------------------------------------------------------", "\n")


    if per_concept:
        save_results(chosen_dataset, attribute_threshold_percentage_list, negative_concepts, difference_string, dataset, images_percentage)