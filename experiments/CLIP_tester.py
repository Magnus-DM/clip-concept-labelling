import torch
import clip
from PIL import Image
import numpy as np
import torch_directml

def cosine_similarity(v1, v2):
    # Flatten tensors to 1D
    return torch.nn.functional.cosine_similarity(v1.squeeze(), v2.squeeze(), dim=0).item()

def dot_product(v1, v2):
    # Flatten tensors to 1D
    return torch.dot(v1.squeeze(), v2.squeeze()).item()

def angle_between(v1, v2):
    cos_sim = cosine_similarity(v1, v2)
    return np.arccos(np.clip(cos_sim, -1.0, 1.0)) * (180 / np.pi)

def norm_difference(v1, v2):
    return torch.norm(v1.squeeze() - v2.squeeze()).item()

# Load CLIP model
device = torch_directml.device()
model, preprocess = clip.load("ViT-B/32", device=device)

# Preprocess image
image = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)

# Define search texts
string_list = ["the dog is light brown", "the dog is not light brown", "the dog is black", "the dog is not black"]

cosine_similarity_dict = {}
for string in string_list:
    cosine_similarity_dict[string] = None

dot_product_dict = {}
for string in string_list:
    dot_product_dict[string] = None

angle_dict = {}
for string in string_list:
    angle_dict[string] = None

norm_difference_dict = {}
for string in string_list:
    norm_difference_dict[string] = None

# Tokenize the texts
texts = clip.tokenize(string_list).to(device)

with torch.no_grad():
    # Calculate embedding vectors
    image_features = model.encode_image(image)
    text_features = model.encode_text(texts)

    # Calculate metrics
    for key in cosine_similarity_dict.keys():
        cosine_similarity_dict[key] = cosine_similarity(image_features, text_features[string_list.index(key)])
        dot_product_dict[key] = dot_product(image_features, text_features[string_list.index(key)])
        angle_dict[key] = angle_between(image_features, text_features[string_list.index(key)])
        norm_difference_dict[key] = norm_difference(image_features, text_features[string_list.index(key)])

print("Cosine similarity:")
for key, value in cosine_similarity_dict.items():
    print(f"{key}: {value}")

print("\nDot product:")
for key, value in dot_product_dict.items():
    print(f"{key}: {value}")

print("\nAngle:")
for key, value in angle_dict.items():
    print(f"{key}: {value}")

print("\nNorm difference:")
for key, value in norm_difference_dict.items():
    print(f"{key}: {value}")
