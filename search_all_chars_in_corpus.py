import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import faiss
from concurrent.futures import ThreadPoolExecutor
import random
import cv2

import warnings
warnings.filterwarnings("ignore") 

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call this function at the beginning of your script
set_deterministic()

# Read mapping data
df = pd.read_excel('final_characteristics-v2.xlsx')
df_1 = df[['UNICODE', 'CHAR']].copy()
char_dict = dict(zip(df_1['UNICODE'], df_1['CHAR']))

# Load pretrained ResNet18 and modify for grayscale images
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify to accept 1-channel (grayscale)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification head
model.eval()

# Preprocessing transforms with correct resizing and normalization
preprocess = transforms.Compose([
    transforms.Resize((100, 100), interpolation=cv2.INTER_LINEAR),  # Resize with linear interpolation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization as requested
])

def extract_features(img_path):
    img = Image.open(img_path).convert('L')  # Open as grayscale
    img_tensor = preprocess(img).unsqueeze(0)  # Preprocess and add batch dimension
    with torch.no_grad():
        features = model(img_tensor).squeeze()  # Extract features
    features = features / torch.norm(features)  # Normalize the feature vector
    return features.numpy()

def process_image(unicode_value, image_folder):
    img_path = os.path.join(image_folder, f"{unicode_value}.jpg")
    if os.path.exists(img_path):
        return unicode_value, extract_features(img_path)
    else:
        print(f"Not found {img_path}!")
        return unicode_value, None

def find_similar_images_faiss(image_folder, top_k=20):
    output_data = []
    feature_dict = {}

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda unicode_value: process_image(unicode_value, image_folder), df_1['UNICODE']))

    # Build feature dictionary
    for unicode_value, feature_vector in results:
        if feature_vector is not None:
            feature_dict[unicode_value] = feature_vector

    if not feature_dict:
        print("No features extracted!")
        return None

    # Prepare feature matrix
    feature_matrix = np.array(list(feature_dict.values()))
    unicode_list = list(feature_dict.keys())

    # Normalize all feature vectors (already normalized above, but this ensures consistency)
    feature_matrix = feature_matrix / np.linalg.norm(feature_matrix, axis=1, keepdims=True)

    # Use Faiss IndexFlatIP (Inner Product)
    index = faiss.IndexFlatIP(feature_matrix.shape[1])  # Inner product as a proxy for cosine similarity
    index.add(feature_matrix)  # Add all features to the Faiss index

    for i, input_unicode in enumerate(unicode_list):
        input_char = char_dict[input_unicode]
        input_feature = feature_matrix[i].reshape(1, -1)  # Vector for the current character

        # Search for the top K similar characters using Faiss (inner product is similar to cosine)
        distances, indices = index.search(input_feature, top_k + 1)  # Search top K + 1 to exclude itself

        similar_images = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != i:  # Exclude the character itself
                similar_images.append((char_dict[unicode_list[idx]], dist))

        # Take top 20 similar characters
        top_10_similar = [char for char, _ in sorted(similar_images, key=lambda x: x[1], reverse=True)[:top_k]]

        output_data.append({
            'Input Character': input_char,
            'Top 20 Similar Characters': top_10_similar
        })

    # Export results to a file
    output_df = pd.DataFrame(output_data)
    output_df.to_excel('./output/output_top_k_similar_char.xlsx', index=False)
    output_df.to_csv('./output/output_top_k_similar_char.csv', index=False)

    return output_df

def load_config(config_path):
    config = {}
    with open(config_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            config[key] = float(value)
    return config

def main():
    image_folder = './images'
    result_df = find_similar_images_faiss(image_folder)
    print(result_df)

if __name__ == "__main__":
    main()