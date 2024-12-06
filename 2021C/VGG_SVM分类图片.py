import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

# Load Pre-trained VGG Model
vgg_model = models.vgg11()
vgg_model.eval()

# Remove the final classification layer to get feature vectors
vgg_model.classifier = nn.Sequential(*[vgg_model.classifier[i] for i in range(5)])

device = torch.device("mps")
vgg_model = vgg_model.to(device)

# Load dataset
data = pd.read_excel('2021MCMProblemC_DataSet.xlsx')
img_data = pd.read_excel('2021MCM_ProblemC_ Images_by_GlobalID.xlsx')
positive_ids = data[data['Lab Status'] == 'Positive ID']['GlobalID']
negative_ids = data[data['Lab Status'] == 'Negative ID']['GlobalID']
positive_files = img_data[img_data['GlobalID'].isin(positive_ids)]['FileName']
negative_files = img_data[img_data['GlobalID'].isin(negative_ids)]['FileName']

original_images_dir = "2021MCM_ProblemC_Files"
organized_data_dir = "organized_data"
if not os.path.exists(organized_data_dir):
    positive_dir = os.path.join(organized_data_dir, 'positive')
    negative_dir = os.path.join(organized_data_dir, 'negative')
    os.makedirs(positive_dir)
    os.makedirs(negative_dir)
    for file_name in positive_files:
        src_path = os.path.join(original_images_dir, file_name)
        dest_path = os.path.join(positive_dir, file_name)
        shutil.copy(src_path, dest_path)

    for file_name in negative_files:
        src_path = os.path.join(original_images_dir, file_name)
        dest_path = os.path.join(negative_dir, file_name)
        shutil.copy(src_path, dest_path)

# Transform images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create DataLoader
dataset = datasets.ImageFolder(root=organized_data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# TODO: rotation, gaussian blurring, cropping and affine shifting

# Feature extraction function
def extract_features(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device)
            output = model(inputs)
            features.extend(output.cpu().numpy())
            labels.extend(label.cpu().numpy())
    return np.array(features), np.array(labels)


# Extract features and labels
features, labels = extract_features(vgg_model, dataloader)

# Train SVM on Extracted Features
svm_classifier = SVC(kernel='linear')  # You can experiment with different kernels
svm_classifier.fit(features, labels)

# Evaluation
# Predict on training set (or use a separate test set)
predictions = svm_classifier.predict(features)

accuracy = accuracy_score(labels, predictions)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")
