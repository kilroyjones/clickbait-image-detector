from torchvision import datasets, transforms
from torch import nn, optim
import database as db

# Libraries
import os
import sys
import sys
import torch
import torchvision.models as models
from typing import Final
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

# Modules
import database as db
import sqlite3
from dataset import CustomDataset

class Video:
    def __init__(self, video_id, cls):
        self.video_id = video_id
        self.cls = cls
    
    def __repr__(self):
        return f"Video(video_id={self.video_id}, cls={self.cls})"

# - MobileNetv2 requires inputs of 224x224. 
# - We randomly flip to increase variability of the data.
# - Convert it to a tensor and normalize.
# - The normalization values are needed to match ImageNet, on top of which this was train [check validity of this] 
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


###

###
def getPaths(conn: sqlite3.Connection): 
    videos = db.get_all_videos(conn) 
    video_objects = [Video(**video_data) for video_data in videos]

    image_paths = []
    image_labels = []

    for video in video_objects:
        image_paths.append(f"./downloads/{video.video_id}.jpg")
        image_labels.append(0 if video.cls == '1' else 1)

    return train_test_split(image_paths, image_labels, test_size=0.2, random_state=42)

def createDatasets(train_paths, validate_paths, train_labels, validate_labels): 
    train_dataset = CustomDataset(train_paths, train_labels, transform=transform)
    validate_dataset = CustomDataset(validate_paths, validate_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=False)
    return train_loader, validate_loader


if __name__ == "__main__":
    conn = db.create_connection('output.sqlite')

    if(conn == None):
        print("Unable to connection to database")
        sys.exit(1)  
    
    train_paths, validate_paths, train_labels, validate_labels = getPaths(conn)
    train_loader, validate_loader = createDatasets(train_paths, validate_paths, train_labels, validate_labels)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
 
    # TRAINING
    #############
    acc = []
    num_epochs = 10 
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
        training_accuracy = correct_predictions / total_predictions
        acc.append(training_accuracy)

        # Fix this to handle overtraining
        if len(acc) > 3: 
            if sum(acc[:-3]):
                diff1 = abs(acc[-1] - acc[-2])
                diff2 = abs(acc[-2] - acc[-3])
                avg_diff = (diff1 + diff2) / 2
                print(diff1, diff2, avg_diff)
                if(avg_diff < 0.01): 
                    break
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Training Accuracy: {training_accuracy}")
    
    # TRAINING
    #############
    model.eval()  # Set the model to evaluation mode
    validation_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Inference mode, gradients not needed
        for inputs, labels in validate_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    validation_accuracy = correct_predictions / total_predictions
    print(f"Validation Loss: {validation_loss/len(validate_loader)}, Validation Accuracy: {validation_accuracy}")


    
    # Add validation loop here if needed
