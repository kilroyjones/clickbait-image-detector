"""

TODO: Adjust based on this https://github.com/akrapukhin/MobileNetV3/blob/main/train.py
"""


import sys
import torch

from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split

import app.database.database as db

from app.models.dataset import CustomDataset
from app.models.video import VideoIdAndClass


def transform_image():
    """
    Define transformations for input images.

    The input for MobileNet is 224x224 and the normalization values are used to
    get our images inline with those used in the original training set.  
    """ 

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_paths(conn):
    """
    Retrieve image paths and labels from the database.
    """

    videos = db.get_all_videos(conn) 
    video_objects = [VideoIdAndClass(**video_data) for video_data in videos]

    image_paths = [f"./downloads/{video.video_id}.jpg" for video in video_objects]
    image_labels = [0 if video.cls == '1' else 1 for video in video_objects]

    return train_test_split(image_paths, image_labels, test_size=0.2, random_state=42)


def create_datasets(train_paths, validate_paths, train_labels, validate_labels, transform):
    """
    Create and return DataLoader for train and validation datasets.
    """

    train_dataset = CustomDataset(train_paths, train_labels, transform=transform)
    validate_dataset = CustomDataset(validate_paths, validate_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=False)
    
    return train_loader, validate_loader



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    Train the model and return training and validation accuracy history.

    """

    train_acc_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

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
        
        train_accuracy = correct_predictions / total_predictions
        train_acc_history.append(train_accuracy)
        
        val_loss = evaluate_on_validation_set(model, val_loader, criterion)
        val_loss_history.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Training Accuracy: {train_accuracy}, Validation Loss: {val_loss}")

    return train_acc_history, val_loss_history


def evaluate_on_validation_set(model, val_loader, criterion):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)



def validate_model(model, validate_loader, criterion):
    """
    Validate the model and print validation loss and accuracy.
    """

    model.eval()
    validation_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
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


def main():
    """
    
    """
    conn = db.create_connection('output.sqlite')
    if not conn:
        print("Unable to connect to database.")
        sys.exit(1)

    transform = transform_image()
    train_paths, validate_paths, train_labels, validate_labels = get_paths(conn)
    train_loader, validate_loader = create_datasets(train_paths, validate_paths, train_labels, validate_labels, transform)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # num_ftrs = model.classifier[-1].in_features  # Access the in_features of the last layer

    # Replace the classifier with a new one that includes dropout and a new final layer for 2 classes
    # model.classifier = nn.Sequential(
    #     nn.Dropout(0.2),  # Example dropout rate
    #     nn.Linear(num_ftrs, 2)  # Adjust to 2 classes
    # )

    # Move the model to the GPU
    model = model.to('cuda')

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer, adding L2 regularization via weight_decay
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


    # Use these returned values?
    train_acc_history, val_loss_history = train_model(model, train_loader, validate_loader, criterion, optimizer)

    validate_model(model, validate_loader, criterion)

    conn.close()

if __name__ == "__main__":
    main()
