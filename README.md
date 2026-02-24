# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.

## Problem Statement and Dataset

Image classification from scratch requires a huge dataset and long training times. To overcome this, transfer learning can be applied using pre-trained models like VGG-19, which has already learned feature representations from a large dataset (ImageNet).

Problem Statement: Build an image classifier using VGG-19 pre-trained architecture, fine-tuned for a custom dataset (e.g., CIFAR-10, Flowers dataset, or any small image dataset).

Dataset: A dataset consisting of multiple image classes (e.g., train, test, and validation sets). For example, CIFAR-10 (10 classes of small images) or a custom dataset with multiple classes.

</br>
</br>
</br>

## DESIGN STEPS

### STEP 1:
Import the required libraries (PyTorch, torchvision, matplotlib, etc.) and set up the device (CPU/GPU).

### STEP 2:
Load the dataset (train and test). Apply transformations such as resizing, normalization, and augmentation. Create DataLoader objects.

### STEP 3:
Load the pre-trained VGG-19 model from torchvision.models. Modify the final fully connected layer to match the number of classes in the dataset.

### STEP 4:
Define the loss function (CrossEntropyLoss) and the optimizer (Adam).

### STEP 5:
Train the model for the required number of epochs while recording training loss and validation loss.

### STEP 6:
Evaluate the model using a confusion matrix, classification report, and test it on new samples.

### PROGRAM
```
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset (example: CIFAR-10 or custom dataset)
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(pretrained=True)

# Freeze all layers except classifier
for param in model.features.parameters():
    param.requires_grad = False

num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

# Train the model
def train_model(model, train_loader, test_loader, num_epochs=5):
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss/len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss/len(test_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

train_losses, val_losses = train_model(model, train_loader, test_loader, num_epochs=10)
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
```
Epoch [1/5], Train Loss: 0.6178, Validation Loss: 0.3984
Epoch [2/5], Train Loss: 0.3926, Validation Loss: 0.3350
Epoch [3/5], Train Loss: 0.2882, Validation Loss: 0.2718
Epoch [4/5], Train Loss: 0.2648, Validation Loss: 0.2425
Epoch [5/5], Train Loss: 0.2112, Validation Loss: 0.2309
Name: MEENAKSHI R       
Register Number: 212224220062       
```
<img width="700" height="547" alt="download" src="https://github.com/user-attachments/assets/673d33a5-3a77-4bec-bc31-8ee5f3ddb2d2" />


</br>
</br>
</br>

### Confusion Matrix
```
Test Accuracy: 0.9091
Name: MEENAKSHI R    
Register Number: 212224220062
```
<img width="640" height="547" alt="download" src="https://github.com/user-attachments/assets/5c5e8365-66bf-4b35-9e18-6786d792e3d2" />

</br>
</br>
</br>

### Classification Report
```
Name: MEENAKSHI R
Register Number: 212224220062
Classification Report:
              precision    recall  f1-score   support

      defect       0.84      0.82      0.83        33
   notdefect       0.93      0.94      0.94        88

    accuracy                           0.91       121
   macro avg       0.89      0.88      0.88       121
weighted avg       0.91      0.91      0.91       121
```
</br>
</br>
</br>

### New Sample Prediction

<img width="328" height="371" alt="download" src="https://github.com/user-attachments/assets/0580d021-bf03-4430-98d5-dfc091e002ab" />

<img width="328" height="371" alt="download" src="https://github.com/user-attachments/assets/444c616b-dff4-48c4-ac26-575748f7a50b" />

</br>
</br>
</br>

## RESULT

Thus, the transfer Learning for classification using VGG-19 architecture has succesfully implemented.
</br>
</br>
</br>
