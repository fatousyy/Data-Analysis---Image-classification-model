import os
import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

# ============================
# Step 1: Download the Dataset
# ============================

print("Downloading dataset...")
path = kagglehub.dataset_download("karakaggle/kaggle-cat-vs-dog-dataset")
print("Path to dataset files:", path)

# ==============================
# Step 2: Define Dataset Directory
# ==============================

data_dir = os.path.join(path, "kagglecatsanddogs_3367a", "PetImages")
assert os.path.exists(data_dir), f"Dataset directory not found at {data_dir}!"

# ====================================
# Step 3: Clean Corrupted Image Files
# ====================================

print("Cleaning corrupted files...")
for category in ["Cat", "Dog"]:
    folder = os.path.join(data_dir, category)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify if it's a valid image
        except (IOError, SyntaxError):
            print(f"Removing corrupted file: {file_path}")
            os.remove(file_path)
print("Cleaning completed.")

# =================================
# Step 4: Define Data Transformations
# =================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# ==========================
# Step 5: Load the Dataset
# ==========================

print("Loading dataset...")
dataset = datasets.ImageFolder(data_dir, transform=transform)
print(f"Total images: {len(dataset)}")

# =====================================
# Step 6: Split Dataset into Train/Val/Test
# =====================================

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# ============================
# Step 7: Create Data Loaders
# ============================

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================================
# Step 8: Visualize Sample Images
# ==================================

def show_images(dataset, n=6):
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    for i in range(n):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).numpy() * 0.5 + 0.5  # Denormalize
        axes[i].imshow(image)
        axes[i].set_title("Cat" if label == 0 else "Dog")
        axes[i].axis("off")
    plt.show()

print("Displaying sample images...")
show_images(train_dataset)

# ==========================
# Step 9: Define the Device
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================
# Step 10: Define the CNN Model
# ============================

class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        # Load pre-trained ResNet18 with updated weights parameter
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
    
    def forward(self, x):
        return self.model(x)

model = CatDogClassifier().to(device)
print("Model defined.")

# ==================================
# Step 11: Define Loss Function & Optimizer
# ==================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================================
# Step 12: Train the Model
# =========================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")

# =============================
# Step 13: Start Training
# =============================

num_epochs = 5  # We can increase this for better performance

print("Starting training...")
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)
print("Training completed.")

# ==========================
# Step 14: Save the Model
# ==========================

torch.save(model.state_dict(), "cat_dog_classifier.pth")
print("Model saved as 'cat_dog_classifier.pth'.")

# ==========================
# Step 15: Evaluate on Test Set
# ==========================

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

print("Evaluating on test set...")
evaluate_model(model, test_loader, criterion)

# ==========================
# Step 16: Visualize Predictions
# ==========================

def visualize_predictions(model, dataset, num_images=6):
    model.eval()
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    with torch.no_grad():
        for i in range(num_images):
            image, label = dataset[i]
            input_img = image.unsqueeze(0).to(device)
            output = model(input_img)
            _, predicted = torch.max(output, 1)
            image = image.permute(1, 2, 0).numpy() * 0.5 + 0.5  # Denormalize
            axes[i].imshow(image)
            axes[i].set_title(f"True: {'Cat' if label == 0 else 'Dog'}\nPred: {'Cat' if predicted.item() == 0 else 'Dog'}")
            axes[i].axis("off")
    plt.show()

print("Visualizing sample predictions...")
visualize_predictions(model, test_dataset)
