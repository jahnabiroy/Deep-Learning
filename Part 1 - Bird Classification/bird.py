# Best Accuracy on Test Set : 92.41%

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import csv
import time

LR = 0.0005
EPOCHS = 100
SPLIT = 0.8

torch.manual_seed(0)


class birdClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(birdClassifier, self).__init__()

        # Background removal subnetwork (expanded U-Net-like structure)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        mask = self.decoder(self.encoder(x))
        x = x * mask

        out = self.classifier(x)
        return out


class BirdDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.image_files = []
        self.labels = []
        self.transform = transform

        for label, subdir in enumerate(sorted(os.listdir(data_path))):
            subdir_path = os.path.join(data_path, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        self.image_files.append(os.path.join(subdir_path, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        label = int(os.path.basename(img_path).split("_")[0])

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    model_path,
    epochs=12,
    patience=10,
):
    model.train()
    best_val_loss = float("inf")
    no_improvement_epochs = 0

    for epoch in range(epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train * 100
        print(f"Learning Rate: {scheduler.get_last_lr()[0]}")
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        )

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()

                # Calculate validation accuracy
                _, val_predicted = torch.max(val_outputs, 1)
                correct_val += (val_predicted == val_labels).sum().item()
                total_val += val_labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val * 100
        print(
            f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at {model_path} after epoch {epoch+1}")
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

        model.train()


if __name__ == "__main__":
    dataPath = sys.argv[1]
    trainStatus = sys.argv[2]
    modelPath = sys.argv[3] if len(sys.argv) > 3 else "bird.pth"

    print(f"Training: {trainStatus}")
    print(f"path to dataset: {dataPath}")
    print(f"path to model: {modelPath}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 10
    model = birdClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-7
    )

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]
    )

    testTransform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    dataset = BirdDataset(dataPath, transform=transform)

    X = dataset.image_files
    y = dataset.labels

    train_size = int(SPLIT * len(dataset))
    val_size = int((len(dataset) - train_size))
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    if trainStatus == "train":
        start_time = time.time()
        print("Training...")

        # Train the model
        train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            modelPath,
            epochs=EPOCHS,
            patience=20,
        )
        end_time = time.time()
        print("Final model saved at", modelPath)
        print("Time Taken to Train:", end_time - start_time)

    else:
        print("Inference...")

        test_dataset = BirdDataset(dataPath, transform=testTransform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        model.load_state_dict(torch.load(modelPath))
        model.eval()

        output_csv = "bird.csv"
        with open(output_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Predicted_Label"])

            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                with torch.no_grad():
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)

                for j in range(len(labels)):
                    writer.writerow([predicted[j].item()])
