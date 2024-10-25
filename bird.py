import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv

# CNN Model for bird classification
class BirdClassifier(nn.Module):
    def __init__(self):
        super(BirdClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # Output 10 classes
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Testing function
def test_model(model, test_loader, criterion, device, output_csv='bird.csv'):
    model.eval()
    results = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            results.extend(predicted.cpu().numpy())
    
    # Write predicted labels to a CSV file
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted_Label'])
        for label in results:
            writer.writerow([label])

# Main script logic
if __name__ == "__main__":
    dataPath = sys.argv[1]
    trainStatus = sys.argv[2]
    modelPath = sys.argv[3] if len(sys.argv) > 3 else "default/model/path"

    batch_size = 1
    shuffle = False

    # Example dataset loading (replace with actual dataset logic)
    # Assuming 'test_dataset' is defined properly
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    # Check if we are training or testing
    if trainStatus == "train":
        print("Training mode")
        # Add training logic here...
    else:
        print("Inference mode")
        # Load the model for inference
        model = BirdClassifier()
        model.load_state_dict(torch.load(modelPath))  # Assuming model saved path
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        criterion = nn.CrossEntropyLoss()
        test_model(model, test_loader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training: {trainStatus}")
    print(f"path to dataset: {dataPath}")
    print(f"path to model: {modelPath}")
