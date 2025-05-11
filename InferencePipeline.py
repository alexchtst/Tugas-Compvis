import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(524288, 512)
        self.bn3 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)

        self.fc5 = nn.Linear(128, 64)
        self.bn7 = nn.BatchNorm1d(64)

        self.fc6 = nn.Linear(64, 32)
        self.bn8 = nn.BatchNorm1d(32)

        self.fc7 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (16, H/2, W/2)
        x = self.pool(F.sigmoid(self.bn2(self.conv2(x))))  # (32, H/4, W/4)

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = F.relu(self.bn5(self.fc3(x)))
        x = F.relu(self.bn6(self.fc4(x)))
        x = F.relu(self.bn7(self.fc5(x)))
        x = F.relu(self.bn8(self.fc6(x)))

        x = self.fc7(x)
        return x

class Inference:
    def __init__(self, model_path, num_classes = 5, device = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('L')
        image = image.resize((512,512))
        image = np.array(image).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        return torch.tensor(image, dtype=torch.float32).to(self.device)
    
    def predict(self, image_path):
        image = self.preprocess_image(image_path)
        with torch.no_grad():
            output = self.model(image)
            probs = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
        return predicted_class.item(), confidence.item()
    
    def predict_and_plot(self, image_path, save_path=None):
        pred_class, confidence = self.predict(image_path)
        image = np.array(Image.open(image_path))
        plt.imshow(image, cmap='gray')
        title = f'Predicted Class : {pred_class}, Confidence : {confidence*100:.2f}%'
        plt.title(title)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"saved plot to {save_path}")
        plt.show()