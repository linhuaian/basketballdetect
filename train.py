import torch
import clip
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Custom Dataset for CLIP
class VideoDataset(Dataset):
    def __init__(self, data_path, transform):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.transform = transform

    def extract_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Select 5 evenly spaced frames
        frame_idxs = np.linspace(0, total_frames - 1, 5, dtype=int)
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(Image.fromarray(frame)))
        cap.release()
        
        return torch.stack(frames) if frames else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = list(self.data.items())[idx]
        frames = self.extract_frame(video_path)
        if frames is None:
            return None  # Skip invalid videos
        return frames, torch.tensor(label, dtype=torch.long)

# Load data
train_dataset = VideoDataset("dataset_gen/train_data.json", preprocess)
test_dataset = VideoDataset("dataset_gen/test_data.json", preprocess)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

class VideoClassifier(torch.nn.Module): # Simple model with 3 linear layer
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(512, 256)  
        self.fc2 = torch.nn.Linear(256, 128)  
        self.fc3 = torch.nn.Linear(128, 2)  

    def forward(self, x):
        o1 = self.fc(x) 
        o2 = self.fc2(o1)
        o3 = self.fc3(o2)
        return o3

classifier = VideoClassifier().to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# # Training loop
# for epoch in range(20):
#     model.eval()
#     classifier.train()
    
#     total_loss = 0
#     for frames, labels in train_loader:
#         frames, labels = frames.to(device), labels.to(device)
        
#         with torch.no_grad():
#             features = model.encode_image(frames.mean(dim=1))  # Average 5 frames
        
#         outputs = classifier(features)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
    
#     print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# torch.save(classifier, "v1.pth")

classifier = torch.load("v1.pth")

correct = 0
total = 0
classifier.eval()

with torch.no_grad():
    for frames, labels in test_loader:
        frames, labels = frames.to(device), labels.to(device)
        features = model.encode_image(frames.mean(dim=1))
        outputs = classifier(features)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {correct / total:.2%}")