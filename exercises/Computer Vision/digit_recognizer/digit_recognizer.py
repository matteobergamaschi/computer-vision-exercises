import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class CustomMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None, is_test=False):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        item = self.data_frame.iloc[index]

        if self.is_test:
            image = item.values.reshape(28, 28).astype(np.uint8)
            label = None
        else:
            image = item[1:].values.reshape(28, 28).astype(np.uint8)
            label = item.iloc[0]
        
        image = transforms.ToPILImage()(image)

        if self.transform is not None:
            image = self.transform(image)
        
        if self.is_test:
            return image
        else:
            return image, label
    
transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = CustomMNISTDataset(csv_file="D:\Matteo\Desktop\Magistrale\Computer Vision\Datasets\digit-recognizer\\train.csv", 
                                   transform=transform, is_test=False)
test_dataset = CustomMNISTDataset(csv_file="D:\Matteo\Desktop\Magistrale\Computer Vision\Datasets\digit-recognizer\\test.csv", 
                                   transform=transform, is_test=True)

#print('Train Size: ' + str(len(train_dataset)) + ', Test Size: ' + str(len(test_dataset)))

batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

'''
for images, labels in train_loader:
    image = images[0]

    image_numpy = image.permute(1,2,0).numpy()

    plt.imshow(image_numpy)
    plt.show()
'''


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 28/2=14 -> 14/2=7: final_shape=(128,7,7) -> flatten

        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool(x)

        x_size = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, x_size)

        # "-1" indicates the batch size auto-completion
        # flatten all the dims

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = CustomCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 1
running_loss = 0.0

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] Loss: {running_loss / 100:.3f}")

print("Training finished")

torch.save(model.state_dict(), 'D:\Matteo\Desktop\Python\Computer Vision\digit_recognizer\\final_model.pth')


model.eval()
predictions = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().tolist())

for i in range(10):
    image = test_dataset[i]
    image_numpy = image.permute(1,2,0).numpy()
    plt.imshow(image_numpy)
    plt.show()
    print(predictions[i])
