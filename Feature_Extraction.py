import os
import openslide
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import h5py  # Import the h5py library
from torchvision.models import resnet18

svs_folder = "adeno_tcga"
csv_file = "tcga_bal_158_modified.csv"

patch_size = 512

df = pd.read_csv(csv_file)

class SVSPatchDataset(Dataset):
    def __init__(self, data_frame, root_dir, patch_size, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        slide = openslide.open_slide(img_name)
        image = np.array(slide.read_region((0, 0), 0, slide.dimensions).convert("RGB"))

        label = int(self.data_frame.iloc[idx, 1])

        patches = []
        for i in range(0, image.shape[0], self.patch_size):
            for j in range(0, image.shape[1], self.patch_size):
                patch = image[i:i + self.patch_size, j:j + self.patch_size, :]
                patches.append(patch)

        if self.transform:
            patches = [self.transform(patch) for patch in patches]

        return patches, label
import torch.utils.data as data
from PIL import Image
import os
import pandas as pd
import numpy as np
import random
from openslide import open_slide
from torchvision import datasets
from torchvision import transforms
import torchvision.transforms.functional as F
import torch
import time
# from histomicstk.preprocessing.color_normalization import reinhard

class GetLoader(data.Dataset):
    def __init__(self, csv_path, folder):

        self.csv_path=csv_path
        self.folder = folder
        self.df=pd.read_csv(self.csv_path)

#        
    def __getitem__(self, item):
        slide_name=self.df.loc[item,'slide_id']
        labels=self.df.loc[item,'label']
        image='adeno_tcga/'+slide_name
#         image=self.df.loc[item,'full_path']
        slide_name='adenoCoordfilt2/'+ slide_name+'.csv'
#         slide_name=slide_name.replace('adeno_tcga/','',1)
    
        slide_name=slide_name.replace('.svs','',1)
#         file_path = os.path.join(self.folder, slide_name)
        file_path=slide_name
#         print(file_path)
        if labels==1:
            label=1
        else:
            label=0
        df=pd.read_csv(file_path)
        coordx=np.array(df['dim1'])
        coordy=np.array(df['dim2'])
        seed_value = 42
        top_indices = df['count'].nlargest(1).index
        random.seed(seed_value)
        start_time = time.time()
        wsi= open_slide(image)
        patches = []
#         trans = transforms.Compose([transforms.ToTensor()])
#         trans = transforms.Compose([transforms.Resize(128),transforms.ToTensor(),transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
        trans = transforms.Compose([
                transforms.ToPILImage(),\
                transforms.ColorJitter(brightness=0,\
                                         contrast=0,\
                                         saturation=0.5,\
                                         hue=[-0.1, 0.1]),\
                transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(30),transforms.RandomRotation(60), transforms.RandomRotation(90), transforms.RandomRotation(180), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                 std=[0.229, 0.224, 0.225])
            ])

        for i in range(len(top_indices)):
            patch=wsi.read_region((coordx[top_indices[i]],coordy[top_indices[i]]), 0, (512,512)) 
            patch = patch.convert('RGB')
            patch = F.equalize(patch)
            patch = np.array(patch)
#             patch = torch.tensor(patch, dtype=torch.float32, requires_grad=True)
            patch=trans(patch)
            patches.append(patch)
#         patches = torch.tensor(patches, requires_grad=True)
        patches = torch.stack(patches, dim=0)
        return patches, label
    
    def __len__(self):
        return len(self.df)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

custom_dataset = GetLoader(csv_file, svs_folder)

dataloader = DataLoader(dataset=custom_dataset, batch_size=1, shuffle=True)

model = resnet18(pretrained=True)
num_classes = 2
# print(model.classifier[3].in_features)

model.classifier = nn.Sequential(
    nn.Dropout(0.2)  # Add dropout if needed
#     nn.Linear(model.classifier[3].in_features, num_classes),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model on patches
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model.to(device)
print(1)
hdf5_file_path = "features.h5"

hdf5_file = h5py.File(hdf5_file_path, 'w')

file_names_dataset = hdf5_file.create_dataset('file_names', (len(df),), dtype='S50')
features_dataset = hdf5_file.create_dataset('features', (len(df), 20,512), dtype='f')
print(len(df))
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    for i, (patches, labels) in enumerate(dataloader):
#         print(labels)
        patches = torch.unbind(patches)
        patches = torch.stack(patches, dim=0).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(patches.view(-1, 3, 512, 512))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        epoch_features = []
        for i, (patches, labels) in enumerate(dataloader):
            patches = torch.unbind(patches)
            patches = torch.stack(patches, dim=0).to(device)
            labels = labels.to(device)

            features = model(patches.view(-1, 3, 512, 512))
            epoch_features.append(features.cpu().numpy())

            file_names_dataset[i] = df.iloc[i, 0].encode('utf-8')

        epoch_features = np.concatenate(epoch_features, axis=0)
        features_dataset[:, epoch, :] = epoch_features

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

hdf5_file.close()

test_dataloader = DataLoader(dataset=custom_dataset, batch_size=1, shuffle=False)

model.eval()

correct_predictions = 0
total_samples = len(test_dataloader.dataset)

with torch.no_grad():
    for patches, labels in test_dataloader:
        patches = torch.unbind(patches)
        patches = torch.stack(patches, dim=0).to(device)
        labels = labels.to(device)

        outputs = model(patches.view(-1, 3, 512, 512))
        _, predicted = torch.max(outputs, 1)

        correct_predictions += (predicted == labels).sum().item()

accuracy = correct_predictions / total_samples
print(f"Accuracy: {accuracy * 100:.2f}%")