
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cpu')

save_path = 'face_gender_classification_transfer_learning_with_ResNet18.pth'

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) # binary classification (num_of_class == 2)
model.load_state_dict(torch.load(save_path))
model.to(device)

df = pd.read_csv('./cache/output/for_DL.csv',index_col=0)


root = './data/images'
i=0

transforms_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # data augmentation every epoch you are using a different dataset, 可以理解为increase the data size
    transforms.ToTensor(),

    # transforms.Lambda(lambda x: x.unsqueeze(0))
])

transforms_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization pretrained on Imagenet

label_list = []
model.eval()
for i in range(len(df)):
    dir = os.path.join(root,str(i)+'.jpg')
    try:
        img = Image.open(dir)
        input = transforms_pipeline(img)
        if input.shape[0]==1:
            input = torch.cat([input, input, input], dim=0)
        elif input.shape[0]>3:
            input = input[:3]
        input = transforms_norm(input)
        input = torch.unsqueeze(input,dim=0)
        input = input.to(device)
        outputs = model(input)
        _, preds = torch.max(outputs, 1)
        preds = preds.item()
    except FileNotFoundError: # the link is not available
        preds = ''
    label_list.append(preds)

plt.imshow(img)

df['gender_from_resnet'] = label_list

df['gender_from_resnet'].value_counts()

df.reset_index(drop=True,inplace=True)











