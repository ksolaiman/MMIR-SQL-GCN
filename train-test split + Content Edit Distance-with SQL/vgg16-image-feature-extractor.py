import os

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T

from PIL import Image

if __name__ == '__main__':
    vgg16_model = models.vgg16(pretrained=True)
    vgg16_model.classifier = nn.Sequential(*list(vgg16_model.classifier.children())[:-1])
    #print(vgg16_model.classifier)

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    with open('image_paths.txt', 'r') as image_file:
        for line in image_file:
            image = Image.open(line.rstrip("\n"))
            image = transform(image)
            batch = torch.unsqueeze(image, 0)
            vgg16_model.eval()
            features = vgg16_model(batch) # [1, 4096] tensor
            # each tensor will be paired with related image and put into json format
    


