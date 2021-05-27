import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T

from PIL import Image

if __name__ == '__main__':
    vgg16_model = models.vgg16(pretrained=True)
    vgg16_model.classifier = nn.Sequential(*list(vgg16_model.classifier.children())[:-1])

    '''
    # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
    # https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
    # https://pytorch.org/vision/stable/models.html
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = {
        TRAIN: transforms.Compose([
            # Data augmentation is a good practice for the train set
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally. 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
	    normalize
        ]),
        VAL: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
	    normalize
        ]),
        TEST: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
	    normalize
        ])
    }
    '''

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



