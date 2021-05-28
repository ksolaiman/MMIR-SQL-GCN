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

    '''
    # Original Pedestrian Video Transformations 
    transform_train = T.Compose([
            T.Random2DTranslation(args.height, args.width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    transform_test = T.Compose([
            T.Resize((args.height, args.width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    '''

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TODO-1: use the data_transforms (from first commented out section) appropriate for the image/video based on it
    # being training and validation/test data
    # We are using the data_transforms for vgg16 instead of mars as we are extracting pre-trained vectors
    # For video, if C3D has any specific transformation (training and test data may have different transformation), follow that
    
    # TODO-2: read the mgid for training, validation and testing using the
    # "for_METU_read_train_val_test_data.py" file;
    # imagetestset/imagetrainset/ imagevalidset are lists of mgids
    
    # TODO-3: read the corresponding row from mmir_ground table for each mgid

    # TODO-4: save the 4096d vector as value with mgid as key for each sample

    # Suggestion: you can pass batch images to vgg-16 instead of 1 at a time
    # (if you know about it, do it, else it will take more time)
    
    with open('image_paths.txt', 'r') as image_file:
        for line in image_file:
            image = Image.open(line.rstrip("\n"))
            image = transform(image)
            batch = torch.unsqueeze(image, 0)
            vgg16_model.eval()
            features = vgg16_model(batch) # [1, 4096] tensor
            # each tensor will be paired with related image and put into json format



