import torch
import cv2
import numpy as np
import json
from torch.utils.data import Dataset
import albumentations as A
from glob import glob
import os

from torchvision.transforms import transforms
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize

class TrainImageDataset(Dataset):
    """
    image format: 512*512
    """
    def __init__(self, input_image_path, target_image_path):
        super().__init__()
        self.input_image_path = sorted(glob(os.path.join(input_image_path, '*')))
        self.target_image_path = sorted(glob(os.path.join(target_image_path, '*'))) 

        self.transform = A.Compose([
            A.resize(256, 256), 
            A.OneOf([
                A.RandomSizedCrop(min_max_height=(128, 128), height=256, width=256, p=0.5),
                A.PadIfNeeded(min_height=256, min_width=256, p=0.5)
            ], p=1),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
                ], p=0.8),
            A.CLAHE(p=0.8),
            A.RandomBrightnessContrast(p=0.8),
            A.RandomGamma(p=0.8),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2()
        ])

    def __getitem__(self, index):
        input_img = cv2.imread(self.input_image_path[index], cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(self.target_image_path[index], cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=input_img, mask=target_img)
        input_tensor = transformed['image']
        target_tensor = transformed['mask']

        return input_tensor, target_tensor
    
    def __len__(self):
        return len(self.input_image_path)




class ValImageDataset(Dataset):
    """
    image format: 512*512
    """
    def __init__(self, input_image_path, target_image_path):
        super().__init__()
        self.input_image_path = sorted(glob(os.path.join(input_image_path, '*')))
        self.target_image_path = sorted(glob(os.path.join(target_image_path, '*'))) 

        self.transform = A.Compose([
            A.resize(256, 256), 
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2()
        ])

    def __getitem__(self, index):
        input_img = cv2.imread(self.input_image_path[index], cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(self.target_image_path[index], cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=input_img, mask=target_img)
        input_tensor = transformed['image']
        target_tensor = transformed['mask']

        return input_tensor, target_tensor
    
    def __len__(self):
        return len(self.input_image_path)
