import torch
import cv2
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset
# import albumentations as A
from torchvision.transforms import transforms, Resize, ToTensor, ToPILImage, Normalize, Grayscale
from glob import glob
import os


class TrainImageDataset(Dataset):
    """
    image format: 512*512
    """
    def __init__(self, input_image_path, target_image_path):
        super().__init__()
        self.input_image_path = sorted(glob(os.path.join(input_image_path, '*')))
        self.target_image_path = sorted(glob(os.path.join(target_image_path, '*'))) 
        # print(self.input_image_path)

        self.transform = transforms.Compose([
            ToPILImage(),
            Resize((128, 128)), 
            Grayscale(num_output_channels = 1),
            ToTensor(),
            Normalize([0.5], [0.1]),
            # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # self.transform = A.Compose([
        #     A.ToTensorV2(),
        #     A.resize(128, 128, p=1, always_apply=True), 
        #     A.OneOf([
        #         A.RandomSizedCrop(min_max_height=(128, 128), height=128, width=128, p=0.5),
        #         A.PadIfNeeded(min_height=128, min_width=128, p=0.5)
        #     ], p=1),
        #     A.VerticalFlip(p=0.5),
        #     A.RandomRotate90(p=0.5),
        #     A.OneOf([
        #         A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        #         A.GridDistortion(p=0.5),
        #         A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
        #         ], p=0.8),
        #     A.CLAHE(p=0.8),
        #     A.RandomBrightnessContrast(p=0.8),
        #     A.RandomGamma(p=0.8),
        #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            
        # ])



    def __getitem__(self, index):
        
        ### use torch transform 
        input_img = cv2.imread(self.input_image_path[index]) # , cv2.IMREAD_GRAYSCALE
        target_img = cv2.imread(self.target_image_path[index]) # , cv2.IMREAD_GRAYSCALE

        # print(input_img.shape)

        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)

        ### use albumentations
        # input_img = Image.open(self.input_image_path[index]).convert('L')
        # target_img = Image.open(self.target_image_path[index]).convert('L')
       
        # transformed = self.transform(image=input_img, mask=target_img)
        # input_tensor = transformed['image']
        # target_tensor = transformed['mask']

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

        self.transform = transforms.Compose([
            ToPILImage(),
            Resize((128, 128)), 
            Grayscale(num_output_channels = 1),
            ToTensor(),
            Normalize([0.5], [0.1]),
            # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # self.transform = A.Compose([
        #     A.resize(128, 128, p=1, always_apply=True), 
        #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     A.ToTensorV2()
        # ])

    def __getitem__(self, index):
        ### use torch transform
        input_img = cv2.imread(self.input_image_path[index], cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(self.target_image_path[index], cv2.IMREAD_GRAYSCALE)
        

        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)

        ### use albumentations
        # input_img = Image.open(self.input_image_path[index]).convert('L')
        # target_img = Image.open(self.target_image_path[index]).convert('L')
       
        # transformed = self.transform(image=input_img, mask=target_img)
        # input_tensor = transformed['image']
        # target_tensor = transformed['mask']

        return input_tensor, target_tensor
    
    def __len__(self):
        return len(self.input_image_path)
