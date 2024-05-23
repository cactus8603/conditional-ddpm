import torch
# import torch.nn as nn
# import cv2
import numpy as np
import json
from torch.utils.data import Dataset

class TrainImageDataset(Dataset):
    """
    image format: 512*512
    """
    def __init__(self, input_image, target_image):
        super().__init__()
        self.input_image


class ValImageDataset(Dataset):
    def __init__(self, input_image, target_image):
        super().__init__()
        self.input_image