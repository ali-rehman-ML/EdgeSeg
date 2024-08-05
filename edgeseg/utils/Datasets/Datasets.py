import os
from PIL import Image
import numpy as np
import cv2
import time

# Transformation functions defined earlier
def to_tensor(image):
    if isinstance(image, np.ndarray):
        img = image / 255.0
    else:
        img = np.array(image) / 255.0
    return img.transpose((2, 0, 1))

def normalize(image, mean, std):
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    return (image - mean) / std

def compose(transforms):
    def apply_transforms(image, label):
        for t in transforms:
            image = t(image)
            label = np.array(label) if isinstance(label, Image.Image) else label
        return image, label
    return apply_transforms

class CityscapesDataset:
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory where leftImg8bit and gtFine are located.
            split (str): Split to use ('train', 'val', 'test').
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.labels_dir = os.path.join(root_dir, 'gtFine', split)
        self.image_paths = self._get_paths(self.images_dir)
        self.label_paths = self._get_paths(self.labels_dir, label=True)

        assert len(self.image_paths) == len(self.label_paths), "Mismatch between image and label count."

    def _get_paths(self, dir_path, label=False):
        """Helper function to get image or label paths."""
        file_paths = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if label and file.endswith('_gtFine_labelIds.png'):
                    file_paths.append(os.path.join(root, file))
                elif not label and file.endswith('_leftImg8bit.png'):
                    file_paths.append(os.path.join(root, file))
        return sorted(file_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        else:
            image_path = self.image_paths[idx]
            label_path = self.label_paths[idx]

            image = Image.open(image_path).convert('RGB')
            label = Image.open(label_path)

            # Apply transform, if provided
            if self.transform:
                image, label = self.transform(image, label)

            return image, label



def torchvision_cityscapes(dir='./',split='val',transforms=None):
    import torchvision
    dataset=torchvision.datasets.Cityscapes(root=dir,split=split,target_type='semantic',transform=transforms)
    return dataset

def Cityscapes(type='torchvision',split='val',dir='./',transforms=None):
    if type=='torchvision':
        return torchvision_cityscapes(root=dir,split=split,target_type='semantic')
    if type=='numpy':
        return CityscapesDataset(root_dir=dir, split=split, transform=transforms)
        


    


