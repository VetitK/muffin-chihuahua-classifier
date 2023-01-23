from pytorch_lightning import LightningDataModule
import torch
import os
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class MuffinChihuahuaDataset(Dataset):
    def __init__(self,
                 data_dir: str = "data",
                 split: str = "train",
                 transforms=None,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transforms = transforms
        
        if self.split == "train" or split == "val":
            self.all_imgs = os.listdir(os.path.join(self.data_dir, "train"))
            for im in self.all_imgs:
                if im[-3:] != "jpg":
                    self.all_imgs.remove(im)
            # split
            train_amount = int(len(self.all_imgs) * 0.8)
            val_amount = len(self.all_imgs) - train_amount
            self.train_data, self.val_data = random_split(self.all_imgs, [train_amount, val_amount])
            
            
        if split == "test":
            self.all_imgs = os.listdir(os.path.join(self.data_dir, "test"))
            for im in self.all_imgs:
                if im[-3:] != "jpg":
                    self.all_imgs.remove(im)
            
    def __len__(self):
        return len(self.all_imgs)
    
    def __getitem__(self, index):
        # Get image path at current index
        if self.split == "train" or self.split == "val":
            img_path = os.path.join(self.data_dir, "train", self.all_imgs[index])
        if self.split == "test":
            img_path = os.path.join(self.data_dir, "test", self.all_imgs[index])

        img = Image.open(img_path).convert("RGB")
        
        # Transform image. Should be  tensorization, resizing, normalization
        if self.transforms:
            img = self.transforms(img)
            
        if self.split == "train" or self.split == "val":
            if "muffin" in self.all_imgs[index]:
                label = torch.tensor(0)
            else:
                label = torch.tensor(1)
        if self.split == "test":
            if "muffin" in self.all_imgs[index]:
                label = torch.tensor(0)
            else:
                label = torch.tensor(1)
        return img, label

class MuffinChihuahuaDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = "data",
                 batch_size: int = 32,
                 numworkers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.numworkers = numworkers
        self._train_transforms = Compose([Resize((224, 224)),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self._val_transforms = Compose([Resize((224, 224)),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self._test_transforms = Compose([Resize((224, 224)),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MuffinChihuahuaDataset(self.data_dir,
                                                        split="train",
                                                        transforms=self._train_transforms)
            self.val_dataset = MuffinChihuahuaDataset(self.data_dir,
                                                      split="val",
                                                      transforms=self._val_transforms)
        if stage == "test" or stage is None:
            self.test_dataset = MuffinChihuahuaDataset(self.data_dir,
                                                       split="test",
                                                       transforms=self._test_transforms)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.numworkers
                          )
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.numworkers
                          )
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.numworkers
                          )    