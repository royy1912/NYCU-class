
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData

from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
from torchvision import transforms
import random
class SequencePairedTransform:
    def __init__(self, image_size=(32, 64), flip_prob=0.5, rotation_range=15):
        self.image_size = image_size
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __call__(self, imgs, labels):
        resized_imgs = [transforms.Resize(self.image_size)(img) for img in imgs]
        resized_labels = [transforms.Resize(self.image_size)(label) for label in labels]

        do_flip = random.random() < self.flip_prob
        angle = random.uniform(-self.rotation_range, self.rotation_range)

        if do_flip:
            resized_imgs = [transforms.functional.hflip(img) for img in resized_imgs]
            resized_labels = [transforms.functional.hflip(label) for label in resized_labels]

        rotated_imgs = [transforms.functional.rotate(img, angle) for img in resized_imgs]
        rotated_labels = [transforms.functional.rotate(label, angle) for label in resized_labels]

        jittered_imgs = [self.color_jitter(img) for img in rotated_imgs]
        blurred_imgs = [transforms.GaussianBlur(kernel_size=3)(img) for img in jittered_imgs]

        final_imgs = [transforms.ToTensor()(img) for img in blurred_imgs]
        final_labels = [transforms.ToTensor()(label) for label in rotated_labels]

        return final_imgs, final_labels

def get_key(fp):
    filename = fp.split('/')[-1]
    filename = filename.split('.')[0].replace('frame', '')
    return int(filename)

class Dataset_Dance(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, root, transform, mode='train', video_len=7, partial=1.0):
        super().__init__()
        assert mode in ['train', 'val'], "There is no such mode !!!"
        if mode == 'train':
            self.img_folder     = sorted(glob(os.path.join(root, 'train/train_img/*.png')), key=get_key)
            self.prefix = 'train'
        elif mode == 'val':
            self.img_folder     = sorted(glob(os.path.join(root, 'val/val_img/*.png')), key=get_key)
            self.prefix = 'val'
        else:
            raise NotImplementedError
        
        self.transform = transform
        self.partial = partial
        self.video_len = video_len

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        try:
            imgs = []
            labels = []
            for i in range(self.video_len):
                label_list = self.img_folder[(index * self.video_len) + i].split('/')
                label_list[-2] = self.prefix + '_label'

                img_name = self.img_folder[(index * self.video_len) + i]
                label_name = '/'.join(label_list)

                img = self.transform(imgloader(img_name))
                label = self.transform(imgloader(label_name))
                
                imgs.append(img)
                labels.append(label)

            return stack(imgs), stack(labels)   
        
        except Exception as e:
            print(f"[WARNING] 壞圖跳過：{img_name} 或 {label_name}，錯誤：{str(e)}")
            new_index = (index + 1) % len(self)
            return self.__getitem__(new_index)
    
    

