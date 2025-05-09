import os
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2  # 用于频域转换

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2

class ForgeryDataset(Dataset):
    def __init__(self, img_paths, labels, use_freq=False, transform=None):
        '''
        img_paths:数据路径
        labels:真假标签
        use_freq:是否使用频域变换
        transform:变换方式
        '''
        self.img_paths = img_paths
        self.labels = labels
        self.use_freq = use_freq
        self.transform = transform

    @staticmethod
    def example_gen(root_dir, use_freq=False, transform=None):
        """
        root_dir: 形如 "data/archive/train"
        自动读取 real 和 fake 子文件夹
        """
        img_paths = []
        labels = []

        for label_name in ['real', 'fake']:
            class_dir = os.path.join(root_dir, label_name)
            label = 1 if label_name == 'real' else 0
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_paths.append(os.path.join(class_dir, fname))
                    labels.append(label)

        return ForgeryDataset(img_paths, labels, use_freq, transform)

    def __len__(self):
        return len(self.img_paths)

    def rgb_to_freq(self, img):
        img_np = np.array(img.convert('L'))
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        magnitude_spectrum = magnitude_spectrum.astype(np.float32)
        magnitude_spectrum = cv2.resize(magnitude_spectrum, img.size)
        magnitude_spectrum = torch.tensor(magnitude_spectrum).unsqueeze(0)
        return magnitude_spectrum

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        rgb_tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)

        if self.use_freq:
            freq_tensor = self.rgb_to_freq(img)
            combined = torch.cat([rgb_tensor, freq_tensor], dim=0)
            return combined, label
        else:
            return rgb_tensor, label



if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = ForgeryDataset.example_gen(
        "data/archive/train",
        use_freq=False,
        transform=transform
    )

    val_dataset = ForgeryDataset.example_gen(
        "data/archive/valid",
        use_freq=False,
        transform=transform
    )

    test_dataset = ForgeryDataset.example_gen(
        "data/archive/test",
        use_freq=False,
        transform=transform
    )

    dataloader = DataLoader(
        train_dataset,   
        batch_size=32, 
        shuffle=True,   # 打乱数据
        num_workers=4,  # 多线程读取，加快速度（可根据电脑性能调整）
        pin_memory=True # 若用 GPU，可加速数据搬运
    )

    print(len(train_dataset))
    print(len(test_dataset))
    print(len(val_dataset))

    print(len(dataloader))
