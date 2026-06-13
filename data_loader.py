from pathlib import Path
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class DigitDataset(Dataset):
    def __init__(self):
        self.data_dir = Path("preprocessed_data").resolve()
        self._get_paths()
        self._get_data_loaders()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        label = int(
            self.paths[idx][self.paths[idx].rfind("-") + 1 : self.paths[idx].rfind(".")]
        )

        img_t = self.transform(img)
        label_t = torch.tensor(label, dtype=torch.long) - 5

        if label < 5 or label > 50:
            return img_t, 5
        return img_t, label_t

    def _discover_files(self):
        found = []
        for p in self.data_dir.rglob("*.png"):
            found.append(str(p))
        return found

    def _get_paths(self):
        self.paths = self._discover_files()

    def _get_data_loaders(self, shuffle=True):
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Resize((128, 128)),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.transform = train_transform
        return torch.utils.data.DataLoader(self, batch_size=32, shuffle=shuffle)


maks = DigitDataset()
