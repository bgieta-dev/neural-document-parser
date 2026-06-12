from pathlib import Path     
import cv2

class DataLoader:
    def __init__(self):
        self.data_dir = Path("preprocessed_data").resolve()
        self.imgs = self._get_images()
        pass
    
    def _discover_files(self):
        found = []
        for p in self.data_dir.rglob("*"):
            found.append(str(p))
        return found
    
    def _get_images(self):
        paths = self._discover_files()
        imgs = []
        for path in paths:
            img = cv2.imread(path)
            
            imgs.append(img)
                    
        return imgs
                    
maks = DataLoader()