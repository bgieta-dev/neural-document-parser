from image_processor import ImageProcessor
from pathlib import Path     
import numpy as np
from PIL import Image

class DataGetterGUI:
    def __init__(self, raw_data_dir, file_list):
        self._setup_ui()

    
class DataGetter:
    def __init__(self, raw_dat_dir):
        self.raw_data_dir = Path(raw_dat_dir).resolve()
        self.file_list = self._discover_files()
        self._run_gui()

    def _discover_files(self):
        found = []
        for p in self.raw_data_dir.rglob("*"):
            if p.is_file():
                found.append(str(p))
        return found
    
    def _get_images(self):
        for path in self.file_list:
            imgs = ImageProcessor

    def _run_gui(self):
        pass

if __name__ == "__main__":
    DataGetter(raw_dat_dir="raw_data/")
