from image_processor import ImageProcessor
from pathlib import Path
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import sys


class DataGetterGUI(QWidget):
    def __init__(self, imgs_data):
        super().__init__()
        self.all_data = imgs_data
        self.current_index = 0
        self.init_ui()

    def normalize_image(self, arr):
        """Basic normalization: ensure 0-255 uint8."""
        if arr.dtype == bool:
            return arr.astype(np.uint8) * 255
        if arr.max() <= 1:
            return (arr * 255).astype(np.uint8)
        return arr.astype(np.uint8)

    def show_image(self):
        if self.current_index < len(self.all_data):
            arr, fname, page, idx = self.all_data[self.current_index]

            # Basic conversion
            display_arr = self.normalize_image(arr)
            # Ensure memory is contiguous for Qt
            display_arr = np.ascontiguousarray(display_arr)

            h, w = display_arr.shape

            # Create QImage from raw data
            q_img = QImage(
                display_arr.data, w, h, w, QImage.Format.Format_Grayscale8
            ).copy()

            self.image_label.setPixmap(QPixmap.fromImage(q_img))
            self.info_label.setText(f"File: {fname} | Page: {page} | Cell: {idx}")
        else:
            self.image_label.setText("Done")
            self.info_label.setText("")

    def init_ui(self):
        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet(
            "border: 1px solid black; background-color: white;"
        )
        layout.addWidget(self.image_label)

        self.info_label = QLabel(self)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.info_label)

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Enter text here...")
        self.input_field.returnPressed.connect(self.on_next_clicked)
        layout.addWidget(self.input_field)

        btn_layout = QHBoxLayout()

        self.skip_button = QPushButton("Skip", self)
        self.skip_button.clicked.connect(self.on_skip_clicked)
        btn_layout.addWidget(self.skip_button)

        self.delete_button = QPushButton("Delete (Bad cell)", self)
        self.delete_button.clicked.connect(self.on_delete_clicked)
        self.delete_button.setStyleSheet("background-color: #ffcccc; color: black;")
        btn_layout.addWidget(self.delete_button)

        self.next_button = QPushButton("Next (Save)", self)
        self.next_button.clicked.connect(self.on_next_clicked)
        self.next_button.setStyleSheet(
            "font-weight: bold; background-color: #ccffcc; color: black;"
        )
        btn_layout.addWidget(self.next_button)

        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.setWindowTitle("Data Getter")
        self.resize(600, 500)
        self.show_image()
        self.input_field.setFocus()

    def on_skip_clicked(self):
        self.input_field.clear()
        self.current_index += 1
        self.show_image()

    def on_delete_clicked(self):
        if self.current_index < len(self.all_data):
            _, fname, page, idx = self.all_data[self.current_index]
            entry = f"{Path(fname).stem}-{page}-{idx}"
            with open("deleted_cells.txt", "a") as f:
                f.write(entry + "\n")
            self.input_field.clear()
            self.current_index += 1
            self.show_image()

    def on_next_clicked(self):
        if self.current_index < len(self.all_data):
            arr, fname, page, idx = self.all_data[self.current_index]
            text = self.input_field.text().strip()

            if text:
                output_dir = Path("preprocessed_data")
                output_dir.mkdir(exist_ok=True)

                # Basic normalization for save
                save_arr = self.normalize_image(arr)

                base_fname = Path(fname).stem
                save_name = f"{base_fname}-{page}-{idx}-{text}.png"
                Image.fromarray(save_arr).save(output_dir / save_name)
                print(f"Saved: {save_name}")

            self.input_field.clear()
            self.current_index += 1
            self.show_image()


class DataGetter:
    def __init__(self, raw_dat_dir):
        self.raw_data_dir = Path(raw_dat_dir).resolve()
        self.output_dir = Path("preprocessed_data")
        self.output_dir.mkdir(exist_ok=True)
        self.file_list = self._discover_files()
        self._run_gui()

    def _discover_files(self):
        found = []
        for p in self.raw_data_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".pdf":
                found.append(str(p))
        return found

    def _is_already_processed(self, fname, page, idx):
        prefix = f"{Path(fname).stem}-{page}-{idx}-"
        for p in self.output_dir.glob(f"{prefix}*.png"):
            return True
        deleted_path = Path("deleted_cells.txt")
        if deleted_path.exists():
            entry = f"{Path(fname).stem}-{page}-{idx}"
            with open(deleted_path, "r") as f:
                if entry in f.read().splitlines():
                    return True
        return False

    def _get_images(self):
        all_data = []
        for path in self.file_list:
            fname = Path(path).name
            imgs = ImageProcessor(path=path).get_cell_images_data_getter()
            for arr, _, page, idx in imgs:
                if not self._is_already_processed(fname, page, idx):
                    all_data.append((arr, fname, page, idx))
        return all_data

    def _run_gui(self):
        app = QApplication(sys.argv)
        data = self._get_images()
        if not data:
            print("Everything is already processed!")
            return
        window = DataGetterGUI(imgs_data=data)
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    DataGetter(raw_dat_dir="raw_data/")
