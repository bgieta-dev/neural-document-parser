from image_processor import ImageProcessor
from pathlib import Path     
import tkinter as tk
from tkinter import messagebox, Text, Canvas, Button
import numpy as np
from PIL import Image, ImageTk

class DataGetterGUI:
    def __init__(self, root, file_list):
        self.root = root
        self.root.title("Template Reader Viewer")
        self.file_list = file_list
        self.current_index = 0
        self.tk_img = None  # Critical: hold reference to prevent GC

        self._setup_ui()

    def _setup_ui(self):
        # Image Display (Canvas) - Fixed at 128x128
        self.canvas = Canvas(self.root, width=128, height=128)
        self.canvas.pack(side="left", before=tk.Label(self.root))

        # Text Area (for numbers)
        self.text_area = Text(self.root, wrap="char", width=30, height=150)
        self.text_area.pack(side="bottom", expand=True)
        self.text_area.insert("1.0", "")

        # Next Button
        self.next_btn = Button(self.root, text="Next", command=self.load_next)
        self.next_btn.pack(side="top")

    def load_next(self):
        if self.current_index < len(self.file_list):
            filename = self.file_list[self.current_index]
            try:
                # Get images from your existing processor
                imgs = ImageProcessor(path=filename).get_cell_images()
                # Extract the 128x128 array (based on user's print(imgs[1][0]))
                img_array = imgs[1][0]
                self.update_image(img_array)
                self.current_index += 1
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                self.current_index += 1
        else:
            messagebox.showinfo("End of files", "No more images to display.")

    def update_image(self, array):
        # Ensure it is exactly 128x128 for the canvas
        if array.shape[0] != 128 or array.shape[1] != 128:
            img_pil = Image.fromarray(array)
            img_pil = img_pil.resize((128, 128))
            array = np.array(img_pil)

        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(array))
        # Refresh canvas drawing
        try:
            self.canvas.delete("all")
            self.canvas = Canvas(self.root, width=128, height=128)
            self.canvas.pack(side="left", before=tk.Label(self.root))
        except: 
            pass # Fallback if canvas is already managed

        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

class DataGetter:
    def __init__(self, root_dir):
        self.root = Path(root_dir).resolve()
        self.file_list = self._discover_files()
        self._run_gui()

    def _discover_files(self):
        found = []
        for p in self.root.rglob("*"):
            if p.is_file():
                found.append(str(p))
        return found

    def _run_gui(self):
        self.root_gui = tk.Tk()
        app = DataGetterGUI(self.root_gui, self.file_list)
        app.load_next()
        self.root_gui.mainloop()

if __name__ == "__main__":
    DataGetter(root_dir="raw_data/")
