from image_processor import ImageProcessor

if __name__ == "__main__":
    imgs = ImageProcessor("raw_data/3.pdf", DEBUG_MODE=True).get_cell_images()
