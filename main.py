from image_processor import ImageProcessor
if __name__ == "__main__":
    imgs = ImageProcessor('raw_data/scan1.pdf').get_cell_images()
    #ImageProcessor('raw_data/scan2.pdf')